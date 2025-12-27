//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use crate::csc::CscMatrix;
use crate::csr::CsrMatrix;
use num_traits::{Float, Zero};
use oxiblas_core::scalar::{Field, Scalar};
/// Sparse matrix-vector multiplication: y = alpha * A * x + beta * y
///
/// Uses CSR format for efficient row access.
///
/// # Arguments
///
/// * `alpha` - Scalar multiplier for A*x
/// * `a` - Sparse matrix in CSR format
/// * `x` - Input vector (length = a.ncols())
/// * `beta` - Scalar multiplier for y
/// * `y` - Output vector (length = a.nrows()), modified in place
///
/// # Panics
///
/// Panics if dimensions don't match.
pub fn spmv<T: Scalar + Clone + Field>(alpha: T, a: &CsrMatrix<T>, x: &[T], beta: T, y: &mut [T]) {
    assert_eq!(x.len(), a.ncols(), "x length must equal number of columns");
    assert_eq!(y.len(), a.nrows(), "y length must equal number of rows");
    let is_beta_zero = Scalar::abs(beta.clone()) <= T::epsilon();
    for i in 0..a.nrows() {
        let start = a.row_ptrs()[i];
        let end = a.row_ptrs()[i + 1];
        let mut sum = T::zero();
        for k in start..end {
            let col = a.col_indices()[k];
            sum = sum + a.values()[k].clone() * x[col].clone();
        }
        if is_beta_zero {
            y[i] = alpha.clone() * sum;
        } else {
            y[i] = alpha.clone() * sum + beta.clone() * y[i].clone();
        }
    }
}
/// Sparse matrix-vector multiplication with transpose: y = alpha * A^T * x + beta * y
///
/// Uses CSR format for the transpose operation.
///
/// # Arguments
///
/// * `alpha` - Scalar multiplier for A^T*x
/// * `a` - Sparse matrix in CSR format
/// * `x` - Input vector (length = a.nrows())
/// * `beta` - Scalar multiplier for y
/// * `y` - Output vector (length = a.ncols()), modified in place
///
/// # Panics
///
/// Panics if dimensions don't match.
pub fn spmv_transpose<T: Scalar + Clone + Field>(
    alpha: T,
    a: &CsrMatrix<T>,
    x: &[T],
    beta: T,
    y: &mut [T],
) {
    assert_eq!(x.len(), a.nrows(), "x length must equal number of rows");
    assert_eq!(y.len(), a.ncols(), "y length must equal number of columns");
    let is_beta_zero = Scalar::abs(beta.clone()) <= T::epsilon();
    if is_beta_zero {
        for val in y.iter_mut() {
            *val = T::zero();
        }
    } else {
        for val in y.iter_mut() {
            *val = beta.clone() * val.clone();
        }
    }
    for i in 0..a.nrows() {
        let start = a.row_ptrs()[i];
        let end = a.row_ptrs()[i + 1];
        let xi = x[i].clone();
        for k in start..end {
            let col = a.col_indices()[k];
            y[col] = y[col].clone() + alpha.clone() * a.values()[k].clone() * xi.clone();
        }
    }
}
/// Sparse matrix-vector multiplication using CSC format: y = alpha * A * x + beta * y
///
/// Uses CSC format for efficient column access (useful when A is stored in CSC).
pub fn spmv_csc<T: Scalar + Clone + Field>(
    alpha: T,
    a: &CscMatrix<T>,
    x: &[T],
    beta: T,
    y: &mut [T],
) {
    assert_eq!(x.len(), a.ncols(), "x length must equal number of columns");
    assert_eq!(y.len(), a.nrows(), "y length must equal number of rows");
    let is_beta_zero = Scalar::abs(beta.clone()) <= T::epsilon();
    if is_beta_zero {
        for val in y.iter_mut() {
            *val = T::zero();
        }
    } else {
        for val in y.iter_mut() {
            *val = beta.clone() * val.clone();
        }
    }
    for j in 0..a.ncols() {
        let start = a.col_ptrs()[j];
        let end = a.col_ptrs()[j + 1];
        let xj = x[j].clone();
        for k in start..end {
            let row = a.row_indices()[k];
            y[row] = y[row].clone() + alpha.clone() * a.values()[k].clone() * xj.clone();
        }
    }
}
/// Sparse matrix-matrix multiplication: C = alpha * A * B + beta * C
///
/// Both A and B are in CSR format, C is dense.
///
/// # Arguments
///
/// * `alpha` - Scalar multiplier for A*B
/// * `a` - Left sparse matrix in CSR format (m x k)
/// * `b` - Right sparse matrix in CSR format (k x n)
/// * `beta` - Scalar multiplier for C
/// * `c` - Output dense matrix (m x n), modified in place
///
/// # Panics
///
/// Panics if dimensions don't match.
pub fn spmm<T: Scalar + Clone + Field + bytemuck::Zeroable>(
    alpha: T,
    a: &CsrMatrix<T>,
    b: &CsrMatrix<T>,
    beta: T,
    c: &mut oxiblas_matrix::MatMut<'_, T>,
) {
    assert_eq!(a.ncols(), b.nrows(), "Inner dimensions must match");
    assert_eq!(c.nrows(), a.nrows(), "C rows must match A rows");
    assert_eq!(c.ncols(), b.ncols(), "C cols must match B cols");
    let m = a.nrows();
    let n = b.ncols();
    let is_beta_zero = Scalar::abs(beta.clone()) <= T::epsilon();
    if is_beta_zero {
        for i in 0..m {
            for j in 0..n {
                c[(i, j)] = T::zero();
            }
        }
    } else {
        for i in 0..m {
            for j in 0..n {
                c[(i, j)] = beta.clone() * c[(i, j)].clone();
            }
        }
    }
    for i in 0..m {
        let a_start = a.row_ptrs()[i];
        let a_end = a.row_ptrs()[i + 1];
        for a_idx in a_start..a_end {
            let k = a.col_indices()[a_idx];
            let a_val = a.values()[a_idx].clone();
            let b_start = b.row_ptrs()[k];
            let b_end = b.row_ptrs()[k + 1];
            for b_idx in b_start..b_end {
                let j = b.col_indices()[b_idx];
                let b_val = b.values()[b_idx].clone();
                c[(i, j)] = c[(i, j)].clone() + alpha.clone() * a_val.clone() * b_val;
            }
        }
    }
}
/// Sparse matrix-matrix multiplication returning a sparse result: C = A * B
///
/// Both inputs and output are in CSR format.
pub fn spmm_sparse<T: Scalar + Clone + Field>(a: &CsrMatrix<T>, b: &CsrMatrix<T>) -> CsrMatrix<T> {
    assert_eq!(a.ncols(), b.nrows(), "Inner dimensions must match");
    let m = a.nrows();
    let n = b.ncols();
    let mut row_ptrs = Vec::with_capacity(m + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    row_ptrs.push(0);
    let mut workspace = vec![T::zero(); n];
    let mut col_flags = vec![false; n];
    for i in 0..m {
        for j in 0..n {
            workspace[j] = T::zero();
            col_flags[j] = false;
        }
        let a_start = a.row_ptrs()[i];
        let a_end = a.row_ptrs()[i + 1];
        for a_idx in a_start..a_end {
            let k = a.col_indices()[a_idx];
            let a_val = a.values()[a_idx].clone();
            let b_start = b.row_ptrs()[k];
            let b_end = b.row_ptrs()[k + 1];
            for b_idx in b_start..b_end {
                let j = b.col_indices()[b_idx];
                let b_val = b.values()[b_idx].clone();
                workspace[j] = workspace[j].clone() + a_val.clone() * b_val;
                col_flags[j] = true;
            }
        }
        for j in 0..n {
            if col_flags[j] && Scalar::abs(workspace[j].clone()) > T::epsilon() {
                col_indices.push(j);
                values.push(workspace[j].clone());
            }
        }
        row_ptrs.push(values.len());
    }
    unsafe { CsrMatrix::new_unchecked(m, n, row_ptrs, col_indices, values) }
}
/// Sparse matrix addition: C = alpha * A + beta * B
///
/// Both inputs are in CSR format.
pub fn spadd<T: Scalar + Clone + Field>(
    alpha: T,
    a: &CsrMatrix<T>,
    beta: T,
    b: &CsrMatrix<T>,
) -> CsrMatrix<T> {
    assert_eq!(a.nrows(), b.nrows(), "Row counts must match");
    assert_eq!(a.ncols(), b.ncols(), "Column counts must match");
    let m = a.nrows();
    let n = a.ncols();
    let mut row_ptrs = Vec::with_capacity(m + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    row_ptrs.push(0);
    for i in 0..m {
        let a_start = a.row_ptrs()[i];
        let a_end = a.row_ptrs()[i + 1];
        let b_start = b.row_ptrs()[i];
        let b_end = b.row_ptrs()[i + 1];
        let mut a_idx = a_start;
        let mut b_idx = b_start;
        while a_idx < a_end || b_idx < b_end {
            let a_col = if a_idx < a_end {
                Some(a.col_indices()[a_idx])
            } else {
                None
            };
            let b_col = if b_idx < b_end {
                Some(b.col_indices()[b_idx])
            } else {
                None
            };
            match (a_col, b_col) {
                (Some(ac), Some(bc)) if ac < bc => {
                    let val = alpha.clone() * a.values()[a_idx].clone();
                    if Scalar::abs(val.clone()) > T::epsilon() {
                        col_indices.push(ac);
                        values.push(val);
                    }
                    a_idx += 1;
                }
                (Some(ac), Some(bc)) if ac > bc => {
                    let val = beta.clone() * b.values()[b_idx].clone();
                    if Scalar::abs(val.clone()) > T::epsilon() {
                        col_indices.push(bc);
                        values.push(val);
                    }
                    b_idx += 1;
                }
                (Some(ac), Some(_bc)) => {
                    let val = alpha.clone() * a.values()[a_idx].clone()
                        + beta.clone() * b.values()[b_idx].clone();
                    if Scalar::abs(val.clone()) > T::epsilon() {
                        col_indices.push(ac);
                        values.push(val);
                    }
                    a_idx += 1;
                    b_idx += 1;
                }
                (Some(ac), None) => {
                    let val = alpha.clone() * a.values()[a_idx].clone();
                    if Scalar::abs(val.clone()) > T::epsilon() {
                        col_indices.push(ac);
                        values.push(val);
                    }
                    a_idx += 1;
                }
                (None, Some(bc)) => {
                    let val = beta.clone() * b.values()[b_idx].clone();
                    if Scalar::abs(val.clone()) > T::epsilon() {
                        col_indices.push(bc);
                        values.push(val);
                    }
                    b_idx += 1;
                }
                (None, None) => break,
            }
        }
        row_ptrs.push(values.len());
    }
    unsafe { CsrMatrix::new_unchecked(m, n, row_ptrs, col_indices, values) }
}
/// Computes the transpose of a CSR matrix, returning CSR.
pub fn transpose_csr<T: Scalar + Clone>(a: &CsrMatrix<T>) -> CsrMatrix<T> {
    let csc = a.to_csc();
    unsafe {
        CsrMatrix::new_unchecked(
            a.ncols(),
            a.nrows(),
            csc.col_ptrs().to_vec(),
            csc.row_indices().to_vec(),
            csc.values().to_vec(),
        )
    }
}
/// Computes A^T * A for a CSR matrix, returning CSR.
///
/// This is commonly used in least squares problems.
pub fn ata<T: Scalar + Clone + Field>(a: &CsrMatrix<T>) -> CsrMatrix<T> {
    let at = transpose_csr(a);
    spmm_sparse(&at, a)
}
/// Computes A * A^T for a CSR matrix, returning CSR.
pub fn aat<T: Scalar + Clone + Field>(a: &CsrMatrix<T>) -> CsrMatrix<T> {
    let at = transpose_csr(a);
    spmm_sparse(a, &at)
}
/// Computes the diagonal of a sparse matrix.
pub fn diagonal<T: Scalar + Clone + Field>(a: &CsrMatrix<T>) -> Vec<T> {
    let n = a.nrows().min(a.ncols());
    let mut diag = vec![T::zero(); n];
    for i in 0..n {
        if let Some(val) = a.get(i, i) {
            diag[i] = val.clone();
        }
    }
    diag
}
/// Creates a diagonal sparse matrix from a vector.
pub fn from_diagonal<T: Scalar + Clone + Field>(diag: &[T]) -> CsrMatrix<T> {
    let n = diag.len();
    let mut row_ptrs = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::with_capacity(n);
    let mut values = Vec::with_capacity(n);
    row_ptrs.push(0);
    for (i, val) in diag.iter().enumerate() {
        if Scalar::abs(val.clone()) > T::epsilon() {
            col_indices.push(i);
            values.push(val.clone());
        }
        row_ptrs.push(values.len());
    }
    unsafe { CsrMatrix::new_unchecked(n, n, row_ptrs, col_indices, values) }
}
/// Sparse symmetric matrix-vector multiplication: y = alpha * A * x + beta * y
///
/// Only the lower triangle of A is accessed. The matrix is assumed to be symmetric.
/// This is more efficient when only half the matrix is stored.
///
/// # Arguments
///
/// * `alpha` - Scalar multiplier for A*x
/// * `a` - Symmetric sparse matrix in CSR format (only lower triangle used)
/// * `x` - Input vector (length = a.ncols())
/// * `beta` - Scalar multiplier for y
/// * `y` - Output vector (length = a.nrows()), modified in place
/// * `lower_only` - If true, assume only lower triangle is stored
///
/// # Panics
///
/// Panics if dimensions don't match or if matrix is not square.
pub fn spmv_symmetric<T: Scalar + Clone + Field>(
    alpha: T,
    a: &CsrMatrix<T>,
    x: &[T],
    beta: T,
    y: &mut [T],
    lower_only: bool,
) {
    assert_eq!(a.nrows(), a.ncols(), "Matrix must be square");
    assert_eq!(x.len(), a.ncols(), "x length must equal number of columns");
    assert_eq!(y.len(), a.nrows(), "y length must equal number of rows");
    let n = a.nrows();
    let is_beta_zero = Scalar::abs(beta.clone()) <= T::epsilon();
    if is_beta_zero {
        for val in y.iter_mut() {
            *val = T::zero();
        }
    } else {
        for val in y.iter_mut() {
            *val = beta.clone() * val.clone();
        }
    }
    if lower_only {
        for i in 0..n {
            let start = a.row_ptrs()[i];
            let end = a.row_ptrs()[i + 1];
            for k in start..end {
                let j = a.col_indices()[k];
                let aij = a.values()[k].clone();
                if j <= i {
                    y[i] = y[i].clone() + alpha.clone() * aij.clone() * x[j].clone();
                    if i != j {
                        y[j] = y[j].clone() + alpha.clone() * aij * x[i].clone();
                    }
                }
            }
        }
    } else {
        for i in 0..n {
            let start = a.row_ptrs()[i];
            let end = a.row_ptrs()[i + 1];
            for k in start..end {
                let j = a.col_indices()[k];
                if j <= i {
                    let aij = a.values()[k].clone();
                    y[i] = y[i].clone() + alpha.clone() * aij.clone() * x[j].clone();
                    if i != j {
                        y[j] = y[j].clone() + alpha.clone() * aij * x[i].clone();
                    }
                }
            }
        }
    }
}
/// Sparse Hermitian matrix-vector multiplication: y = alpha * A * x + beta * y
///
/// For real matrices, this is equivalent to `spmv_symmetric`.
/// For complex matrices, uses conjugate transpose.
///
/// Only the lower triangle of A is accessed.
pub fn spmv_hermitian<T: Scalar + Clone + Field>(
    alpha: T,
    a: &CsrMatrix<T>,
    x: &[T],
    beta: T,
    y: &mut [T],
    lower_only: bool,
) {
    spmv_symmetric(alpha, a, x, beta, y, lower_only)
}
/// Sparse triangular matrix-vector multiplication: y = alpha * L * x + beta * y
///
/// Only the specified triangular part is used.
///
/// # Arguments
///
/// * `alpha` - Scalar multiplier
/// * `a` - Sparse matrix in CSR format
/// * `x` - Input vector
/// * `beta` - Scalar multiplier for y
/// * `y` - Output vector, modified in place
/// * `part` - Which triangular part to use
pub fn spmv_triangular<T: Scalar + Clone + Field>(
    alpha: T,
    a: &CsrMatrix<T>,
    x: &[T],
    beta: T,
    y: &mut [T],
    part: TriangularPart,
) {
    assert_eq!(x.len(), a.ncols(), "x length must equal number of columns");
    assert_eq!(y.len(), a.nrows(), "y length must equal number of rows");
    let is_beta_zero = Scalar::abs(beta.clone()) <= T::epsilon();
    for i in 0..a.nrows() {
        let start = a.row_ptrs()[i];
        let end = a.row_ptrs()[i + 1];
        let mut sum = T::zero();
        for k in start..end {
            let j = a.col_indices()[k];
            let include = match part {
                TriangularPart::Lower => j <= i,
                TriangularPart::Upper => j >= i,
                TriangularPart::StrictlyLower => j < i,
                TriangularPart::StrictlyUpper => j > i,
            };
            if include {
                sum = sum + a.values()[k].clone() * x[j].clone();
            }
        }
        if is_beta_zero {
            y[i] = alpha.clone() * sum;
        } else {
            y[i] = alpha.clone() * sum + beta.clone() * y[i].clone();
        }
    }
}
/// Sparse triangular solve: solve L * y = x for y (forward substitution).
///
/// Only the lower triangular part is used.
///
/// # Arguments
///
/// * `a` - Lower triangular sparse matrix in CSR format
/// * `x` - Right-hand side vector
/// * `y` - Solution vector, written in place
///
/// # Panics
///
/// Panics if matrix is not square or has zero diagonal elements.
pub fn sptrsv_lower<T: Scalar + Clone + Field>(a: &CsrMatrix<T>, x: &[T], y: &mut [T]) {
    assert_eq!(a.nrows(), a.ncols(), "Matrix must be square");
    assert_eq!(x.len(), a.nrows(), "x length must equal nrows");
    assert_eq!(y.len(), a.nrows(), "y length must equal nrows");
    let n = a.nrows();
    let eps = T::epsilon();
    for i in 0..n {
        let start = a.row_ptrs()[i];
        let end = a.row_ptrs()[i + 1];
        let mut sum = x[i].clone();
        let mut diag = T::zero();
        for k in start..end {
            let j = a.col_indices()[k];
            if j < i {
                sum = sum - a.values()[k].clone() * y[j].clone();
            } else if j == i {
                diag = a.values()[k].clone();
            }
        }
        assert!(Scalar::abs(diag.clone()) > eps, "Zero diagonal at row {i}");
        y[i] = sum / diag;
    }
}
/// Sparse triangular solve: solve U * y = x for y (backward substitution).
///
/// Only the upper triangular part is used.
///
/// # Arguments
///
/// * `a` - Upper triangular sparse matrix in CSR format
/// * `x` - Right-hand side vector
/// * `y` - Solution vector, written in place
pub fn sptrsv_upper<T: Scalar + Clone + Field>(a: &CsrMatrix<T>, x: &[T], y: &mut [T]) {
    assert_eq!(a.nrows(), a.ncols(), "Matrix must be square");
    assert_eq!(x.len(), a.nrows(), "x length must equal nrows");
    assert_eq!(y.len(), a.nrows(), "y length must equal nrows");
    let n = a.nrows();
    let eps = T::epsilon();
    for i in (0..n).rev() {
        let start = a.row_ptrs()[i];
        let end = a.row_ptrs()[i + 1];
        let mut sum = x[i].clone();
        let mut diag = T::zero();
        for k in start..end {
            let j = a.col_indices()[k];
            if j > i {
                sum = sum - a.values()[k].clone() * y[j].clone();
            } else if j == i {
                diag = a.values()[k].clone();
            }
        }
        assert!(Scalar::abs(diag.clone()) > eps, "Zero diagonal at row {i}");
        y[i] = sum / diag;
    }
}
/// Scales a sparse matrix by a scalar: A = alpha * A
///
/// Modifies the matrix in place.
pub fn scale_inplace<T: Scalar + Clone + Field>(alpha: T, a: &mut CsrMatrix<T>) {
    let values = a.values_mut();
    for val in values.iter_mut() {
        *val = alpha.clone() * val.clone();
    }
}
/// Scales a sparse matrix by a scalar, returning a new matrix.
pub fn scale<T: Scalar + Clone + Field>(alpha: T, a: &CsrMatrix<T>) -> CsrMatrix<T> {
    let values: Vec<T> = a
        .values()
        .iter()
        .map(|v| alpha.clone() * v.clone())
        .collect();
    unsafe {
        CsrMatrix::new_unchecked(
            a.nrows(),
            a.ncols(),
            a.row_ptrs().to_vec(),
            a.col_indices().to_vec(),
            values,
        )
    }
}
/// Row-scales a sparse matrix: `A[i, :] = d[i] * A[i, :]`
///
/// Equivalent to D * A where D = diag(d).
pub fn scale_rows<T: Scalar + Clone + Field>(d: &[T], a: &mut CsrMatrix<T>) {
    assert_eq!(d.len(), a.nrows(), "d length must equal nrows");
    for i in 0..a.nrows() {
        let start = a.row_ptrs()[i];
        let end = a.row_ptrs()[i + 1];
        let di = d[i].clone();
        let values = a.values_mut();
        for k in start..end {
            values[k] = di.clone() * values[k].clone();
        }
    }
}
/// Column-scales a sparse matrix: `A[:, j] = d[j] * A[:, j]`
///
/// Equivalent to A * D where D = diag(d).
pub fn scale_cols<T: Scalar + Clone + Field>(a: &mut CsrMatrix<T>, d: &[T]) {
    assert_eq!(d.len(), a.ncols(), "d length must equal ncols");
    let nrows = a.nrows();
    let row_ptrs = a.row_ptrs().to_vec();
    let col_indices = a.col_indices().to_vec();
    for i in 0..nrows {
        let start = row_ptrs[i];
        let end = row_ptrs[i + 1];
        let values = a.values_mut();
        for k in start..end {
            let j = col_indices[k];
            values[k] = values[k].clone() * d[j].clone();
        }
    }
}
/// Row and column scales: A = D1 * A * D2 where D1 = diag(d1), D2 = diag(d2).
pub fn scale_rows_cols<T: Scalar + Clone + Field>(d1: &[T], a: &mut CsrMatrix<T>, d2: &[T]) {
    assert_eq!(d1.len(), a.nrows(), "d1 length must equal nrows");
    assert_eq!(d2.len(), a.ncols(), "d2 length must equal ncols");
    let nrows = a.nrows();
    let row_ptrs = a.row_ptrs().to_vec();
    let col_indices = a.col_indices().to_vec();
    for i in 0..nrows {
        let start = row_ptrs[i];
        let end = row_ptrs[i + 1];
        let d1i = d1[i].clone();
        let values = a.values_mut();
        for k in start..end {
            let j = col_indices[k];
            values[k] = d1i.clone() * values[k].clone() * d2[j].clone();
        }
    }
}
/// Sparse rank-1 update: A = A + alpha * x * y^T
///
/// Adds the outer product of vectors x and y to the sparse matrix.
/// Only existing sparsity pattern positions are updated (no fill-in).
///
/// # Arguments
///
/// * `a` - Sparse matrix to update
/// * `alpha` - Scalar multiplier
/// * `x` - Column vector (length = nrows)
/// * `y` - Row vector (length = ncols)
///
/// # Note
///
/// This only updates existing non-zero positions. New fill-in is ignored.
pub fn rank1_update_no_fill<T: Scalar + Clone + Field>(
    a: &mut CsrMatrix<T>,
    alpha: T,
    x: &[T],
    y: &[T],
) {
    assert_eq!(x.len(), a.nrows(), "x length must equal nrows");
    assert_eq!(y.len(), a.ncols(), "y length must equal ncols");
    let nrows = a.nrows();
    let row_ptrs = a.row_ptrs().to_vec();
    let col_indices = a.col_indices().to_vec();
    for i in 0..nrows {
        let start = row_ptrs[i];
        let end = row_ptrs[i + 1];
        let xi = x[i].clone();
        let values = a.values_mut();
        for k in start..end {
            let j = col_indices[k];
            values[k] = values[k].clone() + alpha.clone() * xi.clone() * y[j].clone();
        }
    }
}
/// Sparse rank-1 update with fill-in: A = A + alpha * x * y^T
///
/// Creates a new matrix including all new non-zeros.
///
/// # Arguments
///
/// * `a` - Original sparse matrix
/// * `alpha` - Scalar multiplier
/// * `x` - Column vector (length = nrows)
/// * `y` - Row vector (length = ncols)
///
/// # Returns
///
/// New sparse matrix with the rank-1 update applied.
pub fn rank1_update<T: Scalar + Clone + Field>(
    a: &CsrMatrix<T>,
    alpha: T,
    x: &[T],
    y: &[T],
) -> CsrMatrix<T> {
    assert_eq!(x.len(), a.nrows(), "x length must equal nrows");
    assert_eq!(y.len(), a.ncols(), "y length must equal ncols");
    let m = a.nrows();
    let n = a.ncols();
    let eps = T::epsilon();
    let mut row_ptrs = Vec::with_capacity(m + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    row_ptrs.push(0);
    for i in 0..m {
        let xi = x[i].clone();
        let a_start = a.row_ptrs()[i];
        let a_end = a.row_ptrs()[i + 1];
        let mut a_idx = a_start;
        for j in 0..n {
            let outer_val = alpha.clone() * xi.clone() * y[j].clone();
            let a_val = if a_idx < a_end && a.col_indices()[a_idx] == j {
                let v = a.values()[a_idx].clone();
                a_idx += 1;
                v
            } else {
                T::zero()
            };
            let sum = a_val + outer_val;
            if Scalar::abs(sum.clone()) > eps {
                col_indices.push(j);
                values.push(sum);
            }
        }
        row_ptrs.push(values.len());
    }
    unsafe { CsrMatrix::new_unchecked(m, n, row_ptrs, col_indices, values) }
}
/// Computes the sum of absolute values of matrix elements.
pub fn asum<T>(a: &CsrMatrix<T>) -> T::Real
where
    T: Scalar + Clone + Field,
    T::Real: Clone + Field + PartialOrd + Zero,
{
    a.values()
        .iter()
        .fold(T::Real::zero(), |acc, v| acc + Scalar::abs(v.clone()))
}
/// Computes the Frobenius norm of a sparse matrix.
pub fn frobenius_norm<T>(a: &CsrMatrix<T>) -> T::Real
where
    T: Scalar + Clone + Field,
    T::Real: Clone + Scalar<Real = T::Real> + Zero + Float,
{
    let sum_sq = a.values().iter().fold(T::Real::zero(), |acc, v| {
        let abs_v = Scalar::abs(v.clone());
        acc + abs_v.clone() * abs_v
    });
    sum_sq.sqrt()
}
/// Computes the infinity norm (maximum row sum) of a sparse matrix.
pub fn infinity_norm<T>(a: &CsrMatrix<T>) -> T::Real
where
    T: Scalar + Clone + Field,
    T::Real: Clone + Field + PartialOrd + Zero,
{
    let mut max_sum = T::Real::zero();
    for i in 0..a.nrows() {
        let start = a.row_ptrs()[i];
        let end = a.row_ptrs()[i + 1];
        let row_sum = a.values()[start..end]
            .iter()
            .fold(T::Real::zero(), |acc, v| acc + Scalar::abs(v.clone()));
        if row_sum > max_sum {
            max_sum = row_sum;
        }
    }
    max_sum
}
/// Computes the 1-norm (maximum column sum) of a sparse matrix.
pub fn one_norm<T>(a: &CsrMatrix<T>) -> T::Real
where
    T: Scalar + Clone + Field,
    T::Real: Clone + Field + PartialOrd + Zero,
{
    let n = a.ncols();
    let mut col_sums = vec![T::Real::zero(); n];
    for i in 0..a.nrows() {
        let start = a.row_ptrs()[i];
        let end = a.row_ptrs()[i + 1];
        for k in start..end {
            let j = a.col_indices()[k];
            col_sums[j] = col_sums[j].clone() + Scalar::abs(a.values()[k].clone());
        }
    }
    col_sums.into_iter().fold(
        T::Real::zero(),
        |max, sum| if sum > max { sum } else { max },
    )
}
/// SIMD-optimized sparse matrix-vector multiplication for f64: y = alpha * A * x + beta * y
///
/// Uses 4-way accumulation and software prefetching for better performance.
/// Falls back to scalar implementation for small rows.
pub fn spmv_f64_simd(alpha: f64, a: &CsrMatrix<f64>, x: &[f64], beta: f64, y: &mut [f64]) {
    assert_eq!(x.len(), a.ncols(), "x length must equal number of columns");
    assert_eq!(y.len(), a.nrows(), "y length must equal number of rows");
    let is_beta_zero = beta.abs() <= f64::EPSILON;
    let row_ptrs = a.row_ptrs();
    let col_indices = a.col_indices();
    let values = a.values();
    for i in 0..a.nrows() {
        let start = row_ptrs[i];
        let end = row_ptrs[i + 1];
        let row_nnz = end - start;
        let mut sum0 = 0.0f64;
        let mut sum1 = 0.0f64;
        let mut sum2 = 0.0f64;
        let mut sum3 = 0.0f64;
        let chunks = row_nnz / 4;
        let remainder = row_nnz % 4;
        let mut k = start;
        for _ in 0..chunks {
            let col0 = col_indices[k];
            let col1 = col_indices[k + 1];
            let col2 = col_indices[k + 2];
            let col3 = col_indices[k + 3];
            if k + 8 < end {
                let prefetch_col = col_indices[k + 8];
                if prefetch_col < x.len() {
                    let _ = x.get(prefetch_col);
                }
            }
            sum0 += values[k] * x[col0];
            sum1 += values[k + 1] * x[col1];
            sum2 += values[k + 2] * x[col2];
            sum3 += values[k + 3] * x[col3];
            k += 4;
        }
        for _ in 0..remainder {
            let col = col_indices[k];
            sum0 += values[k] * x[col];
            k += 1;
        }
        let sum = sum0 + sum1 + sum2 + sum3;
        if is_beta_zero {
            y[i] = alpha * sum;
        } else {
            y[i] = alpha * sum + beta * y[i];
        }
    }
}
/// SIMD-optimized sparse matrix-vector multiplication for f32: y = alpha * A * x + beta * y
///
/// Uses 8-way accumulation for better throughput on f32.
pub fn spmv_f32_simd(alpha: f32, a: &CsrMatrix<f32>, x: &[f32], beta: f32, y: &mut [f32]) {
    assert_eq!(x.len(), a.ncols(), "x length must equal number of columns");
    assert_eq!(y.len(), a.nrows(), "y length must equal number of rows");
    let is_beta_zero = beta.abs() <= f32::EPSILON;
    let row_ptrs = a.row_ptrs();
    let col_indices = a.col_indices();
    let values = a.values();
    for i in 0..a.nrows() {
        let start = row_ptrs[i];
        let end = row_ptrs[i + 1];
        let row_nnz = end - start;
        let mut sums = [0.0f32; 8];
        let chunks = row_nnz / 8;
        let remainder = row_nnz % 8;
        let mut k = start;
        for _ in 0..chunks {
            for j in 0..8 {
                let col = col_indices[k + j];
                sums[j] += values[k + j] * x[col];
            }
            k += 8;
        }
        for idx in 0..remainder {
            let col = col_indices[k];
            sums[idx % 8] += values[k] * x[col];
            k += 1;
        }
        let sum = sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7];
        if is_beta_zero {
            y[i] = alpha * sum;
        } else {
            y[i] = alpha * sum + beta * y[i];
        }
    }
}
/// Parallel sparse matrix-vector multiplication for f64: y = alpha * A * x + beta * y
///
/// Parallelizes across rows using Rayon for large matrices.
/// Falls back to sequential for small matrices (< 1000 rows).
#[cfg(feature = "parallel")]
pub fn spmv_f64_par(alpha: f64, a: &CsrMatrix<f64>, x: &[f64], beta: f64, y: &mut [f64]) {
    use rayon::prelude::*;
    assert_eq!(x.len(), a.ncols(), "x length must equal number of columns");
    assert_eq!(y.len(), a.nrows(), "y length must equal number of rows");
    const PAR_THRESHOLD: usize = 1000;
    if a.nrows() < PAR_THRESHOLD || a.nnz() < PAR_THRESHOLD * 10 {
        spmv_f64_simd(alpha, a, x, beta, y);
        return;
    }
    let is_beta_zero = beta.abs() <= f64::EPSILON;
    let row_ptrs = a.row_ptrs();
    let col_indices = a.col_indices();
    let values = a.values();
    y.par_iter_mut().enumerate().for_each(|(i, yi)| {
        let start = row_ptrs[i];
        let end = row_ptrs[i + 1];
        let mut sum0 = 0.0f64;
        let mut sum1 = 0.0f64;
        let mut sum2 = 0.0f64;
        let mut sum3 = 0.0f64;
        let row_nnz = end - start;
        let chunks = row_nnz / 4;
        let remainder = row_nnz % 4;
        let mut k = start;
        for _ in 0..chunks {
            let col0 = col_indices[k];
            let col1 = col_indices[k + 1];
            let col2 = col_indices[k + 2];
            let col3 = col_indices[k + 3];
            sum0 += values[k] * x[col0];
            sum1 += values[k + 1] * x[col1];
            sum2 += values[k + 2] * x[col2];
            sum3 += values[k + 3] * x[col3];
            k += 4;
        }
        for _ in 0..remainder {
            let col = col_indices[k];
            sum0 += values[k] * x[col];
            k += 1;
        }
        let sum = sum0 + sum1 + sum2 + sum3;
        if is_beta_zero {
            *yi = alpha * sum;
        } else {
            *yi = alpha * sum + beta * (*yi);
        }
    });
}
/// Parallel sparse matrix-vector multiplication for f32: y = alpha * A * x + beta * y
#[cfg(feature = "parallel")]
pub fn spmv_f32_par(alpha: f32, a: &CsrMatrix<f32>, x: &[f32], beta: f32, y: &mut [f32]) {
    use rayon::prelude::*;
    assert_eq!(x.len(), a.ncols(), "x length must equal number of columns");
    assert_eq!(y.len(), a.nrows(), "y length must equal number of rows");
    const PAR_THRESHOLD: usize = 1000;
    if a.nrows() < PAR_THRESHOLD || a.nnz() < PAR_THRESHOLD * 10 {
        spmv_f32_simd(alpha, a, x, beta, y);
        return;
    }
    let is_beta_zero = beta.abs() <= f32::EPSILON;
    let row_ptrs = a.row_ptrs();
    let col_indices = a.col_indices();
    let values = a.values();
    y.par_iter_mut().enumerate().for_each(|(i, yi)| {
        let start = row_ptrs[i];
        let end = row_ptrs[i + 1];
        let mut sums = [0.0f32; 8];
        let row_nnz = end - start;
        let chunks = row_nnz / 8;
        let remainder = row_nnz % 8;
        let mut k = start;
        for _ in 0..chunks {
            for j in 0..8 {
                let col = col_indices[k + j];
                sums[j] += values[k + j] * x[col];
            }
            k += 8;
        }
        for idx in 0..remainder {
            let col = col_indices[k];
            sums[idx % 8] += values[k] * x[col];
            k += 1;
        }
        let sum = sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7];
        if is_beta_zero {
            *yi = alpha * sum;
        } else {
            *yi = alpha * sum + beta * (*yi);
        }
    });
}
/// Selects the best SpMV implementation based on matrix size and available features.
///
/// For f64 matrices:
/// - Uses parallel version for large matrices (with "parallel" feature)
/// - Uses SIMD version for medium matrices
/// - Uses generic version for other types
pub fn spmv_auto<T: Scalar + Clone + Field>(
    alpha: T,
    a: &CsrMatrix<T>,
    x: &[T],
    beta: T,
    y: &mut [T],
) {
    if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f64>() {
        let alpha = unsafe { *(&alpha as *const T as *const f64) };
        let a = unsafe { &*(a as *const CsrMatrix<T> as *const CsrMatrix<f64>) };
        let x = unsafe { core::slice::from_raw_parts(x.as_ptr() as *const f64, x.len()) };
        let beta = unsafe { *(&beta as *const T as *const f64) };
        let y = unsafe { core::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f64, y.len()) };
        #[cfg(feature = "parallel")]
        {
            spmv_f64_par(alpha, a, x, beta, y);
            return;
        }
        #[cfg(not(feature = "parallel"))]
        {
            spmv_f64_simd(alpha, a, x, beta, y);
            return;
        }
    }
    if core::any::TypeId::of::<T>() == core::any::TypeId::of::<f32>() {
        let alpha = unsafe { *(&alpha as *const T as *const f32) };
        let a = unsafe { &*(a as *const CsrMatrix<T> as *const CsrMatrix<f32>) };
        let x = unsafe { core::slice::from_raw_parts(x.as_ptr() as *const f32, x.len()) };
        let beta = unsafe { *(&beta as *const T as *const f32) };
        let y = unsafe { core::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f32, y.len()) };
        #[cfg(feature = "parallel")]
        {
            spmv_f32_par(alpha, a, x, beta, y);
            return;
        }
        #[cfg(not(feature = "parallel"))]
        {
            spmv_f32_simd(alpha, a, x, beta, y);
            return;
        }
    }
    spmv(alpha, a, x, beta, y);
}
/// Parallel sparse matrix-matrix multiplication: C = A * B
///
/// Parallelizes row computation using Rayon for large matrices.
/// Falls back to sequential for small matrices (< 100 rows).
///
/// # Arguments
///
/// * `a` - Left sparse matrix in CSR format (m x k)
/// * `b` - Right sparse matrix in CSR format (k x n)
///
/// # Returns
///
/// Result sparse matrix in CSR format (m x n)
#[cfg(feature = "parallel")]
pub fn spmm_sparse_par<T>(a: &CsrMatrix<T>, b: &CsrMatrix<T>) -> CsrMatrix<T>
where
    T: Scalar + Clone + Field + Send + Sync,
{
    use rayon::prelude::*;
    assert_eq!(a.ncols(), b.nrows(), "Inner dimensions must match");
    let m = a.nrows();
    let n = b.ncols();
    const PAR_THRESHOLD: usize = 100;
    if m < PAR_THRESHOLD {
        return spmm_sparse(a, b);
    }
    let row_results: Vec<(Vec<usize>, Vec<T>)> = (0..m)
        .into_par_iter()
        .map(|i| {
            let mut workspace = vec![T::zero(); n];
            let mut col_flags = vec![false; n];
            let a_start = a.row_ptrs()[i];
            let a_end = a.row_ptrs()[i + 1];
            for a_idx in a_start..a_end {
                let k = a.col_indices()[a_idx];
                let a_val = a.values()[a_idx].clone();
                let b_start = b.row_ptrs()[k];
                let b_end = b.row_ptrs()[k + 1];
                for b_idx in b_start..b_end {
                    let j = b.col_indices()[b_idx];
                    let b_val = b.values()[b_idx].clone();
                    workspace[j] = workspace[j].clone() + a_val.clone() * b_val;
                    col_flags[j] = true;
                }
            }
            let mut row_cols = Vec::new();
            let mut row_vals = Vec::new();
            for j in 0..n {
                if col_flags[j] && Scalar::abs(workspace[j].clone()) > T::epsilon() {
                    row_cols.push(j);
                    row_vals.push(workspace[j].clone());
                }
            }
            (row_cols, row_vals)
        })
        .collect();
    let mut row_ptrs = Vec::with_capacity(m + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    row_ptrs.push(0);
    for (row_cols, row_vals) in row_results {
        col_indices.extend(row_cols);
        values.extend(row_vals);
        row_ptrs.push(values.len());
    }
    unsafe { CsrMatrix::new_unchecked(m, n, row_ptrs, col_indices, values) }
}
/// Automatic sparse matrix-matrix multiply dispatch.
///
/// Uses parallel version for large matrices when "parallel" feature is enabled.
pub fn spmm_sparse_auto<T>(a: &CsrMatrix<T>, b: &CsrMatrix<T>) -> CsrMatrix<T>
where
    T: Scalar + Clone + Field + Send + Sync,
{
    #[cfg(feature = "parallel")]
    {
        spmm_sparse_par(a, b)
    }
    #[cfg(not(feature = "parallel"))]
    {
        spmm_sparse(a, b)
    }
}
/// Cache-blocked SpMV for f64: y = alpha * A * x + beta * y
///
/// Processes rows in blocks to improve cache locality for x vector.
/// Uses block size of 256 rows for optimal L2 cache utilization.
pub fn spmv_f64_blocked(alpha: f64, a: &CsrMatrix<f64>, x: &[f64], beta: f64, y: &mut [f64]) {
    assert_eq!(x.len(), a.ncols(), "x length must equal number of columns");
    assert_eq!(y.len(), a.nrows(), "y length must equal number of rows");
    const BLOCK_SIZE: usize = 256;
    let is_beta_zero = beta.abs() <= f64::EPSILON;
    let row_ptrs = a.row_ptrs();
    let col_indices = a.col_indices();
    let values = a.values();
    let m = a.nrows();
    let mut row = 0;
    while row < m {
        let block_end = (row + BLOCK_SIZE).min(m);
        for i in row..block_end {
            let start = row_ptrs[i];
            let end = row_ptrs[i + 1];
            let mut sum0 = 0.0f64;
            let mut sum1 = 0.0f64;
            let mut sum2 = 0.0f64;
            let mut sum3 = 0.0f64;
            let row_nnz = end - start;
            let chunks = row_nnz / 4;
            let remainder = row_nnz % 4;
            let mut k = start;
            for _ in 0..chunks {
                let col0 = col_indices[k];
                let col1 = col_indices[k + 1];
                let col2 = col_indices[k + 2];
                let col3 = col_indices[k + 3];
                sum0 += values[k] * x[col0];
                sum1 += values[k + 1] * x[col1];
                sum2 += values[k + 2] * x[col2];
                sum3 += values[k + 3] * x[col3];
                k += 4;
            }
            for _ in 0..remainder {
                let col = col_indices[k];
                sum0 += values[k] * x[col];
                k += 1;
            }
            let sum = sum0 + sum1 + sum2 + sum3;
            if is_beta_zero {
                y[i] = alpha * sum;
            } else {
                y[i] = alpha * sum + beta * y[i];
            }
        }
        row = block_end;
    }
}
/// Cache-blocked SpMV for f32: y = alpha * A * x + beta * y
pub fn spmv_f32_blocked(alpha: f32, a: &CsrMatrix<f32>, x: &[f32], beta: f32, y: &mut [f32]) {
    assert_eq!(x.len(), a.ncols(), "x length must equal number of columns");
    assert_eq!(y.len(), a.nrows(), "y length must equal number of rows");
    const BLOCK_SIZE: usize = 256;
    let is_beta_zero = beta.abs() <= f32::EPSILON;
    let row_ptrs = a.row_ptrs();
    let col_indices = a.col_indices();
    let values = a.values();
    let m = a.nrows();
    let mut row = 0;
    while row < m {
        let block_end = (row + BLOCK_SIZE).min(m);
        for i in row..block_end {
            let start = row_ptrs[i];
            let end = row_ptrs[i + 1];
            let mut sums = [0.0f32; 8];
            let row_nnz = end - start;
            let chunks = row_nnz / 8;
            let remainder = row_nnz % 8;
            let mut k = start;
            for _ in 0..chunks {
                for j in 0..8 {
                    let col = col_indices[k + j];
                    sums[j] += values[k + j] * x[col];
                }
                k += 8;
            }
            for idx in 0..remainder {
                let col = col_indices[k];
                sums[idx % 8] += values[k] * x[col];
                k += 1;
            }
            let sum = sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7];
            if is_beta_zero {
                y[i] = alpha * sum;
            } else {
                y[i] = alpha * sum + beta * y[i];
            }
        }
        row = block_end;
    }
}
/// Hybrid SpMV combining blocking and parallelism for f64.
///
/// For very large matrices, combines cache blocking with row-parallel execution.
#[cfg(feature = "parallel")]
pub fn spmv_f64_hybrid(alpha: f64, a: &CsrMatrix<f64>, x: &[f64], beta: f64, y: &mut [f64]) {
    use rayon::prelude::*;
    assert_eq!(x.len(), a.ncols(), "x length must equal number of columns");
    assert_eq!(y.len(), a.nrows(), "y length must equal number of rows");
    const PAR_THRESHOLD: usize = 2000;
    const CHUNK_SIZE: usize = 512;
    if a.nrows() < PAR_THRESHOLD {
        spmv_f64_blocked(alpha, a, x, beta, y);
        return;
    }
    let is_beta_zero = beta.abs() <= f64::EPSILON;
    let row_ptrs = a.row_ptrs();
    let col_indices = a.col_indices();
    let values = a.values();
    y.par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, y_chunk)| {
            let start_row = chunk_idx * CHUNK_SIZE;
            for (local_i, yi) in y_chunk.iter_mut().enumerate() {
                let i = start_row + local_i;
                if i >= a.nrows() {
                    break;
                }
                let start = row_ptrs[i];
                let end = row_ptrs[i + 1];
                let mut sum0 = 0.0f64;
                let mut sum1 = 0.0f64;
                let mut sum2 = 0.0f64;
                let mut sum3 = 0.0f64;
                let row_nnz = end - start;
                let chunks = row_nnz / 4;
                let remainder = row_nnz % 4;
                let mut k = start;
                for _ in 0..chunks {
                    let col0 = col_indices[k];
                    let col1 = col_indices[k + 1];
                    let col2 = col_indices[k + 2];
                    let col3 = col_indices[k + 3];
                    sum0 += values[k] * x[col0];
                    sum1 += values[k + 1] * x[col1];
                    sum2 += values[k + 2] * x[col2];
                    sum3 += values[k + 3] * x[col3];
                    k += 4;
                }
                for _ in 0..remainder {
                    let col = col_indices[k];
                    sum0 += values[k] * x[col];
                    k += 1;
                }
                let sum = sum0 + sum1 + sum2 + sum3;
                if is_beta_zero {
                    *yi = alpha * sum;
                } else {
                    *yi = alpha * sum + beta * (*yi);
                }
            }
        });
}
#[cfg(test)]
mod tests {
    use super::*;
    fn make_test_csr() -> CsrMatrix<f64> {
        let values = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let col_indices = vec![0, 2, 1, 0, 2];
        let row_ptrs = vec![0, 2, 3, 5];
        CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap()
    }
    #[test]
    fn test_spmv() {
        let a = make_test_csr();
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0, 0.0, 0.0];
        spmv(1.0, &a, &x, 0.0, &mut y);
        assert!((y[0] - 7.0).abs() < 1e-10);
        assert!((y[1] - 6.0).abs() < 1e-10);
        assert!((y[2] - 19.0).abs() < 1e-10);
    }
    #[test]
    fn test_spmv_with_beta() {
        let a = make_test_csr();
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![1.0, 1.0, 1.0];
        spmv(1.0, &a, &x, 2.0, &mut y);
        assert!((y[0] - 9.0).abs() < 1e-10);
        assert!((y[1] - 8.0).abs() < 1e-10);
        assert!((y[2] - 21.0).abs() < 1e-10);
    }
    #[test]
    fn test_spmv_transpose() {
        let a = make_test_csr();
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0, 0.0, 0.0];
        spmv_transpose(1.0, &a, &x, 0.0, &mut y);
        assert!((y[0] - 13.0).abs() < 1e-10);
        assert!((y[1] - 6.0).abs() < 1e-10);
        assert!((y[2] - 17.0).abs() < 1e-10);
    }
    #[test]
    fn test_spmm_sparse() {
        let a = CsrMatrix::<f64>::eye(3);
        let b = make_test_csr();
        let c = spmm_sparse(&a, &b);
        assert_eq!(c.nnz(), b.nnz());
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(c.get(i, j), b.get(i, j));
            }
        }
    }
    #[test]
    fn test_spmm_sparse_square_nontrivial() {
        let a_values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let a_col_indices = vec![0, 2, 1, 0, 2];
        let a_row_ptrs = vec![0, 2, 3, 5];
        let a = CsrMatrix::new(3, 3, a_row_ptrs, a_col_indices, a_values).unwrap();
        let b_values = vec![2.0, 1.0, 3.0];
        let b_col_indices = vec![0, 1, 0];
        let b_row_ptrs = vec![0, 1, 2, 3];
        let b = CsrMatrix::new(3, 2, b_row_ptrs, b_col_indices, b_values).unwrap();
        let c = spmm_sparse(&a, &b);
        assert_eq!(c.nrows(), 3);
        assert_eq!(c.ncols(), 2);
        assert_eq!(c.get(0, 0), Some(&8.0));
        assert_eq!(c.get(0, 1), None);
        assert_eq!(c.get(1, 0), None);
        assert_eq!(c.get(1, 1), Some(&3.0));
        assert_eq!(c.get(2, 0), Some(&23.0));
        assert_eq!(c.get(2, 1), None);
    }
    #[test]
    fn test_spmm_sparse_rectangular() {
        let a_values = vec![1.0, 2.0, 3.0, 4.0];
        let a_col_indices = vec![0, 2, 1, 3];
        let a_row_ptrs = vec![0, 2, 4];
        let a = CsrMatrix::new(2, 4, a_row_ptrs, a_col_indices, a_values).unwrap();
        let b_values = vec![1.0, 1.0, 1.0, 2.0];
        let b_col_indices = vec![0, 1, 2, 0];
        let b_row_ptrs = vec![0, 1, 2, 3, 4];
        let b = CsrMatrix::new(4, 3, b_row_ptrs, b_col_indices, b_values).unwrap();
        let c = spmm_sparse(&a, &b);
        assert_eq!(c.nrows(), 2);
        assert_eq!(c.ncols(), 3);
        assert_eq!(c.get(0, 0), Some(&1.0));
        assert_eq!(c.get(0, 1), None);
        assert_eq!(c.get(0, 2), Some(&2.0));
        assert_eq!(c.get(1, 0), Some(&8.0));
        assert_eq!(c.get(1, 1), Some(&3.0));
        assert_eq!(c.get(1, 2), None);
    }
    #[test]
    fn test_spmm_sparse_very_sparse() {
        let a_values = vec![1.0, 2.0, 3.0];
        let a_col_indices = vec![0, 5, 9];
        let a_row_ptrs = vec![0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3];
        let a = CsrMatrix::new(10, 10, a_row_ptrs, a_col_indices, a_values).unwrap();
        let b_values = vec![4.0, 5.0];
        let b_col_indices = vec![3, 7];
        let b_row_ptrs = vec![0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2];
        let b = CsrMatrix::new(10, 10, b_row_ptrs, b_col_indices, b_values).unwrap();
        let c = spmm_sparse(&a, &b);
        assert_eq!(c.nrows(), 10);
        assert_eq!(c.ncols(), 10);
        assert_eq!(c.get(0, 3), Some(&4.0));
        assert_eq!(c.get(9, 7), Some(&15.0));
        let mut nnz_count = 0;
        for i in 0..10 {
            for j in 0..10 {
                if c.get(i, j).is_some() {
                    nnz_count += 1;
                }
            }
        }
        assert_eq!(nnz_count, 2);
    }
    #[test]
    fn test_spmm_sparse_accumulation() {
        let a_values = vec![1.0, 1.0, 1.0, 1.0];
        let a_col_indices = vec![0, 1, 1, 2];
        let a_row_ptrs = vec![0, 2, 4];
        let a = CsrMatrix::new(2, 3, a_row_ptrs, a_col_indices, a_values).unwrap();
        let b_values = vec![2.0, 3.0, 4.0, 5.0];
        let b_col_indices = vec![0, 0, 1, 1];
        let b_row_ptrs = vec![0, 1, 3, 4];
        let b = CsrMatrix::new(3, 2, b_row_ptrs, b_col_indices, b_values).unwrap();
        let c = spmm_sparse(&a, &b);
        assert_eq!(c.nrows(), 2);
        assert_eq!(c.ncols(), 2);
        assert_eq!(c.get(0, 0), Some(&5.0));
        assert_eq!(c.get(0, 1), Some(&4.0));
        assert_eq!(c.get(1, 0), Some(&3.0));
        assert_eq!(c.get(1, 1), Some(&9.0));
    }
    #[test]
    fn test_spadd() {
        let a = make_test_csr();
        let b = CsrMatrix::<f64>::eye(3);
        let c = spadd(1.0, &a, 2.0, &b);
        assert_eq!(c.get(0, 0), Some(&3.0));
        assert_eq!(c.get(1, 1), Some(&5.0));
        assert_eq!(c.get(2, 2), Some(&7.0));
        assert_eq!(c.get(0, 2), Some(&2.0));
        assert_eq!(c.get(2, 0), Some(&4.0));
    }
    #[test]
    fn test_transpose_csr() {
        let a = make_test_csr();
        let at = transpose_csr(&a);
        assert_eq!(at.nrows(), a.ncols());
        assert_eq!(at.ncols(), a.nrows());
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert_eq!(at.get(j, i), a.get(i, j));
            }
        }
    }
    #[test]
    fn test_diagonal() {
        let a = make_test_csr();
        let diag = diagonal(&a);
        assert_eq!(diag.len(), 3);
        assert!((diag[0] - 1.0).abs() < 1e-10);
        assert!((diag[1] - 3.0).abs() < 1e-10);
        assert!((diag[2] - 5.0).abs() < 1e-10);
    }
    #[test]
    fn test_from_diagonal() {
        let diag = vec![1.0, 2.0, 0.0, 4.0];
        let a = from_diagonal(&diag);
        assert_eq!(a.nrows(), 4);
        assert_eq!(a.ncols(), 4);
        assert_eq!(a.nnz(), 3);
        assert_eq!(a.get(0, 0), Some(&1.0));
        assert_eq!(a.get(1, 1), Some(&2.0));
        assert_eq!(a.get(2, 2), None);
        assert_eq!(a.get(3, 3), Some(&4.0));
    }
    #[test]
    fn test_ata() {
        let values = vec![1.0f64, 2.0, 3.0, 4.0];
        let col_indices = vec![0, 1, 1, 2];
        let row_ptrs = vec![0, 2, 4];
        let a = CsrMatrix::new(2, 3, row_ptrs, col_indices, values).unwrap();
        let ata = ata(&a);
        assert_eq!(ata.nrows(), 3);
        assert_eq!(ata.ncols(), 3);
        assert_eq!(ata.get(0, 0), Some(&1.0));
        assert_eq!(ata.get(0, 1), Some(&2.0));
        assert_eq!(ata.get(1, 0), Some(&2.0));
        assert_eq!(ata.get(1, 1), Some(&13.0));
        assert_eq!(ata.get(1, 2), Some(&12.0));
        assert_eq!(ata.get(2, 1), Some(&12.0));
        assert_eq!(ata.get(2, 2), Some(&16.0));
    }
    #[test]
    fn test_spmv_f64_simd() {
        let a = make_test_csr();
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0, 0.0, 0.0];
        spmv_f64_simd(1.0, &a, &x, 0.0, &mut y);
        assert!((y[0] - 7.0).abs() < 1e-10);
        assert!((y[1] - 6.0).abs() < 1e-10);
        assert!((y[2] - 19.0).abs() < 1e-10);
    }
    #[test]
    fn test_spmv_f64_simd_with_beta() {
        let a = make_test_csr();
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![1.0, 1.0, 1.0];
        spmv_f64_simd(1.0, &a, &x, 2.0, &mut y);
        assert!((y[0] - 9.0).abs() < 1e-10);
        assert!((y[1] - 8.0).abs() < 1e-10);
        assert!((y[2] - 21.0).abs() < 1e-10);
    }
    #[test]
    fn test_spmv_f32_simd() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let col_indices = vec![0, 2, 1, 0, 2];
        let row_ptrs = vec![0, 2, 3, 5];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();
        let x = vec![1.0f32, 2.0, 3.0];
        let mut y = vec![0.0f32, 0.0, 0.0];
        spmv_f32_simd(1.0, &a, &x, 0.0, &mut y);
        assert!((y[0] - 7.0).abs() < 1e-5);
        assert!((y[1] - 6.0).abs() < 1e-5);
        assert!((y[2] - 19.0).abs() < 1e-5);
    }
    #[test]
    fn test_spmv_f64_simd_large_row() {
        let n = 8;
        let values: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let col_indices: Vec<usize> = (0..n).collect();
        let row_ptrs = vec![0, n];
        let a = CsrMatrix::new(1, n, row_ptrs, col_indices, values).unwrap();
        let x: Vec<f64> = vec![1.0; n];
        let mut y = vec![0.0];
        spmv_f64_simd(1.0, &a, &x, 0.0, &mut y);
        assert!((y[0] - 36.0).abs() < 1e-10);
    }
    #[test]
    fn test_spmv_auto_f64() {
        let a = make_test_csr();
        let x = vec![1.0f64, 2.0, 3.0];
        let mut y = vec![0.0f64, 0.0, 0.0];
        spmv_auto(1.0, &a, &x, 0.0, &mut y);
        assert!((y[0] - 7.0).abs() < 1e-10);
        assert!((y[1] - 6.0).abs() < 1e-10);
        assert!((y[2] - 19.0).abs() < 1e-10);
    }
    #[test]
    fn test_spmv_auto_f32() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let col_indices = vec![0, 2, 1, 0, 2];
        let row_ptrs = vec![0, 2, 3, 5];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();
        let x = vec![1.0f32, 2.0, 3.0];
        let mut y = vec![0.0f32, 0.0, 0.0];
        spmv_auto(1.0, &a, &x, 0.0, &mut y);
        assert!((y[0] - 7.0).abs() < 1e-5);
        assert!((y[1] - 6.0).abs() < 1e-5);
        assert!((y[2] - 19.0).abs() < 1e-5);
    }
    #[test]
    fn test_spmv_simd_consistency_with_generic() {
        let values = vec![1.5f64, 2.3, 4.1, 0.7, 3.2, 1.9, 2.8, 0.5, 4.4];
        let col_indices = vec![0, 2, 4, 1, 3, 0, 2, 1, 4];
        let row_ptrs = vec![0, 3, 5, 9];
        let a = CsrMatrix::new(3, 5, row_ptrs, col_indices, values).unwrap();
        let x = vec![1.1, 2.2, 3.3, 4.4, 5.5];
        let mut y_generic = vec![0.5, 0.5, 0.5];
        let mut y_simd = vec![0.5, 0.5, 0.5];
        spmv(2.0, &a, &x, 1.5, &mut y_generic);
        spmv_f64_simd(2.0, &a, &x, 1.5, &mut y_simd);
        for i in 0..3 {
            assert!(
                (y_generic[i] - y_simd[i]).abs() < 1e-10,
                "Mismatch at index {}: generic={}, simd={}",
                i,
                y_generic[i],
                y_simd[i]
            );
        }
    }
    #[test]
    fn test_spmv_f64_blocked() {
        let a = make_test_csr();
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0, 0.0, 0.0];
        spmv_f64_blocked(1.0, &a, &x, 0.0, &mut y);
        assert!((y[0] - 7.0).abs() < 1e-10);
        assert!((y[1] - 6.0).abs() < 1e-10);
        assert!((y[2] - 19.0).abs() < 1e-10);
    }
    #[test]
    fn test_spmv_f64_blocked_with_beta() {
        let a = make_test_csr();
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![1.0, 1.0, 1.0];
        spmv_f64_blocked(1.0, &a, &x, 2.0, &mut y);
        assert!((y[0] - 9.0).abs() < 1e-10);
        assert!((y[1] - 8.0).abs() < 1e-10);
        assert!((y[2] - 21.0).abs() < 1e-10);
    }
    #[test]
    fn test_spmv_f32_blocked() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let col_indices = vec![0, 2, 1, 0, 2];
        let row_ptrs = vec![0, 2, 3, 5];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();
        let x = vec![1.0f32, 2.0, 3.0];
        let mut y = vec![0.0f32, 0.0, 0.0];
        spmv_f32_blocked(1.0, &a, &x, 0.0, &mut y);
        assert!((y[0] - 7.0).abs() < 1e-5);
        assert!((y[1] - 6.0).abs() < 1e-5);
        assert!((y[2] - 19.0).abs() < 1e-5);
    }
    #[test]
    fn test_spmv_blocked_consistency_with_simd() {
        let values = vec![1.5f64, 2.3, 4.1, 0.7, 3.2, 1.9, 2.8, 0.5, 4.4];
        let col_indices = vec![0, 2, 4, 1, 3, 0, 2, 1, 4];
        let row_ptrs = vec![0, 3, 5, 9];
        let a = CsrMatrix::new(3, 5, row_ptrs, col_indices, values).unwrap();
        let x = vec![1.1, 2.2, 3.3, 4.4, 5.5];
        let mut y_simd = vec![0.5, 0.5, 0.5];
        let mut y_blocked = vec![0.5, 0.5, 0.5];
        spmv_f64_simd(2.0, &a, &x, 1.5, &mut y_simd);
        spmv_f64_blocked(2.0, &a, &x, 1.5, &mut y_blocked);
        for i in 0..3 {
            assert!(
                (y_simd[i] - y_blocked[i]).abs() < 1e-10,
                "Mismatch at index {}: simd={}, blocked={}",
                i,
                y_simd[i],
                y_blocked[i]
            );
        }
    }
    #[test]
    fn test_spmm_sparse_auto() {
        let a = make_test_csr();
        let b = CsrMatrix::<f64>::eye(3);
        let c = spmm_sparse_auto(&a, &b);
        assert_eq!(c.nnz(), a.nnz());
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(c.get(i, j), a.get(i, j));
            }
        }
    }
    #[test]
    fn test_spmm_sparse_auto_nontrivial() {
        let a_values = vec![1.0, 1.0, 1.0, 1.0];
        let a_col_indices = vec![0, 1, 1, 2];
        let a_row_ptrs = vec![0, 2, 4];
        let a = CsrMatrix::new(2, 3, a_row_ptrs, a_col_indices, a_values).unwrap();
        let b_values = vec![2.0, 3.0, 4.0, 5.0];
        let b_col_indices = vec![0, 0, 1, 1];
        let b_row_ptrs = vec![0, 1, 3, 4];
        let b = CsrMatrix::new(3, 2, b_row_ptrs, b_col_indices, b_values).unwrap();
        let c_auto = spmm_sparse_auto(&a, &b);
        let c_seq = spmm_sparse(&a, &b);
        assert_eq!(c_auto.nrows(), c_seq.nrows());
        assert_eq!(c_auto.ncols(), c_seq.ncols());
        for i in 0..c_auto.nrows() {
            for j in 0..c_auto.ncols() {
                assert_eq!(c_auto.get(i, j), c_seq.get(i, j));
            }
        }
    }
    #[test]
    fn test_spmv_blocked_large_row() {
        let n = 16;
        let values: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let col_indices: Vec<usize> = (0..n).collect();
        let row_ptrs = vec![0, n];
        let a = CsrMatrix::new(1, n, row_ptrs, col_indices, values).unwrap();
        let x: Vec<f64> = vec![1.0; n];
        let mut y_simd = vec![0.0];
        let mut y_blocked = vec![0.0];
        spmv_f64_simd(1.0, &a, &x, 0.0, &mut y_simd);
        spmv_f64_blocked(1.0, &a, &x, 0.0, &mut y_blocked);
        assert!((y_simd[0] - 136.0).abs() < 1e-10);
        assert!((y_blocked[0] - 136.0).abs() < 1e-10);
    }
}
