//! Matrix operations.
//!
//! This module provides various matrix operations including:
//! - In-place transpose
//! - Diagonal extraction and setting
//! - Row/column permutation
//! - Block extraction
//! - Matrix copy utilities

use crate::{Mat, MatMut, MatRef};
use num_traits::Zero;
use oxiblas_core::scalar::Scalar;

/// In-place transpose for square matrices.
///
/// For non-square matrices, use `transpose()` which creates a new matrix.
///
/// # Example
///
/// ```
/// use oxiblas_matrix::{Mat, ops::transpose_inplace};
///
/// let mut m: Mat<f64> = Mat::from_rows(&[
///     &[1.0, 2.0, 3.0],
///     &[4.0, 5.0, 6.0],
///     &[7.0, 8.0, 9.0],
/// ]);
///
/// transpose_inplace(&mut m);
///
/// assert_eq!(m[(0, 1)], 4.0);  // Was at (1, 0)
/// assert_eq!(m[(1, 0)], 2.0);  // Was at (0, 1)
/// ```
pub fn transpose_inplace<T: Scalar>(mat: &mut Mat<T>) {
    assert!(
        mat.nrows() == mat.ncols(),
        "Matrix must be square for in-place transpose"
    );
    let n = mat.nrows();

    for i in 0..n {
        for j in (i + 1)..n {
            let temp = mat[(i, j)];
            mat[(i, j)] = mat[(j, i)];
            mat[(j, i)] = temp;
        }
    }
}

/// In-place transpose for a mutable matrix view.
pub fn transpose_inplace_mut<T: Scalar>(mat: &mut MatMut<'_, T>) {
    assert!(
        mat.is_square(),
        "Matrix must be square for in-place transpose"
    );
    let n = mat.nrows();

    for i in 0..n {
        for j in (i + 1)..n {
            let temp = mat[(i, j)];
            mat[(i, j)] = mat[(j, i)];
            mat[(j, i)] = temp;
        }
    }
}

/// Extracts the diagonal of a matrix into a vector.
///
/// For non-square matrices, returns the shorter diagonal.
pub fn extract_diagonal<T: Scalar>(mat: &MatRef<'_, T>) -> Vec<T> {
    let len = mat.nrows().min(mat.ncols());
    (0..len).map(|i| mat[(i, i)]).collect()
}

/// Sets the diagonal of a matrix from a slice.
///
/// # Panics
/// Panics if the slice is longer than the diagonal.
pub fn set_diagonal<T: Scalar>(mat: &mut MatMut<'_, T>, diag: &[T]) {
    let len = mat.nrows().min(mat.ncols());
    assert!(diag.len() <= len, "Diagonal slice too long");

    for (i, &val) in diag.iter().enumerate() {
        mat.set(i, i, val);
    }
}

/// Adds a scalar to the diagonal (A = A + alpha * I).
pub fn add_to_diagonal<T: Scalar>(mat: &mut MatMut<'_, T>, alpha: T) {
    let len = mat.nrows().min(mat.ncols());

    for i in 0..len {
        let val = mat[(i, i)];
        mat.set(i, i, val + alpha);
    }
}

/// Extracts a specific diagonal from a matrix.
///
/// `k = 0` is the main diagonal, `k > 0` are superdiagonals,
/// `k < 0` are subdiagonals.
pub fn extract_kth_diagonal<T: Scalar>(mat: &MatRef<'_, T>, k: isize) -> Vec<T> {
    let (nrows, ncols) = mat.shape();
    let (start_row, start_col) = if k >= 0 {
        (0, k as usize)
    } else {
        ((-k) as usize, 0)
    };

    if start_row >= nrows || start_col >= ncols {
        return Vec::new();
    }

    let len = (nrows - start_row).min(ncols - start_col);
    (0..len)
        .map(|i| mat[(start_row + i, start_col + i)])
        .collect()
}

/// Sets a specific diagonal of a matrix.
///
/// `k = 0` is the main diagonal, `k > 0` are superdiagonals,
/// `k < 0` are subdiagonals.
pub fn set_kth_diagonal<T: Scalar>(mat: &mut MatMut<'_, T>, k: isize, diag: &[T]) {
    let (nrows, ncols) = mat.shape();
    let (start_row, start_col) = if k >= 0 {
        (0, k as usize)
    } else {
        ((-k) as usize, 0)
    };

    if start_row >= nrows || start_col >= ncols {
        return;
    }

    let max_len = (nrows - start_row).min(ncols - start_col);
    let len = diag.len().min(max_len);

    for i in 0..len {
        mat.set(start_row + i, start_col + i, diag[i]);
    }
}

/// Permutes rows of a matrix according to a permutation vector.
///
/// After this operation, row `i` of the result is row `perm[i]` of the original.
///
/// # Panics
/// Panics if `perm` length doesn't match the number of rows, or if
/// `perm` contains invalid indices.
pub fn permute_rows<T: Scalar + bytemuck::Zeroable>(mat: &MatRef<'_, T>, perm: &[usize]) -> Mat<T> {
    let (nrows, ncols) = mat.shape();
    assert_eq!(perm.len(), nrows, "Permutation length must match row count");

    let mut result = Mat::zeros(nrows, ncols);

    for (new_i, &old_i) in perm.iter().enumerate() {
        assert!(old_i < nrows, "Invalid permutation index");
        for j in 0..ncols {
            result[(new_i, j)] = mat[(old_i, j)];
        }
    }

    result
}

/// Permutes columns of a matrix according to a permutation vector.
///
/// After this operation, column `j` of the result is column `perm[j]` of the original.
pub fn permute_cols<T: Scalar + bytemuck::Zeroable>(mat: &MatRef<'_, T>, perm: &[usize]) -> Mat<T> {
    let (nrows, ncols) = mat.shape();
    assert_eq!(
        perm.len(),
        ncols,
        "Permutation length must match column count"
    );

    let mut result = Mat::zeros(nrows, ncols);

    for (new_j, &old_j) in perm.iter().enumerate() {
        assert!(old_j < ncols, "Invalid permutation index");
        for i in 0..nrows {
            result[(i, new_j)] = mat[(i, old_j)];
        }
    }

    result
}

/// Applies row permutation in-place using swaps.
///
/// `perm` should be a valid permutation of 0..nrows.
pub fn permute_rows_inplace<T: Scalar>(mat: &mut MatMut<'_, T>, perm: &[usize]) {
    let nrows = mat.nrows();
    assert_eq!(perm.len(), nrows, "Permutation length must match row count");

    // Track which positions have been processed
    let mut processed = vec![false; nrows];

    for start in 0..nrows {
        if processed[start] {
            continue;
        }

        // Follow the cycle
        let mut current = start;
        while !processed[current] {
            processed[current] = true;
            let next = perm[current];

            if next != start && !processed[next] {
                mat.swap_rows(current, next);
            }
            current = next;
        }
    }
}

/// Applies column permutation in-place using swaps.
pub fn permute_cols_inplace<T: Scalar>(mat: &mut MatMut<'_, T>, perm: &[usize]) {
    let ncols = mat.ncols();
    assert_eq!(
        perm.len(),
        ncols,
        "Permutation length must match column count"
    );

    let mut processed = vec![false; ncols];

    for start in 0..ncols {
        if processed[start] {
            continue;
        }

        let mut current = start;
        while !processed[current] {
            processed[current] = true;
            let next = perm[current];

            if next != start && !processed[next] {
                mat.swap_cols(current, next);
            }
            current = next;
        }
    }
}

/// Extracts a block/submatrix as a new matrix.
pub fn extract_block<T: Scalar + bytemuck::Zeroable>(
    mat: &MatRef<'_, T>,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
) -> Mat<T> {
    assert!(
        row_start + nrows <= mat.nrows() && col_start + ncols <= mat.ncols(),
        "Block extends beyond matrix bounds"
    );

    let mut result = Mat::zeros(nrows, ncols);

    for j in 0..ncols {
        for i in 0..nrows {
            result[(i, j)] = mat[(row_start + i, col_start + j)];
        }
    }

    result
}

/// Sets a block/submatrix from another matrix.
pub fn set_block<T: Scalar>(
    mat: &mut MatMut<'_, T>,
    row_start: usize,
    col_start: usize,
    block: &MatRef<'_, T>,
) {
    let (block_rows, block_cols) = block.shape();

    assert!(
        row_start + block_rows <= mat.nrows() && col_start + block_cols <= mat.ncols(),
        "Block extends beyond matrix bounds"
    );

    for j in 0..block_cols {
        for i in 0..block_rows {
            mat.set(row_start + i, col_start + j, block[(i, j)]);
        }
    }
}

/// Copies a matrix to another.
///
/// Both matrices must have the same shape.
pub fn copy<T: Scalar>(dst: &mut MatMut<'_, T>, src: &MatRef<'_, T>) {
    assert_eq!(dst.shape(), src.shape(), "Matrix shapes must match");
    let (nrows, ncols) = dst.shape();

    for j in 0..ncols {
        for i in 0..nrows {
            dst.set(i, j, src[(i, j)]);
        }
    }
}

/// Computes dst = alpha * src + beta * dst (AXPY-like for matrices).
pub fn axpy<T: Scalar>(dst: &mut MatMut<'_, T>, alpha: T, src: &MatRef<'_, T>, beta: T) {
    assert_eq!(dst.shape(), src.shape(), "Matrix shapes must match");
    let (nrows, ncols) = dst.shape();

    for j in 0..ncols {
        for i in 0..nrows {
            let val = beta * dst[(i, j)] + alpha * src[(i, j)];
            dst.set(i, j, val);
        }
    }
}

/// Computes the trace (sum of diagonal elements).
pub fn trace<T: Scalar>(mat: &MatRef<'_, T>) -> T {
    let len = mat.nrows().min(mat.ncols());
    let mut sum = T::zero();

    for i in 0..len {
        sum += mat[(i, i)];
    }

    sum
}

/// Computes the Frobenius norm squared (sum of squared absolute values).
pub fn frobenius_norm_squared<T: Scalar>(mat: &MatRef<'_, T>) -> T::Real
where
    T::Real: Zero,
{
    let (nrows, ncols) = mat.shape();
    let mut sum = T::Real::zero();

    for j in 0..ncols {
        for i in 0..nrows {
            let abs_val = mat[(i, j)].abs();
            sum += abs_val * abs_val;
        }
    }

    sum
}

/// Computes the maximum absolute value in the matrix.
pub fn max_abs<T: Scalar>(mat: &MatRef<'_, T>) -> T::Real
where
    T::Real: Zero,
{
    let (nrows, ncols) = mat.shape();
    let mut max_val = T::Real::zero();

    for j in 0..ncols {
        for i in 0..nrows {
            let abs = mat[(i, j)].abs();
            if abs > max_val {
                max_val = abs;
            }
        }
    }

    max_val
}

/// Fills the lower triangle with the transpose of the upper triangle.
///
/// Useful for completing a symmetric matrix from its upper triangle.
pub fn symmetrize_upper<T: Scalar>(mat: &mut MatMut<'_, T>) {
    assert!(mat.is_square(), "Matrix must be square");
    let n = mat.nrows();

    for j in 0..n {
        for i in (j + 1)..n {
            mat.set(i, j, mat[(j, i)]);
        }
    }
}

/// Fills the upper triangle with the transpose of the lower triangle.
///
/// Useful for completing a symmetric matrix from its lower triangle.
pub fn symmetrize_lower<T: Scalar>(mat: &mut MatMut<'_, T>) {
    assert!(mat.is_square(), "Matrix must be square");
    let n = mat.nrows();

    for i in 0..n {
        for j in (i + 1)..n {
            mat.set(i, j, mat[(j, i)]);
        }
    }
}

/// Zeroes out the strictly lower triangle.
pub fn zero_lower<T: Scalar + num_traits::Zero>(mat: &mut MatMut<'_, T>) {
    let (nrows, ncols) = mat.shape();

    for j in 0..ncols.min(nrows) {
        for i in (j + 1)..nrows {
            mat.set(i, j, T::zero());
        }
    }
}

/// Zeroes out the strictly upper triangle.
pub fn zero_upper<T: Scalar + num_traits::Zero>(mat: &mut MatMut<'_, T>) {
    let (nrows, ncols) = mat.shape();

    for i in 0..nrows.min(ncols) {
        for j in (i + 1)..ncols {
            mat.set(i, j, T::zero());
        }
    }
}

/// Horizontally concatenates two matrices.
///
/// Both matrices must have the same number of rows.
pub fn hcat<T: Scalar + bytemuck::Zeroable>(a: &MatRef<'_, T>, b: &MatRef<'_, T>) -> Mat<T> {
    assert_eq!(a.nrows(), b.nrows(), "Row counts must match for hcat");

    let nrows = a.nrows();
    let ncols = a.ncols() + b.ncols();
    let mut result = Mat::zeros(nrows, ncols);

    // Copy A
    for j in 0..a.ncols() {
        for i in 0..nrows {
            result[(i, j)] = a[(i, j)];
        }
    }

    // Copy B
    for j in 0..b.ncols() {
        for i in 0..nrows {
            result[(i, a.ncols() + j)] = b[(i, j)];
        }
    }

    result
}

/// Vertically concatenates two matrices.
///
/// Both matrices must have the same number of columns.
pub fn vcat<T: Scalar + bytemuck::Zeroable>(a: &MatRef<'_, T>, b: &MatRef<'_, T>) -> Mat<T> {
    assert_eq!(a.ncols(), b.ncols(), "Column counts must match for vcat");

    let nrows = a.nrows() + b.nrows();
    let ncols = a.ncols();
    let mut result = Mat::zeros(nrows, ncols);

    // Copy A
    for j in 0..ncols {
        for i in 0..a.nrows() {
            result[(i, j)] = a[(i, j)];
        }
    }

    // Copy B
    for j in 0..ncols {
        for i in 0..b.nrows() {
            result[(a.nrows() + i, j)] = b[(i, j)];
        }
    }

    result
}

/// Stacks matrices horizontally from a slice.
pub fn hstack<T: Scalar + bytemuck::Zeroable>(mats: &[&MatRef<'_, T>]) -> Mat<T> {
    if mats.is_empty() {
        return Mat::zeros(0, 0);
    }

    let nrows = mats[0].nrows();
    let ncols: usize = mats.iter().map(|m| m.ncols()).sum();

    let mut result = Mat::zeros(nrows, ncols);
    let mut col_offset = 0;

    for mat in mats {
        assert_eq!(mat.nrows(), nrows, "All matrices must have same row count");

        for j in 0..mat.ncols() {
            for i in 0..nrows {
                result[(i, col_offset + j)] = mat[(i, j)];
            }
        }
        col_offset += mat.ncols();
    }

    result
}

/// Stacks matrices vertically from a slice.
pub fn vstack<T: Scalar + bytemuck::Zeroable>(mats: &[&MatRef<'_, T>]) -> Mat<T> {
    if mats.is_empty() {
        return Mat::zeros(0, 0);
    }

    let ncols = mats[0].ncols();
    let nrows: usize = mats.iter().map(|m| m.nrows()).sum();

    let mut result = Mat::zeros(nrows, ncols);
    let mut row_offset = 0;

    for mat in mats {
        assert_eq!(
            mat.ncols(),
            ncols,
            "All matrices must have same column count"
        );

        for j in 0..ncols {
            for i in 0..mat.nrows() {
                result[(row_offset + i, j)] = mat[(i, j)];
            }
        }
        row_offset += mat.nrows();
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_inplace() {
        let mut m = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        transpose_inplace(&mut m);

        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 1)], 4.0); // Was (1, 0)
        assert_eq!(m[(0, 2)], 7.0); // Was (2, 0)
        assert_eq!(m[(1, 0)], 2.0); // Was (0, 1)
        assert_eq!(m[(2, 1)], 6.0); // Was (1, 2)
    }

    #[test]
    fn test_extract_diagonal() {
        let m = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        let diag = extract_diagonal(&m.as_ref());
        assert_eq!(diag, vec![1.0, 5.0, 9.0]);
    }

    #[test]
    fn test_set_diagonal() {
        let mut m: Mat<f64> = Mat::zeros(3, 3);
        set_diagonal(&mut m.as_mut(), &[1.0, 2.0, 3.0]);

        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 1)], 2.0);
        assert_eq!(m[(2, 2)], 3.0);
        assert_eq!(m[(0, 1)], 0.0);
    }

    #[test]
    fn test_add_to_diagonal() {
        let mut m = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);

        add_to_diagonal(&mut m.as_mut(), 10.0);

        assert_eq!(m[(0, 0)], 11.0);
        assert_eq!(m[(1, 1)], 14.0);
        assert_eq!(m[(0, 1)], 2.0); // Unchanged
    }

    #[test]
    fn test_extract_kth_diagonal() {
        let m = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        // Main diagonal
        assert_eq!(extract_kth_diagonal(&m.as_ref(), 0), vec![1.0, 5.0, 9.0]);

        // Superdiagonal
        assert_eq!(extract_kth_diagonal(&m.as_ref(), 1), vec![2.0, 6.0]);

        // Second superdiagonal
        assert_eq!(extract_kth_diagonal(&m.as_ref(), 2), vec![3.0]);

        // Subdiagonal
        assert_eq!(extract_kth_diagonal(&m.as_ref(), -1), vec![4.0, 8.0]);

        // Second subdiagonal
        assert_eq!(extract_kth_diagonal(&m.as_ref(), -2), vec![7.0]);
    }

    #[test]
    fn test_permute_rows() {
        let m = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        // Reverse rows
        let perm = [2, 1, 0];
        let permuted = permute_rows(&m.as_ref(), &perm);

        assert_eq!(permuted[(0, 0)], 5.0);
        assert_eq!(permuted[(1, 0)], 3.0);
        assert_eq!(permuted[(2, 0)], 1.0);
    }

    #[test]
    fn test_permute_cols() {
        let m = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        // Reverse columns
        let perm = [2, 1, 0];
        let permuted = permute_cols(&m.as_ref(), &perm);

        assert_eq!(permuted[(0, 0)], 3.0);
        assert_eq!(permuted[(0, 1)], 2.0);
        assert_eq!(permuted[(0, 2)], 1.0);
    }

    #[test]
    fn test_extract_block() {
        let m = Mat::from_rows(&[
            &[1.0, 2.0, 3.0, 4.0],
            &[5.0, 6.0, 7.0, 8.0],
            &[9.0, 10.0, 11.0, 12.0],
        ]);

        let block = extract_block(&m.as_ref(), 1, 1, 2, 2);

        assert_eq!(block.shape(), (2, 2));
        assert_eq!(block[(0, 0)], 6.0);
        assert_eq!(block[(0, 1)], 7.0);
        assert_eq!(block[(1, 0)], 10.0);
        assert_eq!(block[(1, 1)], 11.0);
    }

    #[test]
    fn test_set_block() {
        let mut m: Mat<f64> = Mat::zeros(4, 4);
        let block = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);

        set_block(&mut m.as_mut(), 1, 1, &block.as_ref());

        assert_eq!(m[(1, 1)], 1.0);
        assert_eq!(m[(1, 2)], 2.0);
        assert_eq!(m[(2, 1)], 3.0);
        assert_eq!(m[(2, 2)], 4.0);
        assert_eq!(m[(0, 0)], 0.0);
    }

    #[test]
    fn test_trace() {
        let m = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        assert_eq!(trace(&m.as_ref()), 15.0); // 1 + 5 + 9
    }

    #[test]
    fn test_frobenius_norm_squared() {
        let m = Mat::from_rows(&[&[1.0_f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        // 1 + 4 + 9 + 16 + 25 + 36 = 91
        let norm_sq: f64 = frobenius_norm_squared(&m.as_ref());
        assert!((norm_sq - 91.0).abs() < 1e-10);
    }

    #[test]
    fn test_symmetrize_upper() {
        let mut m = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[0.0, 4.0, 5.0], &[0.0, 0.0, 6.0]]);

        symmetrize_upper(&mut m.as_mut());

        // Lower triangle should mirror upper
        assert_eq!(m[(1, 0)], 2.0);
        assert_eq!(m[(2, 0)], 3.0);
        assert_eq!(m[(2, 1)], 5.0);
    }

    #[test]
    fn test_symmetrize_lower() {
        let mut m = Mat::from_rows(&[&[1.0, 0.0, 0.0], &[2.0, 4.0, 0.0], &[3.0, 5.0, 6.0]]);

        symmetrize_lower(&mut m.as_mut());

        // Upper triangle should mirror lower
        assert_eq!(m[(0, 1)], 2.0);
        assert_eq!(m[(0, 2)], 3.0);
        assert_eq!(m[(1, 2)], 5.0);
    }

    #[test]
    fn test_zero_lower() {
        let mut m = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        zero_lower(&mut m.as_mut());

        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 1)], 2.0);
        assert_eq!(m[(1, 0)], 0.0);
        assert_eq!(m[(2, 0)], 0.0);
        assert_eq!(m[(2, 1)], 0.0);
    }

    #[test]
    fn test_hcat() {
        let a = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[5.0], &[6.0]]);

        let result = hcat(&a.as_ref(), &b.as_ref());

        assert_eq!(result.shape(), (2, 3));
        assert_eq!(result[(0, 0)], 1.0);
        assert_eq!(result[(0, 2)], 5.0);
        assert_eq!(result[(1, 2)], 6.0);
    }

    #[test]
    fn test_vcat() {
        let a = Mat::from_rows(&[&[1.0, 2.0]]);
        let b = Mat::from_rows(&[&[3.0, 4.0], &[5.0, 6.0]]);

        let result = vcat(&a.as_ref(), &b.as_ref());

        assert_eq!(result.shape(), (3, 2));
        assert_eq!(result[(0, 0)], 1.0);
        assert_eq!(result[(1, 0)], 3.0);
        assert_eq!(result[(2, 1)], 6.0);
    }

    #[test]
    fn test_axpy() {
        let mut dst = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let src = Mat::from_rows(&[&[10.0, 20.0], &[30.0, 40.0]]);

        // dst = 2.0 * src + 0.5 * dst
        axpy(&mut dst.as_mut(), 2.0, &src.as_ref(), 0.5);

        assert_eq!(dst[(0, 0)], 2.0 * 10.0 + 0.5 * 1.0); // 20.5
        assert_eq!(dst[(1, 1)], 2.0 * 40.0 + 0.5 * 4.0); // 82.0
    }
}
