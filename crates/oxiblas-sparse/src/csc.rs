//! Compressed Sparse Column (CSC) matrix format.
//!
//! CSC stores matrix data using three arrays:
//! - `values`: Non-zero values (column-major order)
//! - `row_indices`: Row index for each value
//! - `col_ptrs`: Index into values/row_indices for start of each column
//!
//! For an m×n matrix with nnz non-zeros:
//! - `values` has length nnz
//! - `row_indices` has length nnz
//! - `col_ptrs` has length n+1

use oxiblas_core::scalar::{Field, Scalar};
use std::ops::Index;

/// Error type for CSC matrix operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CscError {
    /// Invalid column pointer array length.
    InvalidColPtrs {
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },
    /// Mismatched array lengths.
    LengthMismatch {
        /// Number of values.
        values_len: usize,
        /// Number of row indices.
        row_indices_len: usize,
    },
    /// Row index out of bounds.
    InvalidRowIndex {
        /// The invalid index.
        index: usize,
        /// Number of rows.
        nrows: usize,
    },
    /// Column pointers not monotonically increasing.
    InvalidColPtrOrder,
    /// Duplicate entry at same position.
    DuplicateEntry {
        /// Row of duplicate.
        row: usize,
        /// Column of duplicate.
        col: usize,
    },
}

impl core::fmt::Display for CscError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidColPtrs { expected, actual } => {
                write!(
                    f,
                    "Invalid col_ptrs length: expected {expected}, got {actual}"
                )
            }
            Self::LengthMismatch {
                values_len,
                row_indices_len,
            } => {
                write!(
                    f,
                    "Length mismatch: values={values_len}, row_indices={row_indices_len}"
                )
            }
            Self::InvalidRowIndex { index, nrows } => {
                write!(f, "Row index {index} out of bounds for {nrows} rows")
            }
            Self::InvalidColPtrOrder => {
                write!(f, "Column pointers must be monotonically increasing")
            }
            Self::DuplicateEntry { row, col } => {
                write!(f, "Duplicate entry at ({row}, {col})")
            }
        }
    }
}

impl std::error::Error for CscError {}

/// Compressed Sparse Column matrix.
///
/// Efficient for:
/// - Column slicing
/// - Matrix-vector products with transpose (y = A^T * x)
/// - Column-wise traversal
/// - Direct solvers (LU, Cholesky)
#[derive(Debug, Clone)]
pub struct CscMatrix<T: Scalar> {
    /// Number of rows.
    nrows: usize,
    /// Number of columns.
    ncols: usize,
    /// Column pointers (length ncols + 1).
    col_ptrs: Vec<usize>,
    /// Row indices for each non-zero.
    row_indices: Vec<usize>,
    /// Non-zero values.
    values: Vec<T>,
}

impl<T: Scalar + Clone> CscMatrix<T> {
    /// Creates a new CSC matrix from raw components.
    ///
    /// # Arguments
    ///
    /// * `nrows` - Number of rows
    /// * `ncols` - Number of columns
    /// * `col_ptrs` - Column pointers (length ncols + 1)
    /// * `row_indices` - Row indices for each non-zero
    /// * `values` - Non-zero values
    ///
    /// # Errors
    ///
    /// Returns an error if the input is invalid.
    pub fn new(
        nrows: usize,
        ncols: usize,
        col_ptrs: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self, CscError> {
        // Validate col_ptrs length
        if col_ptrs.len() != ncols + 1 {
            return Err(CscError::InvalidColPtrs {
                expected: ncols + 1,
                actual: col_ptrs.len(),
            });
        }

        // Validate values and row_indices have same length
        if values.len() != row_indices.len() {
            return Err(CscError::LengthMismatch {
                values_len: values.len(),
                row_indices_len: row_indices.len(),
            });
        }

        // Validate col_ptrs are monotonically increasing
        for i in 1..col_ptrs.len() {
            if col_ptrs[i] < col_ptrs[i - 1] {
                return Err(CscError::InvalidColPtrOrder);
            }
        }

        // Validate col_ptrs[ncols] equals nnz
        let nnz = values.len();
        if col_ptrs[ncols] != nnz {
            return Err(CscError::InvalidColPtrs {
                expected: nnz,
                actual: col_ptrs[ncols],
            });
        }

        // Validate row indices
        for &row in &row_indices {
            if row >= nrows {
                return Err(CscError::InvalidRowIndex { index: row, nrows });
            }
        }

        Ok(Self {
            nrows,
            ncols,
            col_ptrs,
            row_indices,
            values,
        })
    }

    /// Creates a CSC matrix without validation (unsafe but faster).
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `col_ptrs.len() == ncols + 1`
    /// - `values.len() == row_indices.len()`
    /// - `col_ptrs` is monotonically increasing
    /// - All row indices are < nrows
    #[inline]
    pub unsafe fn new_unchecked(
        nrows: usize,
        ncols: usize,
        col_ptrs: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Self {
        Self {
            nrows,
            ncols,
            col_ptrs,
            row_indices,
            values,
        }
    }

    /// Creates an empty CSC matrix with given dimensions.
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            col_ptrs: vec![0; ncols + 1],
            row_indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Creates an identity matrix in CSC format.
    pub fn eye(n: usize) -> Self
    where
        T: Field,
    {
        let mut col_ptrs = Vec::with_capacity(n + 1);
        let mut row_indices = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);

        for i in 0..n {
            col_ptrs.push(i);
            row_indices.push(i);
            values.push(T::one());
        }
        col_ptrs.push(n);

        Self {
            nrows: n,
            ncols: n,
            col_ptrs,
            row_indices,
            values,
        }
    }

    /// Returns the number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Returns the number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Returns the shape (nrows, ncols).
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Returns the number of non-zero elements.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Returns the density (nnz / total_elements).
    #[inline]
    pub fn density(&self) -> f64 {
        if self.nrows == 0 || self.ncols == 0 {
            0.0
        } else {
            self.nnz() as f64 / (self.nrows * self.ncols) as f64
        }
    }

    /// Returns a reference to the column pointers.
    #[inline]
    pub fn col_ptrs(&self) -> &[usize] {
        &self.col_ptrs
    }

    /// Returns a reference to the row indices.
    #[inline]
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    /// Returns a reference to the values.
    #[inline]
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Returns a mutable reference to the values.
    #[inline]
    pub fn values_mut(&mut self) -> &mut [T] {
        &mut self.values
    }

    /// Gets the value at (row, col), returning None if not present.
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row >= self.nrows || col >= self.ncols {
            return None;
        }

        let start = self.col_ptrs[col];
        let end = self.col_ptrs[col + 1];

        for i in start..end {
            if self.row_indices[i] == row {
                return Some(&self.values[i]);
            }
        }

        None
    }

    /// Gets the value at (row, col), returning zero if not present.
    pub fn get_or_zero(&self, row: usize, col: usize) -> T
    where
        T: Field,
    {
        self.get(row, col).cloned().unwrap_or_else(T::zero)
    }

    /// Returns an iterator over the non-zeros in a column.
    pub fn col_iter(&self, col: usize) -> impl Iterator<Item = (usize, &T)> {
        let start = self.col_ptrs[col];
        let end = self.col_ptrs[col + 1];

        self.row_indices[start..end]
            .iter()
            .zip(self.values[start..end].iter())
            .map(|(&row, val)| (row, val))
    }

    /// Returns an iterator over all non-zeros as (row, col, value).
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, &T)> {
        (0..self.ncols).flat_map(move |col| {
            let start = self.col_ptrs[col];
            let end = self.col_ptrs[col + 1];

            self.row_indices[start..end]
                .iter()
                .zip(self.values[start..end].iter())
                .map(move |(&row, val)| (row, col, val))
        })
    }

    /// Converts to CSR format.
    pub fn to_csr(&self) -> crate::csr::CsrMatrix<T> {
        crate::convert::csc_to_csr(self)
    }

    /// Converts to dense matrix.
    pub fn to_dense(&self) -> oxiblas_matrix::Mat<T>
    where
        T: Field + bytemuck::Zeroable,
    {
        let mut dense = oxiblas_matrix::Mat::zeros(self.nrows, self.ncols);

        for col in 0..self.ncols {
            let start = self.col_ptrs[col];
            let end = self.col_ptrs[col + 1];

            for i in start..end {
                dense[(self.row_indices[i], col)] = self.values[i].clone();
            }
        }

        dense
    }

    /// Creates a CSC matrix from a dense matrix.
    pub fn from_dense(dense: &oxiblas_matrix::MatRef<'_, T>) -> Self
    where
        T: Field,
    {
        let (nrows, ncols) = dense.shape();
        let mut col_ptrs = Vec::with_capacity(ncols + 1);
        let mut row_indices = Vec::new();
        let mut values = Vec::new();

        let eps = <T as Scalar>::epsilon();

        col_ptrs.push(0);

        for j in 0..ncols {
            for i in 0..nrows {
                let val = dense[(i, j)].clone();
                if Scalar::abs(val.clone()) > eps {
                    row_indices.push(i);
                    values.push(val);
                }
            }
            col_ptrs.push(values.len());
        }

        Self {
            nrows,
            ncols,
            col_ptrs,
            row_indices,
            values,
        }
    }

    /// Returns the transpose of this matrix.
    pub fn transpose(&self) -> Self {
        // Transpose of CSC is equivalent to CSR with swapped interpretation
        // Then convert back to CSC
        let csr = self.to_csr();
        csr.to_csc()
    }

    /// Scales all values by a scalar.
    pub fn scale(&mut self, alpha: T) {
        for val in &mut self.values {
            *val = val.clone() * alpha.clone();
        }
    }

    /// Returns a scaled copy of this matrix.
    pub fn scaled(&self, alpha: T) -> Self {
        let mut result = self.clone();
        result.scale(alpha);
        result
    }

    /// Returns the number of non-zeros in a column.
    #[inline]
    pub fn col_nnz(&self, col: usize) -> usize {
        self.col_ptrs[col + 1] - self.col_ptrs[col]
    }

    /// Checks if the matrix is structurally symmetric.
    ///
    /// Returns true if A\[i,j\] != 0 implies A\[j,i\] != 0.
    pub fn is_structurally_symmetric(&self) -> bool {
        if self.nrows != self.ncols {
            return false;
        }

        for col in 0..self.ncols {
            let start = self.col_ptrs[col];
            let end = self.col_ptrs[col + 1];

            for i in start..end {
                let row = self.row_indices[i];
                if self.get(col, row).is_none() {
                    return false;
                }
            }
        }

        true
    }
}

impl<T: Scalar + Clone> Index<(usize, usize)> for CscMatrix<T>
where
    T: Field,
{
    type Output = T;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        self.get(row, col)
            .expect("Index out of bounds or zero element")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csc_new() {
        // [1 0 4]
        // [0 3 0]
        // [2 0 5]
        // Column 0: 1, 2 at rows 0, 2
        // Column 1: 3 at row 1
        // Column 2: 4, 5 at rows 0, 2
        let values = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let row_indices = vec![0, 2, 1, 0, 2];
        let col_ptrs = vec![0, 2, 3, 5];

        let csc = CscMatrix::new(3, 3, col_ptrs, row_indices, values).unwrap();

        assert_eq!(csc.nrows(), 3);
        assert_eq!(csc.ncols(), 3);
        assert_eq!(csc.nnz(), 5);
    }

    #[test]
    fn test_csc_get() {
        let values = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let row_indices = vec![0, 2, 1, 0, 2];
        let col_ptrs = vec![0, 2, 3, 5];

        let csc = CscMatrix::new(3, 3, col_ptrs, row_indices, values).unwrap();

        assert_eq!(csc.get(0, 0), Some(&1.0));
        assert_eq!(csc.get(2, 0), Some(&2.0));
        assert_eq!(csc.get(1, 1), Some(&3.0));
        assert_eq!(csc.get(0, 2), Some(&4.0));
        assert_eq!(csc.get(2, 2), Some(&5.0));

        // Zero elements
        assert_eq!(csc.get(1, 0), None);
        assert_eq!(csc.get(0, 1), None);
    }

    #[test]
    fn test_csc_zeros() {
        let csc: CscMatrix<f64> = CscMatrix::zeros(5, 3);

        assert_eq!(csc.nrows(), 5);
        assert_eq!(csc.ncols(), 3);
        assert_eq!(csc.nnz(), 0);
    }

    #[test]
    fn test_csc_eye() {
        let csc: CscMatrix<f64> = CscMatrix::eye(4);

        assert_eq!(csc.nrows(), 4);
        assert_eq!(csc.ncols(), 4);
        assert_eq!(csc.nnz(), 4);

        for i in 0..4 {
            assert_eq!(csc.get(i, i), Some(&1.0));
        }
    }

    #[test]
    fn test_csc_density() {
        let values = vec![1.0f64, 2.0, 3.0];
        let row_indices = vec![0, 1, 2];
        let col_ptrs = vec![0, 1, 2, 3];

        let csc = CscMatrix::new(3, 3, col_ptrs, row_indices, values).unwrap();

        let density = csc.density();
        assert!((density - 3.0 / 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_csc_col_iter() {
        let values = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let row_indices = vec![0, 2, 1, 0, 2];
        let col_ptrs = vec![0, 2, 3, 5];

        let csc = CscMatrix::new(3, 3, col_ptrs, row_indices, values).unwrap();

        let col0: Vec<_> = csc.col_iter(0).collect();
        assert_eq!(col0, vec![(0, &1.0), (2, &2.0)]);

        let col1: Vec<_> = csc.col_iter(1).collect();
        assert_eq!(col1, vec![(1, &3.0)]);
    }

    #[test]
    fn test_csc_scale() {
        let values = vec![1.0f64, 2.0, 3.0];
        let row_indices = vec![0, 1, 2];
        let col_ptrs = vec![0, 1, 2, 3];

        let mut csc = CscMatrix::new(3, 3, col_ptrs, row_indices, values).unwrap();
        csc.scale(2.0);

        assert_eq!(csc.values(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_csc_invalid_col_ptrs() {
        let values = vec![1.0f64, 2.0];
        let row_indices = vec![0, 1];
        let col_ptrs = vec![0, 1]; // Should have length 3 for 2 columns

        let result = CscMatrix::new(2, 2, col_ptrs, row_indices, values);
        assert!(matches!(result, Err(CscError::InvalidColPtrs { .. })));
    }

    #[test]
    fn test_csc_invalid_row_index() {
        let values = vec![1.0f64];
        let row_indices = vec![5]; // Out of bounds
        let col_ptrs = vec![0, 1];

        let result = CscMatrix::new(3, 1, col_ptrs, row_indices, values);
        assert!(matches!(result, Err(CscError::InvalidRowIndex { .. })));
    }

    #[test]
    fn test_csc_col_nnz() {
        let values = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let row_indices = vec![0, 2, 1, 0, 2];
        let col_ptrs = vec![0, 2, 3, 5];

        let csc = CscMatrix::new(3, 3, col_ptrs, row_indices, values).unwrap();

        assert_eq!(csc.col_nnz(0), 2);
        assert_eq!(csc.col_nnz(1), 1);
        assert_eq!(csc.col_nnz(2), 2);
    }

    #[test]
    fn test_csc_structurally_symmetric() {
        // Symmetric pattern
        let values = vec![1.0f64, 2.0, 2.0, 3.0];
        let row_indices = vec![0, 1, 0, 1];
        let col_ptrs = vec![0, 2, 4];

        let csc = CscMatrix::new(2, 2, col_ptrs, row_indices, values).unwrap();
        assert!(csc.is_structurally_symmetric());
    }
}
