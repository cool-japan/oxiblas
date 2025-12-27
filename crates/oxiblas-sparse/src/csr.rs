//! Compressed Sparse Row (CSR) matrix format.
//!
//! CSR stores matrix data using three arrays:
//! - `values`: Non-zero values (row-major order)
//! - `col_indices`: Column index for each value
//! - `row_ptrs`: Index into values/col_indices for start of each row
//!
//! For an m×n matrix with nnz non-zeros:
//! - `values` has length nnz
//! - `col_indices` has length nnz
//! - `row_ptrs` has length m+1

use oxiblas_core::scalar::{Field, Scalar};
use std::ops::Index;

/// Error type for CSR matrix operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CsrError {
    /// Invalid row pointer array length.
    InvalidRowPtrs {
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },
    /// Mismatched array lengths.
    LengthMismatch {
        /// Number of values.
        values_len: usize,
        /// Number of column indices.
        col_indices_len: usize,
    },
    /// Column index out of bounds.
    InvalidColumnIndex {
        /// The invalid index.
        index: usize,
        /// Number of columns.
        ncols: usize,
    },
    /// Row pointers not monotonically increasing.
    InvalidRowPtrOrder,
    /// Duplicate entry at same position.
    DuplicateEntry {
        /// Row of duplicate.
        row: usize,
        /// Column of duplicate.
        col: usize,
    },
}

impl core::fmt::Display for CsrError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidRowPtrs { expected, actual } => {
                write!(
                    f,
                    "Invalid row_ptrs length: expected {expected}, got {actual}"
                )
            }
            Self::LengthMismatch {
                values_len,
                col_indices_len,
            } => {
                write!(
                    f,
                    "Length mismatch: values={values_len}, col_indices={col_indices_len}"
                )
            }
            Self::InvalidColumnIndex { index, ncols } => {
                write!(f, "Column index {index} out of bounds for {ncols} columns")
            }
            Self::InvalidRowPtrOrder => {
                write!(f, "Row pointers must be monotonically increasing")
            }
            Self::DuplicateEntry { row, col } => {
                write!(f, "Duplicate entry at ({row}, {col})")
            }
        }
    }
}

impl std::error::Error for CsrError {}

/// Compressed Sparse Row matrix.
///
/// Efficient for:
/// - Row slicing
/// - Matrix-vector products (y = A*x)
/// - Row-wise traversal
#[derive(Debug, Clone)]
pub struct CsrMatrix<T: Scalar> {
    /// Number of rows.
    nrows: usize,
    /// Number of columns.
    ncols: usize,
    /// Row pointers (length nrows + 1).
    row_ptrs: Vec<usize>,
    /// Column indices for each non-zero.
    col_indices: Vec<usize>,
    /// Non-zero values.
    values: Vec<T>,
}

impl<T: Scalar + Clone> CsrMatrix<T> {
    /// Creates a new CSR matrix from raw components.
    ///
    /// # Arguments
    ///
    /// * `nrows` - Number of rows
    /// * `ncols` - Number of columns
    /// * `row_ptrs` - Row pointers (length nrows + 1)
    /// * `col_indices` - Column indices for each non-zero
    /// * `values` - Non-zero values
    ///
    /// # Errors
    ///
    /// Returns an error if the input is invalid.
    pub fn new(
        nrows: usize,
        ncols: usize,
        row_ptrs: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self, CsrError> {
        // Validate row_ptrs length
        if row_ptrs.len() != nrows + 1 {
            return Err(CsrError::InvalidRowPtrs {
                expected: nrows + 1,
                actual: row_ptrs.len(),
            });
        }

        // Validate values and col_indices have same length
        if values.len() != col_indices.len() {
            return Err(CsrError::LengthMismatch {
                values_len: values.len(),
                col_indices_len: col_indices.len(),
            });
        }

        // Validate row_ptrs are monotonically increasing
        for i in 1..row_ptrs.len() {
            if row_ptrs[i] < row_ptrs[i - 1] {
                return Err(CsrError::InvalidRowPtrOrder);
            }
        }

        // Validate row_ptrs[nrows] equals nnz
        let nnz = values.len();
        if row_ptrs[nrows] != nnz {
            return Err(CsrError::InvalidRowPtrs {
                expected: nnz,
                actual: row_ptrs[nrows],
            });
        }

        // Validate column indices
        for &col in &col_indices {
            if col >= ncols {
                return Err(CsrError::InvalidColumnIndex { index: col, ncols });
            }
        }

        Ok(Self {
            nrows,
            ncols,
            row_ptrs,
            col_indices,
            values,
        })
    }

    /// Creates a CSR matrix without validation (unsafe but faster).
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `row_ptrs.len() == nrows + 1`
    /// - `values.len() == col_indices.len()`
    /// - `row_ptrs` is monotonically increasing
    /// - All column indices are < ncols
    #[inline]
    pub unsafe fn new_unchecked(
        nrows: usize,
        ncols: usize,
        row_ptrs: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Self {
        Self {
            nrows,
            ncols,
            row_ptrs,
            col_indices,
            values,
        }
    }

    /// Creates an empty CSR matrix with given dimensions.
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            row_ptrs: vec![0; nrows + 1],
            col_indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Creates an identity matrix in CSR format.
    pub fn eye(n: usize) -> Self
    where
        T: Field,
    {
        let mut row_ptrs = Vec::with_capacity(n + 1);
        let mut col_indices = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);

        for i in 0..n {
            row_ptrs.push(i);
            col_indices.push(i);
            values.push(T::one());
        }
        row_ptrs.push(n);

        Self {
            nrows: n,
            ncols: n,
            row_ptrs,
            col_indices,
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

    /// Returns a reference to the row pointers.
    #[inline]
    pub fn row_ptrs(&self) -> &[usize] {
        &self.row_ptrs
    }

    /// Returns a reference to the column indices.
    #[inline]
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
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

        let start = self.row_ptrs[row];
        let end = self.row_ptrs[row + 1];

        for i in start..end {
            if self.col_indices[i] == col {
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

    /// Returns an iterator over the non-zeros in a row.
    pub fn row_iter(&self, row: usize) -> impl Iterator<Item = (usize, &T)> {
        let start = self.row_ptrs[row];
        let end = self.row_ptrs[row + 1];

        self.col_indices[start..end]
            .iter()
            .zip(self.values[start..end].iter())
            .map(|(&col, val)| (col, val))
    }

    /// Returns an iterator over all non-zeros as (row, col, value).
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, &T)> {
        (0..self.nrows).flat_map(move |row| {
            let start = self.row_ptrs[row];
            let end = self.row_ptrs[row + 1];

            self.col_indices[start..end]
                .iter()
                .zip(self.values[start..end].iter())
                .map(move |(&col, val)| (row, col, val))
        })
    }

    /// Converts to CSC format.
    pub fn to_csc(&self) -> crate::csc::CscMatrix<T> {
        crate::convert::csr_to_csc(self)
    }

    /// Converts to dense matrix.
    pub fn to_dense(&self) -> oxiblas_matrix::Mat<T>
    where
        T: Field + bytemuck::Zeroable,
    {
        let mut dense = oxiblas_matrix::Mat::zeros(self.nrows, self.ncols);

        for row in 0..self.nrows {
            let start = self.row_ptrs[row];
            let end = self.row_ptrs[row + 1];

            for i in start..end {
                dense[(row, self.col_indices[i])] = self.values[i].clone();
            }
        }

        dense
    }

    /// Creates a CSR matrix from a dense matrix.
    pub fn from_dense(dense: &oxiblas_matrix::MatRef<'_, T>) -> Self
    where
        T: Field,
    {
        let (nrows, ncols) = dense.shape();
        let mut row_ptrs = Vec::with_capacity(nrows + 1);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        let eps = <T as Scalar>::epsilon();

        row_ptrs.push(0);

        for i in 0..nrows {
            for j in 0..ncols {
                let val = dense[(i, j)].clone();
                if Scalar::abs(val.clone()) > eps {
                    col_indices.push(j);
                    values.push(val);
                }
            }
            row_ptrs.push(values.len());
        }

        Self {
            nrows,
            ncols,
            row_ptrs,
            col_indices,
            values,
        }
    }

    /// Returns the transpose of this matrix.
    pub fn transpose(&self) -> Self {
        // A^T in CSR is formed by reinterpreting A's CSC as CSR with swapped dimensions
        // CSC of A: col_ptrs, row_indices, values for m×n matrix
        // CSR of A^T: row_ptrs, col_indices, values for n×m matrix
        //   row_ptrs = col_ptrs
        //   col_indices = row_indices
        //   nrows = ncols, ncols = nrows

        let csc = self.to_csc();

        // Reinterpret CSC of A as CSR of A^T
        // Safety: CSC arrays form valid CSR when dimensions are swapped
        unsafe {
            Self::new_unchecked(
                csc.ncols(),                // new nrows = original ncols
                csc.nrows(),                // new ncols = original nrows
                csc.col_ptrs().to_vec(),    // col_ptrs become row_ptrs
                csc.row_indices().to_vec(), // row_indices become col_indices
                csc.values().to_vec(),
            )
        }
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
}

impl<T: Scalar + Clone> Index<(usize, usize)> for CsrMatrix<T>
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
    fn test_csr_new() {
        // [1 0 2]
        // [0 3 0]
        // [4 0 5]
        let values = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let col_indices = vec![0, 2, 1, 0, 2];
        let row_ptrs = vec![0, 2, 3, 5];

        let csr = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        assert_eq!(csr.nrows(), 3);
        assert_eq!(csr.ncols(), 3);
        assert_eq!(csr.nnz(), 5);
    }

    #[test]
    fn test_csr_get() {
        let values = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let col_indices = vec![0, 2, 1, 0, 2];
        let row_ptrs = vec![0, 2, 3, 5];

        let csr = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

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
    fn test_csr_zeros() {
        let csr: CsrMatrix<f64> = CsrMatrix::zeros(5, 3);

        assert_eq!(csr.nrows(), 5);
        assert_eq!(csr.ncols(), 3);
        assert_eq!(csr.nnz(), 0);
    }

    #[test]
    fn test_csr_eye() {
        let csr: CsrMatrix<f64> = CsrMatrix::eye(4);

        assert_eq!(csr.nrows(), 4);
        assert_eq!(csr.ncols(), 4);
        assert_eq!(csr.nnz(), 4);

        for i in 0..4 {
            assert_eq!(csr.get(i, i), Some(&1.0));
        }
    }

    #[test]
    fn test_csr_density() {
        let values = vec![1.0f64, 2.0, 3.0];
        let col_indices = vec![0, 1, 2];
        let row_ptrs = vec![0, 1, 2, 3];

        let csr = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let density = csr.density();
        assert!((density - 3.0 / 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_csr_row_iter() {
        let values = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let col_indices = vec![0, 2, 1, 0, 2];
        let row_ptrs = vec![0, 2, 3, 5];

        let csr = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        let row0: Vec<_> = csr.row_iter(0).collect();
        assert_eq!(row0, vec![(0, &1.0), (2, &2.0)]);

        let row1: Vec<_> = csr.row_iter(1).collect();
        assert_eq!(row1, vec![(1, &3.0)]);
    }

    #[test]
    fn test_csr_scale() {
        let values = vec![1.0f64, 2.0, 3.0];
        let col_indices = vec![0, 1, 2];
        let row_ptrs = vec![0, 1, 2, 3];

        let mut csr = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();
        csr.scale(2.0);

        assert_eq!(csr.values(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_csr_invalid_row_ptrs() {
        let values = vec![1.0f64, 2.0];
        let col_indices = vec![0, 1];
        let row_ptrs = vec![0, 1]; // Should have length 3 for 2 rows

        let result = CsrMatrix::new(2, 2, row_ptrs, col_indices, values);
        assert!(matches!(result, Err(CsrError::InvalidRowPtrs { .. })));
    }

    #[test]
    fn test_csr_invalid_col_index() {
        let values = vec![1.0f64];
        let col_indices = vec![5]; // Out of bounds
        let row_ptrs = vec![0, 1];

        let result = CsrMatrix::new(1, 3, row_ptrs, col_indices, values);
        assert!(matches!(result, Err(CsrError::InvalidColumnIndex { .. })));
    }
}
