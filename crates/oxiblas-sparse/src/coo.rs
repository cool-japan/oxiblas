//! Coordinate (COO) format for sparse matrices.
//!
//! COO stores matrix data as triplets (row, col, value).
//! This format is ideal for:
//! - Matrix construction/assembly
//! - Adding entries incrementally
//! - Converting to CSR/CSC formats
//!
//! For an m×n matrix with nnz non-zeros:
//! - `row_indices` has length nnz
//! - `col_indices` has length nnz
//! - `values` has length nnz

use oxiblas_core::scalar::{Field, Real, Scalar};

/// Error type for COO matrix operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CooError {
    /// Mismatched array lengths.
    LengthMismatch {
        /// Number of rows.
        rows_len: usize,
        /// Number of columns.
        cols_len: usize,
        /// Number of values.
        values_len: usize,
    },
    /// Row index out of bounds.
    InvalidRowIndex {
        /// The invalid index.
        index: usize,
        /// Number of rows.
        nrows: usize,
    },
    /// Column index out of bounds.
    InvalidColumnIndex {
        /// The invalid index.
        index: usize,
        /// Number of columns.
        ncols: usize,
    },
}

impl core::fmt::Display for CooError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LengthMismatch {
                rows_len,
                cols_len,
                values_len,
            } => {
                write!(
                    f,
                    "Length mismatch: rows={rows_len}, cols={cols_len}, values={values_len}"
                )
            }
            Self::InvalidRowIndex { index, nrows } => {
                write!(f, "Row index {index} out of bounds for {nrows} rows")
            }
            Self::InvalidColumnIndex { index, ncols } => {
                write!(f, "Column index {index} out of bounds for {ncols} columns")
            }
        }
    }
}

impl std::error::Error for CooError {}

/// Coordinate format sparse matrix.
///
/// Stores entries as triplets (row, col, value).
/// May contain duplicate entries - these are summed during conversion.
#[derive(Debug, Clone)]
pub struct CooMatrix<T: Scalar> {
    /// Number of rows.
    nrows: usize,
    /// Number of columns.
    ncols: usize,
    /// Row indices.
    row_indices: Vec<usize>,
    /// Column indices.
    col_indices: Vec<usize>,
    /// Values.
    values: Vec<T>,
}

impl<T: Scalar + Clone> CooMatrix<T> {
    /// Creates a new COO matrix from raw components.
    ///
    /// # Arguments
    ///
    /// * `nrows` - Number of rows
    /// * `ncols` - Number of columns
    /// * `row_indices` - Row index for each entry
    /// * `col_indices` - Column index for each entry
    /// * `values` - Value for each entry
    ///
    /// # Errors
    ///
    /// Returns an error if the input is invalid.
    pub fn new(
        nrows: usize,
        ncols: usize,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self, CooError> {
        // Validate lengths match
        let len = values.len();
        if row_indices.len() != len || col_indices.len() != len {
            return Err(CooError::LengthMismatch {
                rows_len: row_indices.len(),
                cols_len: col_indices.len(),
                values_len: len,
            });
        }

        // Validate indices
        for &row in &row_indices {
            if row >= nrows {
                return Err(CooError::InvalidRowIndex { index: row, nrows });
            }
        }

        for &col in &col_indices {
            if col >= ncols {
                return Err(CooError::InvalidColumnIndex { index: col, ncols });
            }
        }

        Ok(Self {
            nrows,
            ncols,
            row_indices,
            col_indices,
            values,
        })
    }

    /// Creates a COO matrix without validation.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - All arrays have the same length
    /// - All row indices are < nrows
    /// - All column indices are < ncols
    #[inline]
    pub unsafe fn new_unchecked(
        nrows: usize,
        ncols: usize,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Self {
        Self {
            nrows,
            ncols,
            row_indices,
            col_indices,
            values,
        }
    }

    /// Creates an empty COO matrix.
    pub fn new_empty(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Creates an empty COO matrix with pre-allocated capacity.
    pub fn with_capacity(nrows: usize, ncols: usize, capacity: usize) -> Self {
        Self {
            nrows,
            ncols,
            row_indices: Vec::with_capacity(capacity),
            col_indices: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
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

    /// Returns the number of stored entries (may include duplicates).
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns true if the matrix has no entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns a reference to the row indices.
    #[inline]
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
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

    /// Adds an entry to the matrix.
    ///
    /// # Panics
    ///
    /// Panics if row >= nrows or col >= ncols.
    pub fn push(&mut self, row: usize, col: usize, value: T) {
        assert!(row < self.nrows, "Row index {row} out of bounds");
        assert!(col < self.ncols, "Column index {col} out of bounds");

        self.row_indices.push(row);
        self.col_indices.push(col);
        self.values.push(value);
    }

    /// Adds an entry without bounds checking.
    ///
    /// # Safety
    ///
    /// row must be < nrows and col must be < ncols.
    #[inline]
    pub unsafe fn push_unchecked(&mut self, row: usize, col: usize, value: T) {
        self.row_indices.push(row);
        self.col_indices.push(col);
        self.values.push(value);
    }

    /// Returns an iterator over entries as (row, col, value).
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, &T)> {
        self.row_indices
            .iter()
            .zip(self.col_indices.iter())
            .zip(self.values.iter())
            .map(|((&row, &col), val)| (row, col, val))
    }

    /// Returns a mutable iterator over entries.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (usize, usize, &mut T)> {
        self.row_indices
            .iter()
            .zip(self.col_indices.iter())
            .zip(self.values.iter_mut())
            .map(|((&row, &col), val)| (row, col, val))
    }

    /// Converts to CSR format, summing duplicate entries.
    pub fn to_csr(&self) -> crate::csr::CsrMatrix<T>
    where
        T: Scalar<Real = T> + Field + Real,
    {
        crate::convert::coo_to_csr(self)
    }

    /// Converts to CSC format, summing duplicate entries.
    pub fn to_csc(&self) -> crate::csc::CscMatrix<T>
    where
        T: Scalar<Real = T> + Field + Real,
    {
        crate::convert::coo_to_csc(self)
    }

    /// Converts to dense matrix.
    pub fn to_dense(&self) -> oxiblas_matrix::Mat<T>
    where
        T: Field + bytemuck::Zeroable,
    {
        let mut dense: oxiblas_matrix::Mat<T> = oxiblas_matrix::Mat::zeros(self.nrows, self.ncols);

        for (row, col, val) in self.iter() {
            let current = dense[(row, col)].clone();
            dense[(row, col)] = current + val.clone();
        }

        dense
    }

    /// Creates a COO matrix from a dense matrix.
    pub fn from_dense(dense: &oxiblas_matrix::MatRef<'_, T>) -> Self
    where
        T: Field,
    {
        let (nrows, ncols) = dense.shape();
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        let eps = <T as Scalar>::epsilon();

        for i in 0..nrows {
            for j in 0..ncols {
                let val = dense[(i, j)].clone();
                if Scalar::abs(val.clone()) > eps {
                    row_indices.push(i);
                    col_indices.push(j);
                    values.push(val);
                }
            }
        }

        Self {
            nrows,
            ncols,
            row_indices,
            col_indices,
            values,
        }
    }

    /// Removes duplicate entries by summing them.
    pub fn sum_duplicates(&mut self)
    where
        T: Field,
    {
        if self.is_empty() {
            return;
        }

        // Sort by (row, col)
        let mut indices: Vec<usize> = (0..self.len()).collect();
        indices.sort_by_key(|&i| (self.row_indices[i], self.col_indices[i]));

        let mut new_rows = Vec::with_capacity(self.len());
        let mut new_cols = Vec::with_capacity(self.len());
        let mut new_vals = Vec::with_capacity(self.len());

        let mut prev_row = self.row_indices[indices[0]];
        let mut prev_col = self.col_indices[indices[0]];
        let mut acc = self.values[indices[0]].clone();

        for &idx in &indices[1..] {
            let row = self.row_indices[idx];
            let col = self.col_indices[idx];

            if row == prev_row && col == prev_col {
                acc = acc + self.values[idx].clone();
            } else {
                if Scalar::abs(acc.clone()) > T::epsilon() {
                    new_rows.push(prev_row);
                    new_cols.push(prev_col);
                    new_vals.push(acc);
                }
                prev_row = row;
                prev_col = col;
                acc = self.values[idx].clone();
            }
        }

        // Push last entry
        if Scalar::abs(acc.clone()) > T::epsilon() {
            new_rows.push(prev_row);
            new_cols.push(prev_col);
            new_vals.push(acc);
        }

        self.row_indices = new_rows;
        self.col_indices = new_cols;
        self.values = new_vals;
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

    /// Reserves capacity for additional entries.
    pub fn reserve(&mut self, additional: usize) {
        self.row_indices.reserve(additional);
        self.col_indices.reserve(additional);
        self.values.reserve(additional);
    }

    /// Clears all entries but keeps dimensions.
    pub fn clear(&mut self) {
        self.row_indices.clear();
        self.col_indices.clear();
        self.values.clear();
    }
}

/// Builder for constructing COO matrices incrementally.
#[derive(Debug, Clone)]
pub struct CooMatrixBuilder<T: Scalar> {
    matrix: CooMatrix<T>,
}

impl<T: Scalar + Clone> CooMatrixBuilder<T> {
    /// Creates a new builder.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            matrix: CooMatrix::new_empty(nrows, ncols),
        }
    }

    /// Creates a new builder with pre-allocated capacity.
    pub fn with_capacity(nrows: usize, ncols: usize, capacity: usize) -> Self {
        Self {
            matrix: CooMatrix::with_capacity(nrows, ncols, capacity),
        }
    }

    /// Adds an entry to the matrix.
    pub fn add(&mut self, row: usize, col: usize, value: T) -> &mut Self {
        self.matrix.push(row, col, value);
        self
    }

    /// Adds a diagonal entry.
    pub fn add_diagonal(&mut self, index: usize, value: T) -> &mut Self {
        self.matrix.push(index, index, value);
        self
    }

    /// Adds entries for a dense block.
    pub fn add_block(
        &mut self,
        start_row: usize,
        start_col: usize,
        block: &oxiblas_matrix::MatRef<'_, T>,
    ) -> &mut Self
    where
        T: Field,
    {
        let (nrows, ncols) = block.shape();
        let eps = <T as Scalar>::epsilon();

        for i in 0..nrows {
            for j in 0..ncols {
                let val = block[(i, j)].clone();
                if Scalar::abs(val.clone()) > eps {
                    self.matrix.push(start_row + i, start_col + j, val);
                }
            }
        }
        self
    }

    /// Builds the COO matrix.
    pub fn build(self) -> CooMatrix<T> {
        self.matrix
    }

    /// Builds a CSR matrix, summing duplicates.
    pub fn build_csr(self) -> crate::csr::CsrMatrix<T>
    where
        T: Scalar<Real = T> + Field + Real,
    {
        self.matrix.to_csr()
    }

    /// Builds a CSC matrix, summing duplicates.
    pub fn build_csc(self) -> crate::csc::CscMatrix<T>
    where
        T: Scalar<Real = T> + Field + Real,
    {
        self.matrix.to_csc()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coo_new() {
        let row_indices = vec![0, 1, 2];
        let col_indices = vec![0, 1, 2];
        let values = vec![1.0f64, 2.0, 3.0];

        let coo = CooMatrix::new(3, 3, row_indices, col_indices, values).unwrap();

        assert_eq!(coo.nrows(), 3);
        assert_eq!(coo.ncols(), 3);
        assert_eq!(coo.len(), 3);
    }

    #[test]
    fn test_coo_push() {
        let mut coo = CooMatrix::<f64>::new_empty(3, 3);

        coo.push(0, 0, 1.0);
        coo.push(1, 1, 2.0);
        coo.push(2, 2, 3.0);

        assert_eq!(coo.len(), 3);
        assert_eq!(coo.values(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_coo_with_duplicates() {
        let row_indices = vec![0, 0, 1];
        let col_indices = vec![0, 0, 1];
        let values = vec![1.0f64, 2.0, 3.0];

        let mut coo = CooMatrix::new(2, 2, row_indices, col_indices, values).unwrap();
        coo.sum_duplicates();

        assert_eq!(coo.len(), 2);
        // (0,0) should be summed to 3.0
        let entries: Vec<_> = coo.iter().collect();
        assert_eq!(entries[0], (0, 0, &3.0));
        assert_eq!(entries[1], (1, 1, &3.0));
    }

    #[test]
    fn test_coo_iter() {
        let row_indices = vec![0, 1, 2];
        let col_indices = vec![0, 1, 2];
        let values = vec![1.0f64, 2.0, 3.0];

        let coo = CooMatrix::new(3, 3, row_indices, col_indices, values).unwrap();

        let entries: Vec<_> = coo.iter().collect();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0], (0, 0, &1.0));
        assert_eq!(entries[1], (1, 1, &2.0));
        assert_eq!(entries[2], (2, 2, &3.0));
    }

    #[test]
    fn test_coo_scale() {
        let row_indices = vec![0, 1, 2];
        let col_indices = vec![0, 1, 2];
        let values = vec![1.0f64, 2.0, 3.0];

        let mut coo = CooMatrix::new(3, 3, row_indices, col_indices, values).unwrap();
        coo.scale(2.0);

        assert_eq!(coo.values(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_coo_builder() {
        let mut builder = CooMatrixBuilder::<f64>::with_capacity(3, 3, 5);
        builder.add(0, 0, 1.0);
        builder.add(1, 1, 2.0);
        builder.add(2, 2, 3.0);
        builder.add_diagonal(0, 0.5); // Duplicate at (0,0)
        let csr = builder.build_csr();

        assert_eq!(csr.nnz(), 3);
        assert_eq!(csr.get(0, 0), Some(&1.5)); // 1.0 + 0.5
    }

    #[test]
    fn test_coo_empty() {
        let coo: CooMatrix<f64> = CooMatrix::new_empty(5, 5);
        assert!(coo.is_empty());
        assert_eq!(coo.nrows(), 5);
        assert_eq!(coo.ncols(), 5);
    }

    #[test]
    fn test_coo_clear() {
        let mut coo = CooMatrix::<f64>::new_empty(3, 3);
        coo.push(0, 0, 1.0);
        coo.push(1, 1, 2.0);

        assert_eq!(coo.len(), 2);
        coo.clear();
        assert!(coo.is_empty());
        assert_eq!(coo.nrows(), 3); // Dimensions preserved
    }

    #[test]
    fn test_coo_invalid_row() {
        let row_indices = vec![5];
        let col_indices = vec![0];
        let values = vec![1.0f64];

        let result = CooMatrix::new(3, 3, row_indices, col_indices, values);
        assert!(matches!(result, Err(CooError::InvalidRowIndex { .. })));
    }

    #[test]
    fn test_coo_invalid_col() {
        let row_indices = vec![0];
        let col_indices = vec![5];
        let values = vec![1.0f64];

        let result = CooMatrix::new(3, 3, row_indices, col_indices, values);
        assert!(matches!(result, Err(CooError::InvalidColumnIndex { .. })));
    }
}
