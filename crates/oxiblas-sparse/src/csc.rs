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
//! - `col_ptrs` has length n+1, starts at 0, never decreases and ends at nnz
//!
//! Column `j` owns the entries `col_ptrs[j]..col_ptrs[j + 1]`. Within a column
//! the row indices are distinct but need not be sorted; see
//! [`CscMatrix::new_unchecked`] for the full contract.

use crate::csr::first_duplicate_index;
use oxiblas_core::scalar::{Field, Scalar};
use std::collections::HashSet;

/// Error type for CSC matrix operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CscError {
    /// Invalid column pointer array: its length is not `ncols + 1` (then
    /// `expected` is `ncols + 1`, saturating at `usize::MAX`), its first entry
    /// is not 0 (`expected` is 0), or its last entry is not the number of
    /// stored entries (`expected` is that number).
    InvalidColPtrs {
        /// Expected length or value.
        expected: usize,
        /// Actual length or value.
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
    ///
    /// A given `(row, col)` position must be represented by at most one entry
    /// (as for [`crate::csr::CsrError::DuplicateEntry`]): [`CscMatrix::get`]
    /// returns the first match in a column, so a second entry would be
    /// silently ignored. Row indices within a column need *not* be sorted.
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
                    "Invalid col_ptrs (length must be ncols + 1, first entry 0, last entry nnz): \
                     expected {expected}, got {actual}"
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
    /// Returns an error unless the input satisfies the whole contract
    /// documented on [`CscMatrix::new_unchecked`]:
    /// - [`CscError::InvalidColPtrs`] if `col_ptrs.len() != ncols + 1`,
    ///   `col_ptrs[0] != 0` or `col_ptrs[ncols] != values.len()`
    /// - [`CscError::LengthMismatch`] if `values` and `row_indices` differ in
    ///   length
    /// - [`CscError::InvalidColPtrOrder`] if `col_ptrs` decreases somewhere
    /// - [`CscError::InvalidRowIndex`] if a row index is `>= nrows`
    /// - [`CscError::DuplicateEntry`] if a column lists a row twice
    pub fn new(
        nrows: usize,
        ncols: usize,
        col_ptrs: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self, CscError> {
        // Validate col_ptrs length (`ncols + 1` itself must not overflow).
        let expected_len = ncols.checked_add(1);
        if expected_len != Some(col_ptrs.len()) {
            return Err(CscError::InvalidColPtrs {
                expected: expected_len.unwrap_or(usize::MAX),
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

        // Validate col_ptrs[0] == 0: otherwise the entries before it belong to
        // no column (and with `ncols == 0`, `col_ptrs == [nnz]` would smuggle
        // in `nnz` such entries).
        if col_ptrs[0] != 0 {
            return Err(CscError::InvalidColPtrs {
                expected: 0,
                actual: col_ptrs[0],
            });
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

        // Validate that no column lists a row twice (order-independent, as
        // in `CsrMatrix::new`: columns need not be sorted).
        let mut seen_rows = HashSet::new();
        for col in 0..ncols {
            let start = col_ptrs[col];
            let end = col_ptrs[col + 1];

            if let Some(row) = first_duplicate_index(&row_indices[start..end], &mut seen_rows) {
                return Err(CscError::DuplicateEntry { row, col });
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
    /// The caller must ensure the CSC contract that [`CscMatrix::new`]
    /// checks. Structurally, with `nnz = values.len()`:
    /// - `col_ptrs.len() == ncols + 1`
    /// - `col_ptrs[0] == 0` and `col_ptrs[ncols] == nnz`
    /// - `col_ptrs` never decreases (`col_ptrs[j] <= col_ptrs[j + 1]`)
    /// - `row_indices.len() == nnz`
    /// - every row index is `< nrows`
    ///
    /// Logically:
    /// - no column lists the same row twice (each `(row, col)` position has
    ///   at most one entry)
    ///
    /// Row indices within a column need **not** be sorted: the methods of
    /// this type and the format conversions accept unsorted columns.
    ///
    /// Every method of `CscMatrix` and every function in this crate relies on
    /// this contract: column `j` is read as `col_ptrs[j]..col_ptrs[j + 1]` and
    /// row indices are used to index vectors of length `nrows`. Violating it
    /// can therefore cause panics or silently wrong results here, and `unsafe`
    /// code elsewhere may rely on it for memory safety.
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
    ///
    /// The result is `ncols x nrows`, with row indices sorted within each
    /// column.
    pub fn transpose(&self) -> Self {
        // The CSR arrays of A are exactly the CSC arrays of A^T: row `i` of A
        // is column `i` of A^T, and its column indices are A^T's row indices.
        let csr = self.to_csr();
        // SAFETY: `self` satisfies the CSC contract, so `csc_to_csr` returns
        // `row_ptrs` of length `nrows + 1` that start at 0, never decrease and
        // end at nnz, with every one of the nnz slots filled by exactly one
        // entry of `self` (column index `< ncols`, no `(row, col)` repeated).
        // Read as the CSC of the `ncols x nrows` transpose, these are column
        // pointers of length `new_ncols + 1` with the same bounds, row indices
        // `< new_nrows`, and no repeated row within a column.
        unsafe {
            Self::new_unchecked(
                self.ncols,
                self.nrows,
                csr.row_ptrs().to_vec(),
                csr.col_indices().to_vec(),
                csr.values().to_vec(),
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

/// Test helper: asserts that `m`'s raw parts satisfy the CSC contract, by
/// re-validating them with [`CscMatrix::new`].
#[cfg(test)]
pub(crate) fn assert_valid_csc<T: Scalar + Clone>(m: &CscMatrix<T>) {
    let rebuilt = CscMatrix::new(
        m.nrows(),
        m.ncols(),
        m.col_ptrs().to_vec(),
        m.row_indices().to_vec(),
        m.values().to_vec(),
    );
    assert!(
        rebuilt.is_ok(),
        "matrix violates the CSC contract: {:?}",
        rebuilt.err()
    );
}

// Note: `CscMatrix` deliberately does not implement `std::ops::Index`
// (matching `CsrMatrix`).
//
// A sparse matrix cannot honor `Index`'s `&Self::Output` contract for a
// structurally-absent (implicit zero) position: there is no `T` value stored at
// that location to borrow from, so the only implementations available are
// (a) panicking, which violates the crate's no-panic policy and is a surprising
// trap — `let v = a[(0, 1)];` for a merely-empty position is a perfectly
// ordinary sparse read, yet it aborted the program — or (b) manufacturing a
// reference to a shared zero, which would require `T: 'static` plus interior
// storage purely to satisfy the trait shape. Use the checked accessors instead:
//   - [`CscMatrix::get`] returns `Option<&T>` (`None` for implicit zeros and
//     out-of-bounds positions alike).
//   - [`CscMatrix::get_or_zero`] returns an owned `T`, `T::zero()` for implicit
//     zeros (requires `T: Field`).

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

    // --- Regression: `is_structurally_symmetric` must query the transposed
    // position, not re-discover the entry it started from --------------------
    //
    // `is_structurally_symmetric` walks every stored entry (row, col) and must
    // ask "does the mirrored entry (col, row) also exist?" via
    // `self.get(col, row)`. A swapped call — `self.get(row, col)` — would ask
    // "does (row, col) exist?", which is always true because that is the very
    // entry the outer loop is currently standing on: the check degenerates
    // into a no-op self-lookup that can never observe an asymmetry and always
    // reports `true`. The two cases below pin the argument order: a
    // block-diagonal matrix that is genuinely symmetric but is *not* fully
    // dense (so a naive "count the entries" check can't accidentally pass),
    // and a strictly upper-triangular matrix that is genuinely asymmetric and
    // that the swapped/self-lookup form would misreport as symmetric.
    #[test]
    fn test_csc_structurally_symmetric_block_diagonal_non_dense() {
        // [1 2 0]
        // [3 4 0]
        // [0 0 5]
        // Symmetric *structure* (values need not match, only the zero
        // pattern), but not a fully dense matrix, so every stored entry
        // genuinely requires the transpose lookup to find its mirror in a
        // different column.
        let values = vec![1.0f64, 3.0, 2.0, 4.0, 5.0];
        let row_indices = vec![0, 1, 0, 1, 2];
        let col_ptrs = vec![0, 2, 4, 5];

        let csc = CscMatrix::new(3, 3, col_ptrs, row_indices, values).unwrap();
        assert!(csc.is_structurally_symmetric());
    }

    #[test]
    fn test_csc_not_structurally_symmetric_upper_triangular() {
        // [1 2 0]
        // [0 3 0]
        // [0 0 4]
        // (0,1) is stored but its mirror (1,0) is not: not structurally
        // symmetric. A swapped `self.get(row, col)` call would re-find (0,1)
        // itself instead of checking for (1,0) and would wrongly report
        // `true`.
        let values = vec![1.0f64, 2.0, 3.0, 4.0];
        let row_indices = vec![0, 0, 1, 2];
        let col_ptrs = vec![0, 1, 3, 4];

        let csc = CscMatrix::new(3, 3, col_ptrs, row_indices, values).unwrap();
        assert!(!csc.is_structurally_symmetric());
    }

    // --- Regression: reading a structurally-zero element must not panic -----
    //
    // `CscMatrix` used to implement `Index<(usize, usize)>` as
    // `self.get(row, col).expect("Index out of bounds or zero element")`, so
    // `let v = a[(0, 1)];` for an implicit zero — an entirely ordinary sparse
    // read — aborted the program. The impl is gone (matching `CsrMatrix`);
    // `get` / `get_or_zero` are the supported accessors and never panic.
    #[test]
    fn test_csc_no_index_operator_use_get_instead() {
        // [1 0 4]
        // [0 3 0]
        // [2 0 5]
        let values = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let row_indices = vec![0, 2, 1, 0, 2];
        let col_ptrs = vec![0, 2, 3, 5];

        let csc = CscMatrix::new(3, 3, col_ptrs, row_indices, values).expect("valid CSC");

        // Structural zero: `None` / `0.0`, not a panic.
        assert_eq!(csc.get(0, 1), None);
        assert_eq!(csc.get_or_zero(0, 1), 0.0);
        // Out of bounds is the same story.
        assert_eq!(csc.get(99, 99), None);
        assert_eq!(csc.get_or_zero(99, 99), 0.0);
        // Stored entries still read back.
        assert_eq!(csc.get_or_zero(0, 0), 1.0);
        assert_eq!(csc.get_or_zero(2, 2), 5.0);
    }

    // --- Regression: `new` must validate as strictly as `CsrMatrix::new` -----
    //
    // `CscMatrix::new` accepted duplicate row indices within a column (which
    // `get` silently mis-resolves, exactly as for CSR) and never checked
    // `col_ptrs[0] == 0`, so `ncols == 0` with `col_ptrs == [nnz]` smuggled in
    // entries that belong to no column.

    #[test]
    fn test_csc_duplicate_entry_rejected() {
        // Column 0 lists row 1 twice.
        let result = CscMatrix::new(3, 1, vec![0, 3], vec![1, 1, 2], vec![1.0f64, 2.0, 3.0]);
        assert_eq!(
            result.unwrap_err(),
            CscError::DuplicateEntry { row: 1, col: 0 }
        );
    }

    #[test]
    fn test_csc_duplicate_entry_rejected_non_adjacent() {
        // Column 1 stores rows [2, 1, 2]: the two `2`s are a duplicate.
        let result = CscMatrix::new(
            3,
            2,
            vec![0, 1, 4],
            vec![0, 2, 1, 2],
            vec![1.0f64, 2.0, 3.0, 4.0],
        );
        assert_eq!(
            result.unwrap_err(),
            CscError::DuplicateEntry { row: 2, col: 1 }
        );
    }

    #[test]
    fn test_csc_unsorted_and_cross_column_repeats_accepted() {
        // Rows within a column need not be sorted, and the same row index may
        // appear in different columns.
        let csc = CscMatrix::new(
            3,
            2,
            vec![0, 2, 4],
            vec![2, 0, 0, 1],
            vec![1.0f64, 2.0, 3.0, 4.0],
        )
        .expect("unsorted but duplicate-free columns are valid");
        assert_eq!(csc.get(2, 0), Some(&1.0));
        assert_eq!(csc.get(0, 0), Some(&2.0));
        assert_eq!(csc.get(0, 1), Some(&3.0));
    }

    #[test]
    fn test_csc_zero_cols_with_entries_rejected() {
        let result = CscMatrix::new(3, 0, vec![2], vec![0, 1], vec![1.0f64, 2.0]);
        assert_eq!(
            result.unwrap_err(),
            CscError::InvalidColPtrs {
                expected: 0,
                actual: 2
            }
        );
    }

    #[test]
    fn test_csc_first_col_ptr_must_be_zero() {
        let result = CscMatrix::new(3, 2, vec![1, 2, 2], vec![0, 1], vec![1.0f64, 2.0]);
        assert_eq!(
            result.unwrap_err(),
            CscError::InvalidColPtrs {
                expected: 0,
                actual: 1
            }
        );
    }

    #[test]
    fn test_csc_col_ptrs_length_overflow_is_an_error() {
        // `ncols + 1` overflows `usize`: an error, not a panic.
        let result = CscMatrix::<f64>::new(1, usize::MAX, vec![], vec![], vec![]);
        assert!(matches!(result, Err(CscError::InvalidColPtrs { .. })));
    }

    #[test]
    fn test_csc_transpose_satisfies_contract() {
        let csc = CscMatrix::new(3, 2, vec![0, 2, 3], vec![2, 0, 1], vec![1.0f64, 2.0, 3.0])
            .expect("valid CSC");
        let t = csc.transpose();
        let rebuilt = CscMatrix::new(
            t.nrows(),
            t.ncols(),
            t.col_ptrs().to_vec(),
            t.row_indices().to_vec(),
            t.values().to_vec(),
        );
        assert!(rebuilt.is_ok());
        assert_eq!(t.shape(), (2, 3));
        assert_eq!(t.get(0, 2), Some(&1.0));
        assert_eq!(t.get(0, 0), Some(&2.0));
        assert_eq!(t.get(1, 1), Some(&3.0));
    }
}
