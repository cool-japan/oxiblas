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
//! - `row_ptrs` has length m+1, starts at 0, never decreases and ends at nnz
//!
//! Row `i` owns the entries `row_ptrs[i]..row_ptrs[i + 1]`. Within a row the
//! column indices are distinct but need not be sorted; see
//! [`CsrMatrix::new_unchecked`] for the full contract.

use oxiblas_core::scalar::{Field, Scalar};
use std::collections::HashSet;

/// Error type for CSR matrix operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CsrError {
    /// Invalid row pointer array: its length is not `nrows + 1` (then
    /// `expected` is `nrows + 1`, saturating at `usize::MAX`), its first entry
    /// is not 0 (`expected` is 0), or its last entry is not the number of
    /// stored entries (`expected` is that number).
    InvalidRowPtrs {
        /// Expected length or value.
        expected: usize,
        /// Actual length or value.
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
    ///
    /// A given `(row, col)` position must be represented by at most one
    /// entry: with two entries at the same position, [`CsrMatrix::get`]
    /// can only ever surface the first (since it returns on the first
    /// column match within the row), silently discarding the second
    /// value and any updates encoded in it. That is a structurally
    /// corrupt matrix, not merely an inefficient one, so it is rejected
    /// at construction time rather than accepted and silently
    /// mis-resolved.
    ///
    /// Note: column indices within a row are *not* required to be sorted
    /// ascending — every accessor in this type (`get`, `row_iter`, `iter`)
    /// performs a linear scan over the row and is agnostic to order, so a
    /// permuted (but duplicate-free) row is a perfectly valid CSR
    /// representation here.
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
                    "Invalid row_ptrs (length must be nrows + 1, first entry 0, last entry nnz): \
                     expected {expected}, got {actual}"
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

/// Returns the first index in `indices` (in storage order) that repeats an
/// earlier one, or `None` if all are distinct.
///
/// Order-independent: strictly increasing runs (the common, sorted case) are
/// accepted without hashing; anything else is checked with `seen`, which is
/// cleared first and only used as scratch space.
pub(crate) fn first_duplicate_index(indices: &[usize], seen: &mut HashSet<usize>) -> Option<usize> {
    if indices.windows(2).all(|pair| pair[0] < pair[1]) {
        return None;
    }
    seen.clear();
    indices.iter().copied().find(|&index| !seen.insert(index))
}

/// Appends one compressed row (or column) built from `entries`, which are
/// `(index, value)` pairs in any order, to `out_indices` / `out_values`.
///
/// The entries are sorted by index (stably) and entries that share an index
/// are combined into one entry holding their sum, so the appended run is
/// strictly ascending and duplicate-free. `entries` is left empty.
///
/// The format conversions use this to uphold the no-duplicate part of the
/// CSR/CSC contract even when the source format stores one position twice
/// (which the source format's own `matvec` treats as a sum).
pub(crate) fn push_summed_run<T: Scalar>(
    entries: &mut Vec<(usize, T)>,
    out_indices: &mut Vec<usize>,
    out_values: &mut Vec<T>,
) {
    entries.sort_by_key(|&(index, _)| index);
    let run_start = out_indices.len();
    for (index, value) in entries.drain(..) {
        let repeats_last = out_indices.len() > run_start && out_indices.last() == Some(&index);
        match out_values.last_mut() {
            Some(last) if repeats_last => *last = *last + value,
            _ => {
                out_indices.push(index);
                out_values.push(value);
            }
        }
    }
}

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
    /// Returns an error unless the input satisfies the whole contract
    /// documented on [`CsrMatrix::new_unchecked`]:
    /// - [`CsrError::InvalidRowPtrs`] if `row_ptrs.len() != nrows + 1`,
    ///   `row_ptrs[0] != 0` or `row_ptrs[nrows] != values.len()`
    /// - [`CsrError::LengthMismatch`] if `values` and `col_indices` differ in
    ///   length
    /// - [`CsrError::InvalidRowPtrOrder`] if `row_ptrs` decreases somewhere
    /// - [`CsrError::InvalidColumnIndex`] if a column index is `>= ncols`
    /// - [`CsrError::DuplicateEntry`] if a row lists a column twice
    pub fn new(
        nrows: usize,
        ncols: usize,
        row_ptrs: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self, CsrError> {
        // Validate row_ptrs length (`nrows + 1` itself must not overflow).
        let expected_len = nrows.checked_add(1);
        if expected_len != Some(row_ptrs.len()) {
            return Err(CsrError::InvalidRowPtrs {
                expected: expected_len.unwrap_or(usize::MAX),
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

        // Validate row_ptrs[0] == 0: otherwise the entries before it belong to
        // no row (and with `nrows == 0`, `row_ptrs == [nnz]` would smuggle in
        // `nnz` such entries).
        if row_ptrs[0] != 0 {
            return Err(CsrError::InvalidRowPtrs {
                expected: 0,
                actual: row_ptrs[0],
            });
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

        // Validate that no row contains a duplicate column index. Order
        // within a row is deliberately *not* constrained here — `get`,
        // `row_iter`, and `iter` all linear-scan a row and are agnostic to
        // permutation, so only an actual repeated `(row, col)` position is
        // structurally invalid (it makes the stored value at that
        // position ambiguous, see `CsrError::DuplicateEntry`).
        //
        // Detection is order-independent (see `first_duplicate_index`) rather
        // than "check adjacent pairs", precisely because rows are not
        // required to be pre-sorted.
        let mut seen_cols = HashSet::new();
        for row in 0..nrows {
            let start = row_ptrs[row];
            let end = row_ptrs[row + 1];

            if let Some(col) = first_duplicate_index(&col_indices[start..end], &mut seen_cols) {
                return Err(CsrError::DuplicateEntry { row, col });
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
    /// The caller must ensure the CSR contract that [`CsrMatrix::new`]
    /// checks. Structurally, with `nnz = values.len()`:
    /// - `row_ptrs.len() == nrows + 1`
    /// - `row_ptrs[0] == 0` and `row_ptrs[nrows] == nnz`
    /// - `row_ptrs` never decreases (`row_ptrs[i] <= row_ptrs[i + 1]`)
    /// - `col_indices.len() == nnz`
    /// - every column index is `< ncols`
    ///
    /// Logically:
    /// - no row lists the same column twice (each `(row, col)` position has
    ///   at most one entry)
    ///
    /// Column indices within a row need **not** be sorted: the methods of this
    /// type, the format conversions and the [`crate::ops`] functions all
    /// accept unsorted rows (the latter return sorted rows where they build a
    /// new pattern).
    ///
    /// Every method of `CsrMatrix` and every function in this crate relies on
    /// this contract: row `i` is read as `row_ptrs[i]..row_ptrs[i + 1]` and
    /// column indices are used to index vectors of length `ncols`. Violating
    /// it can therefore cause panics or silently wrong results here, and
    /// `unsafe` code elsewhere may rely on it for memory safety.
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

        // Reinterpret CSC of A as CSR of A^T.
        // SAFETY: `self` satisfies the CSR contract, so `csr_to_csc` returns
        // `col_ptrs` of length `ncols + 1` that start at 0, never decrease and
        // end at nnz, with every one of the nnz slots filled by exactly one
        // entry of `self` (row index `< nrows`, no `(row, col)` repeated). Read
        // as the CSR of the `ncols x nrows` transpose, these are row pointers
        // of length `new_nrows + 1` with the same bounds, column indices
        // `< new_ncols`, and no repeated column within a row.
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

/// Test helper: asserts that `m`'s raw parts satisfy the CSR contract, by
/// re-validating them with [`CsrMatrix::new`].
#[cfg(test)]
pub(crate) fn assert_valid_csr<T: Scalar + Clone>(m: &CsrMatrix<T>) {
    let rebuilt = CsrMatrix::new(
        m.nrows(),
        m.ncols(),
        m.row_ptrs().to_vec(),
        m.col_indices().to_vec(),
        m.values().to_vec(),
    );
    assert!(
        rebuilt.is_ok(),
        "matrix violates the CSR contract: {:?}",
        rebuilt.err()
    );
}

// Note: `CsrMatrix` deliberately does not implement `std::ops::Index`.
//
// A sparse matrix cannot honor `Index`'s `&Self::Output` contract for a
// structurally-absent (implicit zero) position: there is no `T` value
// stored at that location to borrow from, so the only implementations
// available are (a) panicking, which violates the crate's no-panic policy
// and is a surprising trap for out-of-bounds *or* merely-empty positions
// alike, or (b) manufacturing a reference to a shared zero, which would
// require `T: 'static` plus interior storage (e.g. a thread-local or
// `OnceLock` per instantiation of `T`) purely to satisfy the trait shape
// and is unnecessary complexity for what the crate does not actually
// need. Use the checked accessors instead:
//   - [`CsrMatrix::get`] returns `Option<&T>` (`None` for implicit zeros
//     and out-of-bounds positions alike).
//   - [`CsrMatrix::get_or_zero`] returns an owned `T`, `T::zero()` for
//     implicit zeros (requires `T: Field`).

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

    #[test]
    fn test_csr_duplicate_entry_rejected() {
        // Row 0 has column 1 listed twice: (0,1)=1.0 and (0,1)=2.0.
        let values = vec![1.0f64, 2.0, 3.0];
        let col_indices = vec![1, 1, 2];
        let row_ptrs = vec![0, 3];

        let result = CsrMatrix::new(1, 3, row_ptrs, col_indices, values);
        assert_eq!(
            result.unwrap_err(),
            CsrError::DuplicateEntry { row: 0, col: 1 }
        );
    }

    #[test]
    fn test_csr_duplicate_entry_rejected_non_adjacent() {
        // Duplicate columns need not be adjacent in storage order to be
        // caught: row 0 stores columns [2, 1, 1] - the two `1`s are a
        // structural duplicate regardless of the intervening `2`.
        let values = vec![1.0f64, 2.0, 3.0];
        let col_indices = vec![2, 1, 1];
        let row_ptrs = vec![0, 3];

        let result = CsrMatrix::new(1, 3, row_ptrs, col_indices, values);
        assert_eq!(
            result.unwrap_err(),
            CsrError::DuplicateEntry { row: 0, col: 1 }
        );
    }

    #[test]
    fn test_csr_unsorted_but_duplicate_free_indices_accepted() {
        // Row 0 columns are [2, 0] - out of ascending order but unique.
        // Ordering within a row is not part of this crate's CSR contract
        // (every accessor linear-scans the row), so this must succeed.
        let values = vec![1.0f64, 2.0];
        let col_indices = vec![2, 0];
        let row_ptrs = vec![0, 2];

        let csr = CsrMatrix::new(1, 3, row_ptrs, col_indices, values).unwrap();
        assert_eq!(csr.get(0, 2), Some(&1.0));
        assert_eq!(csr.get(0, 0), Some(&2.0));
    }

    #[test]
    fn test_csr_sorted_unique_indices_accepted() {
        // Ascending, unique per row: must construct successfully.
        let values = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let col_indices = vec![0, 2, 1, 0, 2];
        let row_ptrs = vec![0, 2, 3, 5];

        let result = CsrMatrix::new(3, 3, row_ptrs, col_indices, values);
        assert!(result.is_ok());
    }

    #[test]
    fn test_csr_empty_and_singleton_rows_accepted() {
        // Empty rows and single-entry rows must never trip the duplicate
        // check (no pair exists to compare within the row).
        let values = vec![1.0f64];
        let col_indices = vec![0];
        let row_ptrs = vec![0, 0, 1, 1];

        let result = CsrMatrix::new(3, 2, row_ptrs, col_indices, values);
        assert!(result.is_ok());
    }

    #[test]
    fn test_csr_duplicate_across_different_rows_accepted() {
        // The same column index appearing in *different* rows is not a
        // duplicate - only a repeat within a single row is invalid.
        let values = vec![1.0f64, 2.0];
        let col_indices = vec![0, 0];
        let row_ptrs = vec![0, 1, 2];

        let result = CsrMatrix::new(2, 2, row_ptrs, col_indices, values);
        assert!(result.is_ok());
    }

    #[test]
    fn test_csr_no_index_operator_use_get_instead() {
        // `CsrMatrix` intentionally has no `Index` impl (see module note);
        // `get` and `get_or_zero` are the supported accessors and must
        // never panic for implicit-zero (structurally-absent) positions.
        let values = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let col_indices = vec![0, 2, 1, 0, 2];
        let row_ptrs = vec![0, 2, 3, 5];

        let csr = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();

        assert_eq!(csr.get(0, 1), None);
        assert_eq!(csr.get_or_zero(0, 1), 0.0);
        assert_eq!(csr.get_or_zero(0, 0), 1.0);
    }

    // --- Regression: `new` must enforce the whole row-pointer contract -------
    //
    // `new` never checked `row_ptrs[0] == 0`. With `nrows == 0` it therefore
    // accepted `row_ptrs == [nnz]` with `nnz > 0` stored entries that belong to
    // no row, and `transpose_csr` / `csr_to_csc` turned such a matrix into one
    // with an out-of-range index via `new_unchecked`.

    #[test]
    fn test_csr_zero_rows_with_entries_rejected() {
        let result = CsrMatrix::new(0, 3, vec![2], vec![0, 1], vec![1.0f64, 2.0]);
        assert_eq!(
            result.unwrap_err(),
            CsrError::InvalidRowPtrs {
                expected: 0,
                actual: 2
            }
        );
    }

    #[test]
    fn test_csr_first_row_ptr_must_be_zero() {
        // Entry 0 would belong to no row.
        let result = CsrMatrix::new(2, 3, vec![1, 2, 2], vec![0, 1], vec![1.0f64, 2.0]);
        assert_eq!(
            result.unwrap_err(),
            CsrError::InvalidRowPtrs {
                expected: 0,
                actual: 1
            }
        );
    }

    #[test]
    fn test_csr_row_ptrs_length_overflow_is_an_error() {
        // `nrows + 1` overflows `usize`: an error, not a panic.
        let result = CsrMatrix::<f64>::new(usize::MAX, 1, vec![], vec![], vec![]);
        assert!(matches!(result, Err(CsrError::InvalidRowPtrs { .. })));
    }

    #[test]
    fn test_csr_empty_shapes_accepted() {
        assert!(CsrMatrix::<f64>::new(0, 0, vec![0], vec![], vec![]).is_ok());
        assert!(CsrMatrix::<f64>::new(0, 5, vec![0], vec![], vec![]).is_ok());
        assert!(CsrMatrix::<f64>::new(3, 0, vec![0, 0, 0, 0], vec![], vec![]).is_ok());
    }

    /// Re-validates a matrix's raw parts with the checking constructor.
    fn revalidate(csr: &CsrMatrix<f64>) -> Result<CsrMatrix<f64>, CsrError> {
        CsrMatrix::new(
            csr.nrows(),
            csr.ncols(),
            csr.row_ptrs().to_vec(),
            csr.col_indices().to_vec(),
            csr.values().to_vec(),
        )
    }

    #[test]
    fn test_csr_transpose_and_round_trip_satisfy_contract() {
        // [2 0 1]
        // [0 3 0], row 0 stored unsorted as columns [2, 0].
        let csr = CsrMatrix::new(2, 3, vec![0, 2, 3], vec![2, 0, 1], vec![1.0f64, 2.0, 3.0])
            .expect("valid CSR");
        let t = csr.transpose();
        assert!(revalidate(&t).is_ok());
        assert_eq!(t.shape(), (3, 2));
        assert_eq!(t.get(0, 0), Some(&2.0));
        assert_eq!(t.get(2, 0), Some(&1.0));
        assert_eq!(t.get(1, 1), Some(&3.0));

        let round_trip = csr.to_csc().to_csr();
        assert!(revalidate(&round_trip).is_ok());
        assert_eq!(round_trip.get(0, 2), Some(&1.0));
        assert_eq!(round_trip.get(0, 0), Some(&2.0));
    }
}
