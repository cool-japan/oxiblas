//! Diagonal (DIA) sparse matrix format.
//!
//! DIA stores matrix data using:
//! - `offsets`: Array of diagonal offsets (0 = main diagonal, positive = super-diagonals, negative = sub-diagonals)
//! - `data`: 2D array where `data[k]` contains the k-th diagonal
//!
//! For an m×n matrix with `ndiag` diagonals:
//! - `offsets` has length `ndiag`
//! - `data[k]` has length `min(m, n)` (padded with zeros where diagonal is shorter)
//!
//! # When to Use DIA
//!
//! DIA format is optimal for:
//! - Banded matrices (tridiagonal, pentadiagonal, etc.)
//! - Matrices where non-zeros are concentrated on a few diagonals
//! - Finite difference discretizations of differential equations
//!
//! DIA is NOT efficient for:
//! - General sparse matrices with scattered non-zeros
//! - Matrices with many diagonals that are mostly empty

use oxiblas_core::scalar::{Field, Scalar};

/// Error type for DIA matrix operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiaError {
    /// Diagonal data has wrong length.
    InvalidDiagonalLength {
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
        /// Diagonal offset.
        offset: isize,
    },
    /// Mismatched number of offsets and data arrays.
    OffsetDataMismatch {
        /// Number of offsets.
        num_offsets: usize,
        /// Number of data arrays.
        num_data: usize,
    },
    /// Duplicate diagonal offset.
    DuplicateOffset {
        /// The duplicate offset.
        offset: isize,
    },
    /// Diagonal offset out of bounds.
    OffsetOutOfBounds {
        /// The invalid offset.
        offset: isize,
        /// Number of rows.
        nrows: usize,
        /// Number of columns.
        ncols: usize,
    },
}

impl core::fmt::Display for DiaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidDiagonalLength {
                expected,
                actual,
                offset,
            } => {
                write!(
                    f,
                    "Diagonal {offset}: expected length {expected}, got {actual}"
                )
            }
            Self::OffsetDataMismatch {
                num_offsets,
                num_data,
            } => {
                write!(
                    f,
                    "Mismatch: {num_offsets} offsets but {num_data} data arrays"
                )
            }
            Self::DuplicateOffset { offset } => {
                write!(f, "Duplicate diagonal offset: {offset}")
            }
            Self::OffsetOutOfBounds {
                offset,
                nrows,
                ncols,
            } => {
                write!(
                    f,
                    "Diagonal offset {offset} out of bounds for {nrows}×{ncols} matrix"
                )
            }
        }
    }
}

impl std::error::Error for DiaError {}

/// Diagonal sparse matrix format.
///
/// Efficient for:
/// - Banded matrices
/// - Finite difference stencils
/// - Matrix-vector products on structured matrices
///
/// # Storage
///
/// Each diagonal is stored as a dense vector. The main diagonal has offset 0,
/// super-diagonals have positive offsets, and sub-diagonals have negative offsets.
///
/// # Example
///
/// ```
/// use oxiblas_sparse::DiaMatrix;
///
/// // Create a tridiagonal matrix:
/// // [4 1 0]
/// // [2 5 1]
/// // [0 3 6]
/// let offsets = vec![-1, 0, 1];  // sub, main, super
/// let data = vec![
///     vec![2.0, 3.0, 0.0],       // sub-diagonal (padded)
///     vec![4.0, 5.0, 6.0],       // main diagonal
///     vec![0.0, 1.0, 1.0],       // super-diagonal (padded)
/// ];
///
/// let dia = DiaMatrix::new(3, 3, offsets, data).unwrap();
/// assert_eq!(dia.ndiag(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct DiaMatrix<T: Scalar> {
    /// Number of rows.
    nrows: usize,
    /// Number of columns.
    ncols: usize,
    /// Diagonal offsets (sorted).
    offsets: Vec<isize>,
    /// Diagonal data (each inner Vec corresponds to one diagonal).
    data: Vec<Vec<T>>,
}

impl<T: Scalar + Clone> DiaMatrix<T> {
    /// Creates a new DIA matrix from raw components.
    ///
    /// # Arguments
    ///
    /// * `nrows` - Number of rows
    /// * `ncols` - Number of columns
    /// * `offsets` - Diagonal offsets (0 = main, positive = super, negative = sub)
    /// * `data` - Diagonal data, one vector per diagonal
    ///
    /// # Errors
    ///
    /// Returns an error if the input is invalid.
    pub fn new(
        nrows: usize,
        ncols: usize,
        offsets: Vec<isize>,
        data: Vec<Vec<T>>,
    ) -> Result<Self, DiaError> {
        // Validate offsets and data count match
        if offsets.len() != data.len() {
            return Err(DiaError::OffsetDataMismatch {
                num_offsets: offsets.len(),
                num_data: data.len(),
            });
        }

        // Check for duplicates and bounds
        let mut seen_offsets = std::collections::HashSet::new();
        for &offset in &offsets {
            if !seen_offsets.insert(offset) {
                return Err(DiaError::DuplicateOffset { offset });
            }

            // Check if offset is valid for matrix dimensions
            if offset >= 0 {
                if offset as usize >= ncols {
                    return Err(DiaError::OffsetOutOfBounds {
                        offset,
                        nrows,
                        ncols,
                    });
                }
            } else if (-offset) as usize >= nrows {
                return Err(DiaError::OffsetOutOfBounds {
                    offset,
                    nrows,
                    ncols,
                });
            }
        }

        // Validate diagonal lengths
        let diag_len = nrows.min(ncols);
        for (k, diag) in data.iter().enumerate() {
            if diag.len() != diag_len {
                return Err(DiaError::InvalidDiagonalLength {
                    expected: diag_len,
                    actual: diag.len(),
                    offset: offsets[k],
                });
            }
        }

        // Sort offsets and data together
        let mut pairs: Vec<_> = offsets.into_iter().zip(data).collect();
        pairs.sort_by_key(|(offset, _)| *offset);
        let (offsets, data): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();

        Ok(Self {
            nrows,
            ncols,
            offsets,
            data,
        })
    }

    /// Creates a DIA matrix without validation (unsafe but faster).
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `offsets.len() == data.len()`
    /// - All offsets are within bounds
    /// - All diagonal vectors have correct length
    /// - Offsets are sorted and unique
    #[inline]
    pub unsafe fn new_unchecked(
        nrows: usize,
        ncols: usize,
        offsets: Vec<isize>,
        data: Vec<Vec<T>>,
    ) -> Self {
        Self {
            nrows,
            ncols,
            offsets,
            data,
        }
    }

    /// Creates an empty DIA matrix (all zeros) with given dimensions.
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            offsets: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Creates an identity matrix in DIA format.
    pub fn eye(n: usize) -> Self
    where
        T: Field,
    {
        Self {
            nrows: n,
            ncols: n,
            offsets: vec![0],
            data: vec![vec![T::one(); n]],
        }
    }

    /// Creates a tridiagonal matrix.
    ///
    /// # Arguments
    ///
    /// * `sub` - Sub-diagonal (length n-1)
    /// * `main` - Main diagonal (length n)
    /// * `super_diag` - Super-diagonal (length n-1)
    pub fn tridiagonal(sub: Vec<T>, main: Vec<T>, super_diag: Vec<T>) -> Result<Self, DiaError>
    where
        T: Field,
    {
        let n = main.len();
        if n == 0 {
            return Ok(Self::zeros(0, 0));
        }

        if sub.len() != n - 1 || super_diag.len() != n - 1 {
            return Err(DiaError::InvalidDiagonalLength {
                expected: n - 1,
                actual: if sub.len() != n - 1 {
                    sub.len()
                } else {
                    super_diag.len()
                },
                offset: if sub.len() != n - 1 { -1 } else { 1 },
            });
        }

        // Pad diagonals to length n
        let mut sub_padded = sub;
        sub_padded.push(T::zero());

        let mut super_padded = vec![T::zero()];
        super_padded.extend(super_diag);

        Self::new(n, n, vec![-1, 0, 1], vec![sub_padded, main, super_padded])
    }

    /// Creates a pentadiagonal matrix (5 diagonals).
    ///
    /// # Arguments
    ///
    /// * `sub2` - Second sub-diagonal (offset -2)
    /// * `sub1` - First sub-diagonal (offset -1)
    /// * `main` - Main diagonal (offset 0)
    /// * `super1` - First super-diagonal (offset 1)
    /// * `super2` - Second super-diagonal (offset 2)
    #[allow(clippy::too_many_arguments)]
    pub fn pentadiagonal(
        sub2: Vec<T>,
        sub1: Vec<T>,
        main: Vec<T>,
        super1: Vec<T>,
        super2: Vec<T>,
    ) -> Result<Self, DiaError>
    where
        T: Field,
    {
        let n = main.len();
        if n < 3 {
            return Err(DiaError::OffsetOutOfBounds {
                offset: 2,
                nrows: n,
                ncols: n,
            });
        }

        // Validate lengths
        if sub2.len() != n - 2
            || sub1.len() != n - 1
            || super1.len() != n - 1
            || super2.len() != n - 2
        {
            return Err(DiaError::InvalidDiagonalLength {
                expected: n,
                actual: 0,
                offset: 0,
            });
        }

        // Pad all diagonals to length n
        let mut sub2_padded = sub2;
        sub2_padded.push(T::zero());
        sub2_padded.push(T::zero());

        let mut sub1_padded = sub1;
        sub1_padded.push(T::zero());

        let mut super1_padded = vec![T::zero()];
        super1_padded.extend(super1);

        let mut super2_padded = vec![T::zero(), T::zero()];
        super2_padded.extend(super2);

        Self::new(
            n,
            n,
            vec![-2, -1, 0, 1, 2],
            vec![sub2_padded, sub1_padded, main, super1_padded, super2_padded],
        )
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

    /// Returns the number of diagonals stored.
    #[inline]
    pub fn ndiag(&self) -> usize {
        self.offsets.len()
    }

    /// Returns the number of non-zero elements.
    ///
    /// Note: This counts actual non-zeros, not just stored values.
    pub fn nnz(&self) -> usize
    where
        T: Field,
    {
        let eps = <T as Scalar>::epsilon();
        let mut count = 0;

        for (k, offset) in self.offsets.iter().enumerate() {
            let (start_row, start_col) = Self::diag_start(*offset, self.nrows, self.ncols);
            let diag_length = Self::diag_length(*offset, self.nrows, self.ncols);

            for i in 0..diag_length {
                let row = start_row + i;
                let col = start_col + i;
                if row < self.nrows && col < self.ncols {
                    let idx = Self::data_index(*offset, row, self.nrows);
                    if Scalar::abs(self.data[k][idx].clone()) > eps {
                        count += 1;
                    }
                }
            }
        }

        count
    }

    /// Returns the number of stored values (including padding zeros).
    #[inline]
    pub fn nstored(&self) -> usize {
        self.data.iter().map(Vec::len).sum()
    }

    /// Returns the diagonal offsets.
    #[inline]
    pub fn offsets(&self) -> &[isize] {
        &self.offsets
    }

    /// Returns the diagonal data.
    #[inline]
    pub fn data(&self) -> &[Vec<T>] {
        &self.data
    }

    /// Returns mutable reference to the diagonal data.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [Vec<T>] {
        &mut self.data
    }

    /// Compute the starting position (row, col) for a diagonal.
    #[inline]
    fn diag_start(offset: isize, _nrows: usize, _ncols: usize) -> (usize, usize) {
        if offset >= 0 {
            (0, offset as usize)
        } else {
            ((-offset) as usize, 0)
        }
    }

    /// Compute the length of a diagonal.
    #[inline]
    fn diag_length(offset: isize, nrows: usize, ncols: usize) -> usize {
        if offset >= 0 {
            nrows.min(ncols.saturating_sub(offset as usize))
        } else {
            ncols.min(nrows.saturating_sub((-offset) as usize))
        }
    }

    /// Compute the index into data array for a given row.
    ///
    /// For diagonals with offset k:
    /// - Super-diagonals (k > 0): data has k padding elements at start, so index = row + k
    /// - Main diagonal (k = 0): index = row
    /// - Sub-diagonals (k < 0): data has |k| padding elements at end, so index = row + k = row - |k|
    #[inline]
    fn data_index(offset: isize, row: usize, _nrows: usize) -> usize {
        // For all cases: index = row + offset (as signed arithmetic)
        // This accounts for padding at start (super) or end (sub)
        (row as isize + offset) as usize
    }

    /// Gets the value at (row, col), returning None if the position is not on a stored diagonal.
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row >= self.nrows || col >= self.ncols {
            return None;
        }

        let offset = col as isize - row as isize;

        // Binary search for the offset
        match self.offsets.binary_search(&offset) {
            Ok(k) => {
                let idx = Self::data_index(offset, row, self.nrows);
                Some(&self.data[k][idx])
            }
            Err(_) => None,
        }
    }

    /// Gets the value at (row, col), returning zero if not on a stored diagonal.
    pub fn get_or_zero(&self, row: usize, col: usize) -> T
    where
        T: Field,
    {
        self.get(row, col).cloned().unwrap_or_else(T::zero)
    }

    /// Returns an iterator over all non-zeros as (row, col, value).
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, &T)> + '_ {
        self.offsets
            .iter()
            .enumerate()
            .flat_map(move |(k, &offset)| {
                let (start_row, start_col) = Self::diag_start(offset, self.nrows, self.ncols);
                let diag_length = Self::diag_length(offset, self.nrows, self.ncols);

                (0..diag_length).filter_map(move |i| {
                    let row = start_row + i;
                    let col = start_col + i;
                    if row < self.nrows && col < self.ncols {
                        let idx = Self::data_index(offset, row, self.nrows);
                        Some((row, col, &self.data[k][idx]))
                    } else {
                        None
                    }
                })
            })
    }

    /// Matrix-vector product: y = A * x.
    pub fn matvec(&self, x: &[T], y: &mut [T])
    where
        T: Field,
    {
        assert_eq!(x.len(), self.ncols, "x length must equal ncols");
        assert_eq!(y.len(), self.nrows, "y length must equal nrows");

        // Initialize y to zero
        for yi in y.iter_mut() {
            *yi = T::zero();
        }

        // Process each diagonal
        for (k, &offset) in self.offsets.iter().enumerate() {
            let (start_row, start_col) = Self::diag_start(offset, self.nrows, self.ncols);
            let diag_length = Self::diag_length(offset, self.nrows, self.ncols);

            for i in 0..diag_length {
                let row = start_row + i;
                let col = start_col + i;
                if row < self.nrows && col < self.ncols {
                    let idx = Self::data_index(offset, row, self.nrows);
                    y[row] = y[row].clone() + self.data[k][idx].clone() * x[col].clone();
                }
            }
        }
    }

    /// Matrix-vector product returning a new vector: y = A * x.
    pub fn mul_vec(&self, x: &[T]) -> Vec<T>
    where
        T: Field,
    {
        let mut y = vec![T::zero(); self.nrows];
        self.matvec(x, &mut y);
        y
    }

    /// Converts to CSR format.
    pub fn to_csr(&self) -> crate::csr::CsrMatrix<T>
    where
        T: Field,
    {
        let eps = <T as Scalar>::epsilon();

        let mut row_ptrs = vec![0usize; self.nrows + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for row in 0..self.nrows {
            let mut row_entries: Vec<(usize, T)> = Vec::new();

            for (k, &offset) in self.offsets.iter().enumerate() {
                let col = (row as isize + offset) as usize;
                if col < self.ncols {
                    let idx = Self::data_index(offset, row, self.nrows);
                    if idx < self.data[k].len() {
                        let val = self.data[k][idx].clone();
                        if Scalar::abs(val.clone()) > eps {
                            row_entries.push((col, val));
                        }
                    }
                }
            }

            // Sort by column index
            row_entries.sort_by_key(|(col, _)| *col);

            for (col, val) in row_entries {
                col_indices.push(col);
                values.push(val);
            }
            row_ptrs[row + 1] = values.len();
        }

        // Safety: we constructed valid CSR data
        unsafe {
            crate::csr::CsrMatrix::new_unchecked(
                self.nrows,
                self.ncols,
                row_ptrs,
                col_indices,
                values,
            )
        }
    }

    /// Converts to dense matrix.
    pub fn to_dense(&self) -> oxiblas_matrix::Mat<T>
    where
        T: Field + bytemuck::Zeroable,
    {
        let mut dense = oxiblas_matrix::Mat::zeros(self.nrows, self.ncols);

        for (k, &offset) in self.offsets.iter().enumerate() {
            let (start_row, start_col) = Self::diag_start(offset, self.nrows, self.ncols);
            let diag_length = Self::diag_length(offset, self.nrows, self.ncols);

            for i in 0..diag_length {
                let row = start_row + i;
                let col = start_col + i;
                if row < self.nrows && col < self.ncols {
                    let idx = Self::data_index(offset, row, self.nrows);
                    dense[(row, col)] = self.data[k][idx].clone();
                }
            }
        }

        dense
    }

    /// Creates a DIA matrix from a dense matrix, extracting specified diagonals.
    ///
    /// # Arguments
    ///
    /// * `dense` - Source dense matrix
    /// * `offsets` - Diagonal offsets to extract (if None, extracts all non-empty diagonals)
    pub fn from_dense(dense: &oxiblas_matrix::MatRef<'_, T>, offsets: Option<Vec<isize>>) -> Self
    where
        T: Field,
    {
        let (nrows, ncols) = dense.shape();

        let offsets = offsets.unwrap_or_else(|| {
            // Find all non-empty diagonals
            let eps = <T as Scalar>::epsilon();
            let mut non_empty = std::collections::HashSet::new();

            for i in 0..nrows {
                for j in 0..ncols {
                    if Scalar::abs(dense[(i, j)].clone()) > eps {
                        non_empty.insert(j as isize - i as isize);
                    }
                }
            }

            let mut offsets: Vec<_> = non_empty.into_iter().collect();
            offsets.sort();
            offsets
        });

        let diag_len = nrows.min(ncols);
        let mut data = Vec::with_capacity(offsets.len());

        for &offset in &offsets {
            let mut diag = vec![T::zero(); diag_len];
            let (start_row, start_col) = Self::diag_start(offset, nrows, ncols);
            let diag_length = Self::diag_length(offset, nrows, ncols);

            for i in 0..diag_length {
                let row = start_row + i;
                let col = start_col + i;
                if row < nrows && col < ncols {
                    let idx = Self::data_index(offset, row, nrows);
                    diag[idx] = dense[(row, col)].clone();
                }
            }

            data.push(diag);
        }

        // Safety: we constructed valid data
        unsafe { Self::new_unchecked(nrows, ncols, offsets, data) }
    }

    /// Scales all values by a scalar.
    pub fn scale(&mut self, alpha: T) {
        for diag in &mut self.data {
            for val in diag.iter_mut() {
                *val = val.clone() * alpha.clone();
            }
        }
    }

    /// Returns a scaled copy of this matrix.
    pub fn scaled(&self, alpha: T) -> Self {
        let mut result = self.clone();
        result.scale(alpha);
        result
    }

    /// Returns the transpose of this matrix.
    pub fn transpose(&self) -> Self
    where
        T: Field,
    {
        // For each diagonal with offset k, the transposed matrix has:
        // - Offset -k
        // - Element A[i, i+k] becomes A^T[i+k, i]
        //
        // The indexing transformation is:
        // - Original: data[k_idx][row + k] for A[row, row+k]
        // - Transpose: data[new_k_idx][col + (-k)] for A^T[col, col-k]
        //   where col = row + k, so index = col - k = row

        let new_nrows = self.ncols;
        let new_ncols = self.nrows;
        let diag_len = new_nrows.min(new_ncols);

        // Negate and re-sort offsets
        let mut offset_pairs: Vec<_> = self
            .offsets
            .iter()
            .enumerate()
            .map(|(k, &offset)| (-offset, k))
            .collect();
        offset_pairs.sort_by_key(|(offset, _)| *offset);

        let new_offsets: Vec<isize> = offset_pairs.iter().map(|(o, _)| *o).collect();
        let mut new_data = Vec::with_capacity(new_offsets.len());

        for (new_offset, old_k) in &offset_pairs {
            let old_offset = -new_offset;
            let mut new_diag = vec![T::zero(); diag_len];

            let old_diag_length = Self::diag_length(old_offset, self.nrows, self.ncols);

            for i in 0..old_diag_length {
                // In original: element A[row, col] where col = row + old_offset
                let (start_row, _start_col) = Self::diag_start(old_offset, self.nrows, self.ncols);
                let row = start_row + i;
                let col = (row as isize + old_offset) as usize;

                if row < self.nrows && col < self.ncols {
                    let old_idx = Self::data_index(old_offset, row, self.nrows);
                    let val = self.data[*old_k][old_idx].clone();

                    // In transpose: this becomes A^T[col, row]
                    // new_offset = row - col = -old_offset
                    // new_row = col, new_col = row
                    let new_idx = Self::data_index(*new_offset, col, new_nrows);
                    if new_idx < diag_len {
                        new_diag[new_idx] = val;
                    }
                }
            }

            new_data.push(new_diag);
        }

        // Safety: we constructed valid DIA data
        unsafe { Self::new_unchecked(new_nrows, new_ncols, new_offsets, new_data) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dia_new() {
        // Tridiagonal matrix:
        // [4 1 0]
        // [2 5 1]
        // [0 3 6]
        let offsets = vec![-1, 0, 1];
        let data = vec![
            vec![2.0, 3.0, 0.0],
            vec![4.0, 5.0, 6.0],
            vec![0.0, 1.0, 1.0],
        ];

        let dia = DiaMatrix::new(3, 3, offsets, data).unwrap();

        assert_eq!(dia.nrows(), 3);
        assert_eq!(dia.ncols(), 3);
        assert_eq!(dia.ndiag(), 3);
    }

    #[test]
    fn test_dia_get() {
        let offsets = vec![-1, 0, 1];
        let data = vec![
            vec![2.0, 3.0, 0.0],
            vec![4.0, 5.0, 6.0],
            vec![0.0, 1.0, 1.0],
        ];

        let dia = DiaMatrix::new(3, 3, offsets, data).unwrap();

        assert_eq!(dia.get(0, 0), Some(&4.0));
        assert_eq!(dia.get(0, 1), Some(&1.0));
        assert_eq!(dia.get(1, 0), Some(&2.0));
        assert_eq!(dia.get(1, 1), Some(&5.0));
        assert_eq!(dia.get(1, 2), Some(&1.0));
        assert_eq!(dia.get(2, 1), Some(&3.0));
        assert_eq!(dia.get(2, 2), Some(&6.0));

        // Not on stored diagonal
        assert_eq!(dia.get(0, 2), None);
        assert_eq!(dia.get(2, 0), None);
    }

    #[test]
    fn test_dia_matvec() {
        // [4 1 0]   [1]   [5]
        // [2 5 1] * [1] = [8]
        // [0 3 6]   [1]   [9]
        let offsets = vec![-1, 0, 1];
        let data = vec![
            vec![2.0, 3.0, 0.0],
            vec![4.0, 5.0, 6.0],
            vec![0.0, 1.0, 1.0],
        ];

        let dia = DiaMatrix::new(3, 3, offsets, data).unwrap();
        let x = vec![1.0, 1.0, 1.0];
        let y = dia.mul_vec(&x);

        assert!((y[0] - 5.0).abs() < 1e-10);
        assert!((y[1] - 8.0).abs() < 1e-10);
        assert!((y[2] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_dia_tridiagonal() {
        let sub = vec![2.0, 3.0];
        let main = vec![4.0, 5.0, 6.0];
        let super_diag = vec![1.0, 1.0];

        let dia = DiaMatrix::tridiagonal(sub, main, super_diag).unwrap();

        assert_eq!(dia.get(0, 0), Some(&4.0));
        assert_eq!(dia.get(0, 1), Some(&1.0));
        assert_eq!(dia.get(1, 0), Some(&2.0));
        assert_eq!(dia.get(1, 1), Some(&5.0));
        assert_eq!(dia.get(2, 2), Some(&6.0));
    }

    #[test]
    fn test_dia_eye() {
        let dia: DiaMatrix<f64> = DiaMatrix::eye(4);

        assert_eq!(dia.nrows(), 4);
        assert_eq!(dia.ncols(), 4);
        assert_eq!(dia.ndiag(), 1);

        for i in 0..4 {
            assert_eq!(dia.get(i, i), Some(&1.0));
        }
    }

    #[test]
    fn test_dia_to_dense() {
        let offsets = vec![-1, 0, 1];
        let data = vec![
            vec![2.0, 3.0, 0.0],
            vec![4.0, 5.0, 6.0],
            vec![0.0, 1.0, 1.0],
        ];

        let dia = DiaMatrix::new(3, 3, offsets, data).unwrap();
        let dense = dia.to_dense();

        assert!((dense[(0, 0)] - 4.0).abs() < 1e-10);
        assert!((dense[(0, 1)] - 1.0).abs() < 1e-10);
        assert!((dense[(0, 2)] - 0.0).abs() < 1e-10);
        assert!((dense[(1, 0)] - 2.0).abs() < 1e-10);
        assert!((dense[(1, 1)] - 5.0).abs() < 1e-10);
        assert!((dense[(1, 2)] - 1.0).abs() < 1e-10);
        assert!((dense[(2, 0)] - 0.0).abs() < 1e-10);
        assert!((dense[(2, 1)] - 3.0).abs() < 1e-10);
        assert!((dense[(2, 2)] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_dia_to_csr() {
        let offsets = vec![-1, 0, 1];
        let data = vec![
            vec![2.0, 3.0, 0.0],
            vec![4.0, 5.0, 6.0],
            vec![0.0, 1.0, 1.0],
        ];

        let dia = DiaMatrix::new(3, 3, offsets, data).unwrap();
        let csr = dia.to_csr();

        assert_eq!(csr.nrows(), 3);
        assert_eq!(csr.ncols(), 3);
        assert_eq!(csr.get(0, 0), Some(&4.0));
        assert_eq!(csr.get(0, 1), Some(&1.0));
        assert_eq!(csr.get(1, 0), Some(&2.0));
    }

    #[test]
    fn test_dia_from_dense() {
        use oxiblas_matrix::Mat;

        let dense = Mat::from_rows(&[&[4.0f64, 1.0, 0.0], &[2.0, 5.0, 1.0], &[0.0, 3.0, 6.0]]);

        let dia = DiaMatrix::from_dense(&dense.as_ref(), None);

        assert_eq!(dia.ndiag(), 3);
        assert_eq!(dia.get(0, 0), Some(&4.0));
        assert_eq!(dia.get(1, 1), Some(&5.0));
    }

    #[test]
    fn test_dia_scale() {
        let offsets = vec![0];
        let data = vec![vec![1.0, 2.0, 3.0]];

        let mut dia = DiaMatrix::new(3, 3, offsets, data).unwrap();
        dia.scale(2.0);

        assert_eq!(dia.get(0, 0), Some(&2.0));
        assert_eq!(dia.get(1, 1), Some(&4.0));
        assert_eq!(dia.get(2, 2), Some(&6.0));
    }

    #[test]
    fn test_dia_transpose() {
        let offsets = vec![-1, 0, 1];
        let data = vec![
            vec![2.0, 3.0, 0.0],
            vec![4.0, 5.0, 6.0],
            vec![0.0, 1.0, 1.0],
        ];

        let dia = DiaMatrix::new(3, 3, offsets, data).unwrap();
        let dia_t = dia.transpose();

        // Original: A[1,0] = 2, A^T[0,1] = 2
        assert_eq!(dia.get(1, 0), Some(&2.0));
        // The transposed value should be at the corresponding position
        let dense = dia.to_dense();
        let dense_t = dia_t.to_dense();

        for i in 0..3 {
            for j in 0..3 {
                assert!((dense[(i, j)] - dense_t[(j, i)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_dia_rectangular() {
        // 4x3 matrix with 2 diagonals
        let offsets = vec![0, 1];
        let data = vec![
            vec![1.0, 2.0, 3.0], // main diagonal
            vec![0.0, 4.0, 5.0], // super diagonal
        ];

        let dia = DiaMatrix::new(4, 3, offsets, data).unwrap();

        assert_eq!(dia.nrows(), 4);
        assert_eq!(dia.ncols(), 3);
        assert_eq!(dia.get(0, 0), Some(&1.0));
        assert_eq!(dia.get(1, 1), Some(&2.0));
        assert_eq!(dia.get(2, 2), Some(&3.0));
        assert_eq!(dia.get(0, 1), Some(&4.0));
        assert_eq!(dia.get(1, 2), Some(&5.0));
    }

    #[test]
    fn test_dia_invalid_duplicate_offset() {
        let offsets = vec![0, 0]; // Duplicate
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

        let result = DiaMatrix::new(2, 2, offsets, data);
        assert!(matches!(result, Err(DiaError::DuplicateOffset { .. })));
    }

    #[test]
    fn test_dia_nnz() {
        let offsets = vec![-1, 0, 1];
        let data = vec![
            vec![2.0, 3.0, 0.0], // One padding zero
            vec![4.0, 5.0, 6.0],
            vec![0.0, 1.0, 1.0], // One padding zero
        ];

        let dia = DiaMatrix::new(3, 3, offsets, data).unwrap();
        // Actual non-zeros: 2,3 on sub, 4,5,6 on main, 1,1 on super = 7
        assert_eq!(dia.nnz(), 7);
    }
}
