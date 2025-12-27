//! ELLPACK (ELL) sparse matrix format.
//!
//! ELL stores matrix data using:
//! - `data`: 2D array of shape (nrows × max_nnz_per_row)
//! - `indices`: 2D array of column indices, same shape as data
//!
//! For an m×n matrix:
//! - Each row stores exactly `max_nnz_per_row` entries
//! - Rows with fewer non-zeros are padded with zeros and invalid indices
//!
//! # When to Use ELL
//!
//! ELL format is optimal for:
//! - Matrices with roughly uniform number of non-zeros per row
//! - GPU computation (enables coalesced memory access)
//! - Vector processors with SIMD operations
//!
//! ELL is NOT efficient for:
//! - Matrices with varying non-zeros per row (wastes memory on padding)
//! - Power-law graphs (a few rows have many entries)

use oxiblas_core::scalar::{Field, Scalar};

/// Error type for ELL matrix operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EllError {
    /// Data array has wrong dimensions.
    InvalidDataDimensions {
        /// Expected rows.
        expected_rows: usize,
        /// Actual rows.
        actual_rows: usize,
        /// Expected columns per row.
        expected_width: usize,
        /// Actual columns per row.
        actual_width: usize,
    },
    /// Data and indices have different dimensions.
    DimensionMismatch {
        /// Data dimensions (rows, width).
        data_dims: (usize, usize),
        /// Indices dimensions (rows, width).
        indices_dims: (usize, usize),
    },
    /// Column index out of bounds.
    InvalidColumnIndex {
        /// Row where invalid index found.
        row: usize,
        /// Position within row.
        pos: usize,
        /// The invalid index.
        index: usize,
        /// Number of columns.
        ncols: usize,
    },
    /// Too many non-zeros in a row.
    TooManyNonZeros {
        /// Row with too many non-zeros.
        row: usize,
        /// Number of non-zeros.
        nnz: usize,
        /// Maximum allowed.
        max_nnz: usize,
    },
}

impl core::fmt::Display for EllError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidDataDimensions {
                expected_rows,
                actual_rows,
                expected_width,
                actual_width,
            } => {
                write!(
                    f,
                    "Invalid data dimensions: expected {expected_rows}×{expected_width}, got {actual_rows}×{actual_width}"
                )
            }
            Self::DimensionMismatch {
                data_dims,
                indices_dims,
            } => {
                write!(
                    f,
                    "Dimension mismatch: data is {}×{}, indices is {}×{}",
                    data_dims.0, data_dims.1, indices_dims.0, indices_dims.1
                )
            }
            Self::InvalidColumnIndex {
                row,
                pos,
                index,
                ncols,
            } => {
                write!(
                    f,
                    "Invalid column index {index} at row {row}, position {pos} (ncols={ncols})"
                )
            }
            Self::TooManyNonZeros { row, nnz, max_nnz } => {
                write!(f, "Row {row} has {nnz} non-zeros, exceeds max {max_nnz}")
            }
        }
    }
}

impl std::error::Error for EllError {}

/// ELLPACK sparse matrix format.
///
/// Efficient for:
/// - GPU computation
/// - Vectorized operations
/// - Matrices with uniform row lengths
///
/// # Storage
///
/// Each row stores exactly `width` entries. The `width` is typically the maximum
/// number of non-zeros in any row. Rows with fewer non-zeros are padded with
/// zeros and a special "invalid" column index (usually ncols or usize::MAX).
///
/// # Example
///
/// ```
/// use oxiblas_sparse::EllMatrix;
///
/// // Create a sparse matrix:
/// // [1 2 0 0]
/// // [0 3 4 0]
/// // [5 0 0 6]
/// let width = 2; // max 2 non-zeros per row
/// let data = vec![
///     vec![1.0, 2.0],  // row 0
///     vec![3.0, 4.0],  // row 1
///     vec![5.0, 6.0],  // row 2
/// ];
/// let indices = vec![
///     vec![0, 1],  // row 0
///     vec![1, 2],  // row 1
///     vec![0, 3],  // row 2
/// ];
///
/// let ell = EllMatrix::new(3, 4, width, data, indices).unwrap();
/// assert_eq!(ell.width(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct EllMatrix<T: Scalar> {
    /// Number of rows.
    nrows: usize,
    /// Number of columns.
    ncols: usize,
    /// Maximum number of non-zeros per row (width of data/indices arrays).
    width: usize,
    /// Data array: data[row][k] is the k-th non-zero value in row.
    data: Vec<Vec<T>>,
    /// Column indices: indices[row][k] is the column of data[row][k].
    indices: Vec<Vec<usize>>,
}

/// Sentinel value for invalid/padding column indices.
const INVALID_INDEX: usize = usize::MAX;

impl<T: Scalar + Clone> EllMatrix<T> {
    /// Creates a new ELL matrix from raw components.
    ///
    /// # Arguments
    ///
    /// * `nrows` - Number of rows
    /// * `ncols` - Number of columns
    /// * `width` - Maximum non-zeros per row
    /// * `data` - Data array (nrows × width)
    /// * `indices` - Column indices array (nrows × width)
    ///
    /// # Errors
    ///
    /// Returns an error if the input is invalid.
    pub fn new(
        nrows: usize,
        ncols: usize,
        width: usize,
        data: Vec<Vec<T>>,
        indices: Vec<Vec<usize>>,
    ) -> Result<Self, EllError> {
        // Validate data dimensions
        if data.len() != nrows {
            return Err(EllError::InvalidDataDimensions {
                expected_rows: nrows,
                actual_rows: data.len(),
                expected_width: width,
                actual_width: if data.is_empty() { 0 } else { data[0].len() },
            });
        }

        for (i, row) in data.iter().enumerate() {
            if row.len() != width {
                return Err(EllError::InvalidDataDimensions {
                    expected_rows: nrows,
                    actual_rows: data.len(),
                    expected_width: width,
                    actual_width: row.len(),
                });
            }

            // Check corresponding indices row
            if i < indices.len() && indices[i].len() != width {
                return Err(EllError::DimensionMismatch {
                    data_dims: (nrows, width),
                    indices_dims: (indices.len(), indices[i].len()),
                });
            }
        }

        // Validate indices dimensions
        if indices.len() != nrows {
            return Err(EllError::DimensionMismatch {
                data_dims: (nrows, width),
                indices_dims: (
                    indices.len(),
                    if indices.is_empty() {
                        0
                    } else {
                        indices[0].len()
                    },
                ),
            });
        }

        // Validate column indices
        for (row, row_indices) in indices.iter().enumerate() {
            for (pos, &col) in row_indices.iter().enumerate() {
                if col != INVALID_INDEX && col >= ncols {
                    return Err(EllError::InvalidColumnIndex {
                        row,
                        pos,
                        index: col,
                        ncols,
                    });
                }
            }
        }

        Ok(Self {
            nrows,
            ncols,
            width,
            data,
            indices,
        })
    }

    /// Creates an ELL matrix without validation (unsafe but faster).
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `data.len() == nrows` and each row has length `width`
    /// - `indices.len() == nrows` and each row has length `width`
    /// - All valid column indices are < ncols
    #[inline]
    pub unsafe fn new_unchecked(
        nrows: usize,
        ncols: usize,
        width: usize,
        data: Vec<Vec<T>>,
        indices: Vec<Vec<usize>>,
    ) -> Self {
        Self {
            nrows,
            ncols,
            width,
            data,
            indices,
        }
    }

    /// Creates an empty ELL matrix with given dimensions.
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            width: 0,
            data: vec![Vec::new(); nrows],
            indices: vec![Vec::new(); nrows],
        }
    }

    /// Creates an identity matrix in ELL format.
    pub fn eye(n: usize) -> Self
    where
        T: Field,
    {
        Self {
            nrows: n,
            ncols: n,
            width: 1,
            data: (0..n).map(|_| vec![T::one()]).collect(),
            indices: (0..n).map(|i| vec![i]).collect(),
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

    /// Returns the width (max non-zeros per row).
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the number of non-zero elements.
    ///
    /// Note: This counts actual non-zeros, not stored values.
    pub fn nnz(&self) -> usize
    where
        T: Field,
    {
        let eps = <T as Scalar>::epsilon();
        let mut count = 0;

        for (row, indices_row) in self.indices.iter().enumerate() {
            for (k, &col) in indices_row.iter().enumerate() {
                if col != INVALID_INDEX && Scalar::abs(self.data[row][k].clone()) > eps {
                    count += 1;
                }
            }
        }

        count
    }

    /// Returns the total stored values (including padding).
    #[inline]
    pub fn nstored(&self) -> usize {
        self.nrows * self.width
    }

    /// Returns the storage efficiency (nnz / nstored).
    pub fn efficiency(&self) -> f64
    where
        T: Field,
    {
        if self.nstored() == 0 {
            1.0
        } else {
            self.nnz() as f64 / self.nstored() as f64
        }
    }

    /// Returns a reference to the data array.
    #[inline]
    pub fn data(&self) -> &[Vec<T>] {
        &self.data
    }

    /// Returns a mutable reference to the data array.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [Vec<T>] {
        &mut self.data
    }

    /// Returns a reference to the indices array.
    #[inline]
    pub fn indices(&self) -> &[Vec<usize>] {
        &self.indices
    }

    /// Gets the value at (row, col), returning None if not present.
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row >= self.nrows || col >= self.ncols {
            return None;
        }

        for k in 0..self.width {
            if self.indices[row][k] == col {
                return Some(&self.data[row][k]);
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

    /// Returns an iterator over non-zeros in a row as (col, value).
    pub fn row_iter(&self, row: usize) -> impl Iterator<Item = (usize, &T)> {
        self.indices[row]
            .iter()
            .zip(self.data[row].iter())
            .filter(|(col, _)| **col != INVALID_INDEX)
            .map(|(col, val)| (*col, val))
    }

    /// Returns an iterator over all non-zeros as (row, col, value).
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, &T)> + '_ {
        (0..self.nrows).flat_map(move |row| {
            self.indices[row]
                .iter()
                .zip(self.data[row].iter())
                .filter(|(col, _)| **col != INVALID_INDEX)
                .map(move |(col, val)| (row, *col, val))
        })
    }

    /// Matrix-vector product: y = A * x.
    pub fn matvec(&self, x: &[T], y: &mut [T])
    where
        T: Field,
    {
        assert_eq!(x.len(), self.ncols, "x length must equal ncols");
        assert_eq!(y.len(), self.nrows, "y length must equal nrows");

        for row in 0..self.nrows {
            let mut sum = T::zero();
            for k in 0..self.width {
                let col = self.indices[row][k];
                if col != INVALID_INDEX {
                    sum = sum + self.data[row][k].clone() * x[col].clone();
                }
            }
            y[row] = sum;
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

            for k in 0..self.width {
                let col = self.indices[row][k];
                if col != INVALID_INDEX {
                    let val = self.data[row][k].clone();
                    if Scalar::abs(val.clone()) > eps {
                        row_entries.push((col, val));
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

        for row in 0..self.nrows {
            for k in 0..self.width {
                let col = self.indices[row][k];
                if col != INVALID_INDEX {
                    dense[(row, col)] = self.data[row][k].clone();
                }
            }
        }

        dense
    }

    /// Creates an ELL matrix from a dense matrix.
    ///
    /// # Arguments
    ///
    /// * `dense` - Source dense matrix
    /// * `max_width` - Maximum width (if None, uses actual max non-zeros per row)
    pub fn from_dense(dense: &oxiblas_matrix::MatRef<'_, T>, max_width: Option<usize>) -> Self
    where
        T: Field,
    {
        let (nrows, ncols) = dense.shape();
        let eps = <T as Scalar>::epsilon();

        // First pass: find max non-zeros per row
        let mut row_nnz = vec![0usize; nrows];
        for i in 0..nrows {
            for j in 0..ncols {
                if Scalar::abs(dense[(i, j)].clone()) > eps {
                    row_nnz[i] += 1;
                }
            }
        }

        let width = max_width.unwrap_or_else(|| row_nnz.iter().copied().max().unwrap_or(0));

        // Build data and indices
        let mut data = Vec::with_capacity(nrows);
        let mut indices = Vec::with_capacity(nrows);

        for i in 0..nrows {
            let mut row_data = Vec::with_capacity(width);
            let mut row_indices = Vec::with_capacity(width);

            for j in 0..ncols {
                if row_data.len() >= width {
                    break;
                }
                let val = dense[(i, j)].clone();
                if Scalar::abs(val.clone()) > eps {
                    row_data.push(val);
                    row_indices.push(j);
                }
            }

            // Pad to width
            while row_data.len() < width {
                row_data.push(T::zero());
                row_indices.push(INVALID_INDEX);
            }

            data.push(row_data);
            indices.push(row_indices);
        }

        // Safety: we constructed valid ELL data
        unsafe { Self::new_unchecked(nrows, ncols, width, data, indices) }
    }

    /// Creates an ELL matrix from CSR format.
    ///
    /// # Arguments
    ///
    /// * `csr` - Source CSR matrix
    /// * `max_width` - Maximum width (if None, uses actual max non-zeros per row)
    pub fn from_csr(
        csr: &crate::csr::CsrMatrix<T>,
        max_width: Option<usize>,
    ) -> Result<Self, EllError>
    where
        T: Field,
    {
        let (nrows, ncols) = csr.shape();
        let row_ptrs = csr.row_ptrs();
        let csr_indices = csr.col_indices();
        let csr_values = csr.values();

        // Find max row width
        let actual_max: usize = (0..nrows)
            .map(|i| row_ptrs[i + 1] - row_ptrs[i])
            .max()
            .unwrap_or(0);

        let width = max_width.unwrap_or(actual_max);

        // Check if any row exceeds max_width
        if let Some(max_w) = max_width {
            for row in 0..nrows {
                let row_nnz = row_ptrs[row + 1] - row_ptrs[row];
                if row_nnz > max_w {
                    return Err(EllError::TooManyNonZeros {
                        row,
                        nnz: row_nnz,
                        max_nnz: max_w,
                    });
                }
            }
        }

        // Build data and indices
        let mut data = Vec::with_capacity(nrows);
        let mut indices = Vec::with_capacity(nrows);

        for row in 0..nrows {
            let start = row_ptrs[row];
            let end = row_ptrs[row + 1];
            let row_nnz = end - start;

            let mut row_data = Vec::with_capacity(width);
            let mut row_indices = Vec::with_capacity(width);

            for k in 0..row_nnz {
                row_data.push(csr_values[start + k].clone());
                row_indices.push(csr_indices[start + k]);
            }

            // Pad to width
            while row_data.len() < width {
                row_data.push(T::zero());
                row_indices.push(INVALID_INDEX);
            }

            data.push(row_data);
            indices.push(row_indices);
        }

        Ok(Self {
            nrows,
            ncols,
            width,
            data,
            indices,
        })
    }

    /// Scales all values by a scalar.
    pub fn scale(&mut self, alpha: T) {
        for row in &mut self.data {
            for val in row.iter_mut() {
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
    ///
    /// Note: This is less efficient than CSR/CSC transpose.
    pub fn transpose(&self) -> Self
    where
        T: Field,
    {
        // Convert to CSR, transpose, convert back
        let csr = self.to_csr();
        let csr_t = csr.transpose();
        Self::from_csr(&csr_t, Some(self.width)).unwrap_or_else(|_| {
            // If width is insufficient, use actual max
            Self::from_csr(&csr_t, None).expect("CSR transpose should be valid")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ell_new() {
        // [1 2 0 0]
        // [0 3 4 0]
        // [5 0 0 6]
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let indices = vec![vec![0, 1], vec![1, 2], vec![0, 3]];

        let ell = EllMatrix::new(3, 4, 2, data, indices).unwrap();

        assert_eq!(ell.nrows(), 3);
        assert_eq!(ell.ncols(), 4);
        assert_eq!(ell.width(), 2);
    }

    #[test]
    fn test_ell_get() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let indices = vec![vec![0, 1], vec![1, 2], vec![0, 3]];

        let ell = EllMatrix::new(3, 4, 2, data, indices).unwrap();

        assert_eq!(ell.get(0, 0), Some(&1.0));
        assert_eq!(ell.get(0, 1), Some(&2.0));
        assert_eq!(ell.get(1, 1), Some(&3.0));
        assert_eq!(ell.get(1, 2), Some(&4.0));
        assert_eq!(ell.get(2, 0), Some(&5.0));
        assert_eq!(ell.get(2, 3), Some(&6.0));

        // Zero elements
        assert_eq!(ell.get(0, 2), None);
        assert_eq!(ell.get(0, 3), None);
    }

    #[test]
    fn test_ell_matvec() {
        // [1 2 0 0]   [1]   [3]
        // [0 3 4 0] * [1] = [7]
        // [5 0 0 6]   [1]   [11]
        //             [1]
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let indices = vec![vec![0, 1], vec![1, 2], vec![0, 3]];

        let ell = EllMatrix::new(3, 4, 2, data, indices).unwrap();
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let y = ell.mul_vec(&x);

        assert!((y[0] - 3.0).abs() < 1e-10);
        assert!((y[1] - 7.0).abs() < 1e-10);
        assert!((y[2] - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_ell_with_padding() {
        // [1 0 0]
        // [2 3 4]
        // [0 5 0]
        let data = vec![
            vec![1.0, 0.0, 0.0], // row 0: 1 value, padded
            vec![2.0, 3.0, 4.0], // row 1: 3 values
            vec![5.0, 0.0, 0.0], // row 2: 1 value, padded
        ];
        let indices = vec![
            vec![0, INVALID_INDEX, INVALID_INDEX],
            vec![0, 1, 2],
            vec![1, INVALID_INDEX, INVALID_INDEX],
        ];

        let ell = EllMatrix::new(3, 3, 3, data, indices).unwrap();

        assert_eq!(ell.nnz(), 5);
        assert_eq!(ell.nstored(), 9);
        assert!((ell.efficiency() - 5.0 / 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_ell_eye() {
        let ell: EllMatrix<f64> = EllMatrix::eye(4);

        assert_eq!(ell.nrows(), 4);
        assert_eq!(ell.ncols(), 4);
        assert_eq!(ell.width(), 1);

        for i in 0..4 {
            assert_eq!(ell.get(i, i), Some(&1.0));
        }
    }

    #[test]
    fn test_ell_to_dense() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let indices = vec![vec![0, 1], vec![1, 2], vec![0, 3]];

        let ell = EllMatrix::new(3, 4, 2, data, indices).unwrap();
        let dense = ell.to_dense();

        assert!((dense[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((dense[(0, 1)] - 2.0).abs() < 1e-10);
        assert!((dense[(0, 2)] - 0.0).abs() < 1e-10);
        assert!((dense[(1, 1)] - 3.0).abs() < 1e-10);
        assert!((dense[(1, 2)] - 4.0).abs() < 1e-10);
        assert!((dense[(2, 0)] - 5.0).abs() < 1e-10);
        assert!((dense[(2, 3)] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_ell_to_csr() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let indices = vec![vec![0, 1], vec![1, 2], vec![0, 3]];

        let ell = EllMatrix::new(3, 4, 2, data, indices).unwrap();
        let csr = ell.to_csr();

        assert_eq!(csr.nrows(), 3);
        assert_eq!(csr.ncols(), 4);
        assert_eq!(csr.nnz(), 6);
        assert_eq!(csr.get(0, 0), Some(&1.0));
        assert_eq!(csr.get(2, 3), Some(&6.0));
    }

    #[test]
    fn test_ell_from_dense() {
        use oxiblas_matrix::Mat;

        let dense = Mat::from_rows(&[
            &[1.0f64, 2.0, 0.0, 0.0],
            &[0.0, 3.0, 4.0, 0.0],
            &[5.0, 0.0, 0.0, 6.0],
        ]);

        let ell = EllMatrix::from_dense(&dense.as_ref(), None);

        assert_eq!(ell.width(), 2);
        assert_eq!(ell.get(0, 0), Some(&1.0));
        assert_eq!(ell.get(1, 2), Some(&4.0));
    }

    #[test]
    fn test_ell_from_csr() {
        let values = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let col_indices = vec![0, 1, 1, 2, 0, 3];
        let row_ptrs = vec![0, 2, 4, 6];

        let csr = crate::csr::CsrMatrix::new(3, 4, row_ptrs, col_indices, values).unwrap();
        let ell = EllMatrix::from_csr(&csr, None).unwrap();

        assert_eq!(ell.width(), 2);
        assert_eq!(ell.get(0, 0), Some(&1.0));
        assert_eq!(ell.get(2, 3), Some(&6.0));
    }

    #[test]
    fn test_ell_scale() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let indices = vec![vec![0, 1], vec![0, 1]];

        let mut ell = EllMatrix::new(2, 2, 2, data, indices).unwrap();
        ell.scale(2.0);

        assert_eq!(ell.get(0, 0), Some(&2.0));
        assert_eq!(ell.get(0, 1), Some(&4.0));
    }

    #[test]
    fn test_ell_transpose() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 0.0]];
        let indices = vec![vec![0, 1], vec![0, INVALID_INDEX]];

        let ell = EllMatrix::new(2, 2, 2, data, indices).unwrap();
        let ell_t = ell.transpose();

        let dense = ell.to_dense();
        let dense_t = ell_t.to_dense();

        for i in 0..2 {
            for j in 0..2 {
                assert!((dense[(i, j)] - dense_t[(j, i)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_ell_row_iter() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let indices = vec![vec![0, 2], vec![1, INVALID_INDEX]];

        let ell = EllMatrix::new(2, 3, 2, data, indices).unwrap();

        let row0: Vec<_> = ell.row_iter(0).collect();
        assert_eq!(row0, vec![(0, &1.0), (2, &2.0)]);

        let row1: Vec<_> = ell.row_iter(1).collect();
        assert_eq!(row1, vec![(1, &3.0)]);
    }

    #[test]
    fn test_ell_invalid_column_index() {
        let data = vec![vec![1.0]];
        let indices = vec![vec![10]]; // Out of bounds

        let result = EllMatrix::new(1, 3, 1, data, indices);
        assert!(matches!(result, Err(EllError::InvalidColumnIndex { .. })));
    }

    #[test]
    fn test_ell_zeros() {
        let ell: EllMatrix<f64> = EllMatrix::zeros(5, 3);

        assert_eq!(ell.nrows(), 5);
        assert_eq!(ell.ncols(), 3);
        assert_eq!(ell.width(), 0);
        assert_eq!(ell.nnz(), 0);
    }
}
