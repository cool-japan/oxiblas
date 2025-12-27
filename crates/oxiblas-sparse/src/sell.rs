//! Sliced ELLPACK (SELL) matrix format.
//!
//! SELL stores sparse matrices by dividing rows into fixed-size slices,
//! where each slice uses a different ELL width based on its densest row.
//!
//! # Key Features
//!
//! - **Sliced storage**: Rows are grouped into slices of size C (e.g., 32, 64, 256)
//! - **Per-slice width**: Each slice has its own ELL width = max row length in slice
//! - **Better memory efficiency**: Avoids padding waste for highly variable row lengths
//! - **GPU-friendly**: Slice size typically matches warp/wavefront size
//!
//! # When to Use SELL
//!
//! SELL format is optimal for:
//! - Matrices with highly variable row lengths
//! - GPU computation where ELL wastes too much memory
//! - Matrices where rows naturally cluster by density
//!
//! # Slice Size Selection
//!
//! Common slice sizes:
//! - C=32: Matches NVIDIA warp size
//! - C=64: Matches AMD wavefront size
//! - C=256: Good balance for CPU SIMD

use crate::csr::CsrMatrix;
use oxiblas_core::scalar::{Field, Scalar};

/// Error type for SELL matrix operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SellError {
    /// Invalid dimensions.
    InvalidDimensions {
        /// Number of rows.
        nrows: usize,
        /// Number of columns.
        ncols: usize,
    },
    /// Invalid slice size.
    InvalidSliceSize {
        /// The invalid slice size.
        slice_size: usize,
    },
    /// Data length mismatch.
    DataLengthMismatch {
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },
}

impl core::fmt::Display for SellError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidDimensions { nrows, ncols } => {
                write!(f, "Invalid dimensions: {nrows}×{ncols}")
            }
            Self::InvalidSliceSize { slice_size } => {
                write!(f, "Invalid slice size: {slice_size}")
            }
            Self::DataLengthMismatch { expected, actual } => {
                write!(f, "Data length mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for SellError {}

/// Common slice sizes for different architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SliceSize {
    /// NVIDIA warp size (32 threads).
    #[default]
    Warp32,
    /// AMD wavefront size (64 threads).
    Wavefront64,
    /// CPU SIMD friendly (256 elements).
    Simd256,
    /// Custom slice size.
    Custom(usize),
}

impl SliceSize {
    /// Returns the slice size as usize.
    #[inline]
    pub fn value(self) -> usize {
        match self {
            Self::Warp32 => 32,
            Self::Wavefront64 => 64,
            Self::Simd256 => 256,
            Self::Custom(c) => c,
        }
    }
}

impl From<usize> for SliceSize {
    fn from(size: usize) -> Self {
        match size {
            32 => Self::Warp32,
            64 => Self::Wavefront64,
            256 => Self::Simd256,
            c => Self::Custom(c),
        }
    }
}

/// Sliced ELLPACK matrix format.
///
/// Divides rows into slices of size C, with each slice having its own width.
///
/// # Storage
///
/// For a matrix with m rows and slice size C:
/// - Number of slices: ceil(m/C)
/// - Each slice has width = max row length in that slice
/// - Data is stored slice by slice, with column-major ordering within slices
///
/// # Example
///
/// ```
/// use oxiblas_sparse::{CsrMatrix, SellMatrix, SliceSize};
///
/// // Create a sparse matrix
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let col_indices = vec![0, 2, 1, 0, 2];
/// let row_ptrs = vec![0, 2, 3, 5];
/// let csr = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();
///
/// // Convert to SELL with slice size 32
/// let sell = SellMatrix::from_csr(&csr, SliceSize::Warp32);
/// assert_eq!(sell.shape(), (3, 3));
/// ```
#[derive(Debug, Clone)]
pub struct SellMatrix<T: Scalar> {
    /// Number of rows.
    nrows: usize,
    /// Number of columns.
    ncols: usize,
    /// Slice size C.
    slice_size: usize,
    /// Number of slices.
    num_slices: usize,
    /// Slice pointers into data arrays (length: num_slices + 1).
    slice_ptrs: Vec<usize>,
    /// Width of each slice (length: num_slices).
    slice_widths: Vec<usize>,
    /// Column indices (column-major within each slice).
    col_indices: Vec<usize>,
    /// Values (column-major within each slice).
    values: Vec<T>,
}

impl<T: Scalar + Clone> SellMatrix<T> {
    /// Creates a new SELL matrix from raw components.
    ///
    /// # Arguments
    ///
    /// * `nrows` - Number of rows
    /// * `ncols` - Number of columns
    /// * `slice_size` - Size of each slice
    /// * `slice_ptrs` - Pointers into data arrays (length: num_slices + 1)
    /// * `slice_widths` - Width of each slice
    /// * `col_indices` - Column indices
    /// * `values` - Values
    ///
    /// # Errors
    ///
    /// Returns an error if the input is invalid.
    pub fn new(
        nrows: usize,
        ncols: usize,
        slice_size: usize,
        slice_ptrs: Vec<usize>,
        slice_widths: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self, SellError> {
        if slice_size == 0 {
            return Err(SellError::InvalidSliceSize { slice_size });
        }

        let num_slices = nrows.div_ceil(slice_size);

        if slice_ptrs.len() != num_slices + 1 {
            return Err(SellError::DataLengthMismatch {
                expected: num_slices + 1,
                actual: slice_ptrs.len(),
            });
        }

        if slice_widths.len() != num_slices {
            return Err(SellError::DataLengthMismatch {
                expected: num_slices,
                actual: slice_widths.len(),
            });
        }

        if col_indices.len() != values.len() {
            return Err(SellError::DataLengthMismatch {
                expected: col_indices.len(),
                actual: values.len(),
            });
        }

        Ok(Self {
            nrows,
            ncols,
            slice_size,
            num_slices,
            slice_ptrs,
            slice_widths,
            col_indices,
            values,
        })
    }

    /// Creates an empty SELL matrix.
    pub fn zeros(nrows: usize, ncols: usize, slice_size: SliceSize) -> Self
    where
        T: Field,
    {
        let c = slice_size.value();
        let num_slices = nrows.div_ceil(c);

        Self {
            nrows,
            ncols,
            slice_size: c,
            num_slices,
            slice_ptrs: vec![0; num_slices + 1],
            slice_widths: vec![0; num_slices],
            col_indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Creates an identity matrix in SELL format.
    pub fn eye(n: usize, slice_size: SliceSize) -> Self
    where
        T: Field,
    {
        let c = slice_size.value();
        let num_slices = n.div_ceil(c);

        let mut slice_ptrs = Vec::with_capacity(num_slices + 1);
        let mut slice_widths = Vec::with_capacity(num_slices);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        slice_ptrs.push(0);

        for s in 0..num_slices {
            let slice_start = s * c;
            let slice_end = ((s + 1) * c).min(n);
            let rows_in_slice = slice_end - slice_start;

            // Identity has 1 entry per row
            slice_widths.push(1);

            // Column-major: store all column 0s, then column 1s, etc.
            // For identity, each row has entry at its diagonal
            for row_in_slice in 0..rows_in_slice {
                let global_row = slice_start + row_in_slice;
                col_indices.push(global_row);
                values.push(T::one());
            }

            // Pad remaining rows in slice
            for _ in rows_in_slice..c {
                col_indices.push(0);
                values.push(T::zero());
            }

            slice_ptrs.push(col_indices.len());
        }

        Self {
            nrows: n,
            ncols: n,
            slice_size: c,
            num_slices,
            slice_ptrs,
            slice_widths,
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

    /// Returns the slice size.
    #[inline]
    pub fn slice_size(&self) -> usize {
        self.slice_size
    }

    /// Returns the number of slices.
    #[inline]
    pub fn num_slices(&self) -> usize {
        self.num_slices
    }

    /// Returns the width of each slice.
    #[inline]
    pub fn slice_widths(&self) -> &[usize] {
        &self.slice_widths
    }

    /// Returns the slice pointers.
    #[inline]
    pub fn slice_ptrs(&self) -> &[usize] {
        &self.slice_ptrs
    }

    /// Returns the column indices.
    #[inline]
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// Returns the values.
    #[inline]
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Returns the number of non-zeros.
    pub fn nnz(&self) -> usize
    where
        T: Field,
    {
        let eps = <T as Scalar>::epsilon();
        self.values
            .iter()
            .filter(|v| Scalar::abs((*v).clone()) > eps)
            .count()
    }

    /// Returns the total stored values (including padding zeros).
    #[inline]
    pub fn nstored(&self) -> usize {
        self.values.len()
    }

    /// Returns the storage efficiency (nnz / stored).
    pub fn storage_efficiency(&self) -> f64
    where
        T: Field,
    {
        let nnz = self.nnz();
        let stored = self.nstored();
        if stored == 0 {
            return 0.0;
        }
        nnz as f64 / stored as f64
    }

    /// Gets the value at (row, col).
    pub fn get(&self, row: usize, col: usize) -> Option<T>
    where
        T: Field,
    {
        if row >= self.nrows || col >= self.ncols {
            return None;
        }

        let eps = <T as Scalar>::epsilon();
        let slice_idx = row / self.slice_size;
        let row_in_slice = row % self.slice_size;

        let slice_start = self.slice_ptrs[slice_idx];
        let width = self.slice_widths[slice_idx];

        // Search through columns in this row
        for k in 0..width {
            let idx = slice_start + k * self.slice_size + row_in_slice;
            if self.col_indices[idx] == col {
                let val = self.values[idx].clone();
                if Scalar::abs(val.clone()) > eps {
                    return Some(val);
                }
            }
        }

        None
    }

    /// Gets the value at (row, col), returning zero if not present.
    pub fn get_or_zero(&self, row: usize, col: usize) -> T
    where
        T: Field,
    {
        self.get(row, col).unwrap_or_else(T::zero)
    }

    /// Matrix-vector product: y = A * x.
    pub fn matvec(&self, x: &[T], y: &mut [T])
    where
        T: Field,
    {
        assert_eq!(x.len(), self.ncols, "x length must equal ncols");
        assert_eq!(y.len(), self.nrows, "y length must equal nrows");

        let eps = <T as Scalar>::epsilon();

        // Initialize y to zero
        for yi in y.iter_mut() {
            *yi = T::zero();
        }

        // Process each slice
        for s in 0..self.num_slices {
            let slice_start_row = s * self.slice_size;
            let slice_end_row = ((s + 1) * self.slice_size).min(self.nrows);
            let rows_in_slice = slice_end_row - slice_start_row;

            let data_start = self.slice_ptrs[s];
            let width = self.slice_widths[s];

            // Column-major iteration within slice for better vectorization
            for k in 0..width {
                let col_offset = data_start + k * self.slice_size;
                for row_in_slice in 0..rows_in_slice {
                    let idx = col_offset + row_in_slice;
                    let val = &self.values[idx];
                    if Scalar::abs(val.clone()) > eps {
                        let col = self.col_indices[idx];
                        let global_row = slice_start_row + row_in_slice;
                        y[global_row] = y[global_row].clone() + val.clone() * x[col].clone();
                    }
                }
            }
        }
    }

    /// Matrix-vector product returning a new vector.
    pub fn mul_vec(&self, x: &[T]) -> Vec<T>
    where
        T: Field,
    {
        let mut y = vec![T::zero(); self.nrows];
        self.matvec(x, &mut y);
        y
    }

    /// Transposed matrix-vector product: y = A^T * x.
    pub fn matvec_transpose(&self, x: &[T], y: &mut [T])
    where
        T: Field,
    {
        assert_eq!(x.len(), self.nrows, "x length must equal nrows");
        assert_eq!(y.len(), self.ncols, "y length must equal ncols");

        let eps = <T as Scalar>::epsilon();

        // Initialize y to zero
        for yi in y.iter_mut() {
            *yi = T::zero();
        }

        // Process each slice
        for s in 0..self.num_slices {
            let slice_start_row = s * self.slice_size;
            let slice_end_row = ((s + 1) * self.slice_size).min(self.nrows);
            let rows_in_slice = slice_end_row - slice_start_row;

            let data_start = self.slice_ptrs[s];
            let width = self.slice_widths[s];

            for k in 0..width {
                let col_offset = data_start + k * self.slice_size;
                for row_in_slice in 0..rows_in_slice {
                    let idx = col_offset + row_in_slice;
                    let val = &self.values[idx];
                    if Scalar::abs(val.clone()) > eps {
                        let col = self.col_indices[idx];
                        let global_row = slice_start_row + row_in_slice;
                        y[col] = y[col].clone() + val.clone() * x[global_row].clone();
                    }
                }
            }
        }
    }

    /// Creates SELL matrix from CSR format.
    pub fn from_csr(csr: &CsrMatrix<T>, slice_size: SliceSize) -> Self
    where
        T: Field,
    {
        let (nrows, ncols) = csr.shape();
        let c = slice_size.value();
        let eps = <T as Scalar>::epsilon();

        if nrows == 0 {
            return Self::zeros(nrows, ncols, slice_size);
        }

        let num_slices = nrows.div_ceil(c);

        let mut slice_ptrs = Vec::with_capacity(num_slices + 1);
        let mut slice_widths = Vec::with_capacity(num_slices);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        slice_ptrs.push(0);

        for s in 0..num_slices {
            let slice_start_row = s * c;
            let slice_end_row = ((s + 1) * c).min(nrows);
            let _rows_in_slice = slice_end_row - slice_start_row;

            // Compute max row length in this slice
            let mut max_width = 0;
            for row in slice_start_row..slice_end_row {
                let mut count = 0;
                for (_, val) in csr.row_iter(row) {
                    if Scalar::abs((*val).clone()) > eps {
                        count += 1;
                    }
                }
                max_width = max_width.max(count);
            }

            slice_widths.push(max_width);

            if max_width == 0 {
                slice_ptrs.push(col_indices.len());
                continue;
            }

            // Collect entries for each row in slice
            let mut row_entries: Vec<Vec<(usize, T)>> = Vec::with_capacity(c);
            for row in slice_start_row..slice_end_row {
                let entries: Vec<(usize, T)> = csr
                    .row_iter(row)
                    .filter(|(_, val)| Scalar::abs((*val).clone()) > eps)
                    .map(|(col, val)| (col, (*val).clone()))
                    .collect();
                row_entries.push(entries);
            }

            // Pad with empty rows if needed
            while row_entries.len() < c {
                row_entries.push(Vec::new());
            }

            // Store in column-major order within slice
            for k in 0..max_width {
                for row_in_slice in 0..c {
                    if k < row_entries[row_in_slice].len() {
                        let (col, val) = &row_entries[row_in_slice][k];
                        col_indices.push(*col);
                        values.push(val.clone());
                    } else {
                        col_indices.push(0);
                        values.push(T::zero());
                    }
                }
            }

            slice_ptrs.push(col_indices.len());
        }

        Self {
            nrows,
            ncols,
            slice_size: c,
            num_slices,
            slice_ptrs,
            slice_widths,
            col_indices,
            values,
        }
    }

    /// Converts to CSR format.
    pub fn to_csr(&self) -> CsrMatrix<T>
    where
        T: Field,
    {
        let eps = <T as Scalar>::epsilon();

        let mut row_ptrs = vec![0usize; self.nrows + 1];
        let mut csr_col_indices = Vec::new();
        let mut csr_values = Vec::new();

        for s in 0..self.num_slices {
            let slice_start_row = s * self.slice_size;
            let slice_end_row = ((s + 1) * self.slice_size).min(self.nrows);
            let rows_in_slice = slice_end_row - slice_start_row;

            let data_start = self.slice_ptrs[s];
            let width = self.slice_widths[s];

            // Collect entries for each row in slice
            let mut row_entries: Vec<Vec<(usize, T)>> = vec![Vec::new(); rows_in_slice];

            for k in 0..width {
                let col_offset = data_start + k * self.slice_size;
                for row_in_slice in 0..rows_in_slice {
                    let idx = col_offset + row_in_slice;
                    let val = self.values[idx].clone();
                    if Scalar::abs(val.clone()) > eps {
                        let col = self.col_indices[idx];
                        row_entries[row_in_slice].push((col, val));
                    }
                }
            }

            // Output rows in sorted order
            for (row_in_slice, entries) in row_entries.into_iter().enumerate() {
                let global_row = slice_start_row + row_in_slice;
                let mut sorted_entries = entries;
                sorted_entries.sort_by_key(|(col, _)| *col);

                for (col, val) in sorted_entries {
                    csr_col_indices.push(col);
                    csr_values.push(val);
                }
                row_ptrs[global_row + 1] = csr_col_indices.len();
            }
        }

        // Fill any remaining rows (shouldn't happen but just in case)
        for i in 1..=self.nrows {
            if row_ptrs[i] == 0 && row_ptrs[i - 1] > 0 {
                row_ptrs[i] = row_ptrs[i - 1];
            }
        }

        // Safety: we constructed valid CSR data
        unsafe {
            CsrMatrix::new_unchecked(
                self.nrows,
                self.ncols,
                row_ptrs,
                csr_col_indices,
                csr_values,
            )
        }
    }

    /// Converts to dense matrix.
    pub fn to_dense(&self) -> oxiblas_matrix::Mat<T>
    where
        T: Field + bytemuck::Zeroable,
    {
        let eps = <T as Scalar>::epsilon();
        let mut dense = oxiblas_matrix::Mat::zeros(self.nrows, self.ncols);

        for s in 0..self.num_slices {
            let slice_start_row = s * self.slice_size;
            let slice_end_row = ((s + 1) * self.slice_size).min(self.nrows);
            let rows_in_slice = slice_end_row - slice_start_row;

            let data_start = self.slice_ptrs[s];
            let width = self.slice_widths[s];

            for k in 0..width {
                let col_offset = data_start + k * self.slice_size;
                for row_in_slice in 0..rows_in_slice {
                    let idx = col_offset + row_in_slice;
                    let val = self.values[idx].clone();
                    if Scalar::abs(val.clone()) > eps {
                        let col = self.col_indices[idx];
                        let global_row = slice_start_row + row_in_slice;
                        dense[(global_row, col)] = val;
                    }
                }
            }
        }

        dense
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

    /// Returns an iterator over non-zero entries as (row, col, value).
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, T)> + '_
    where
        T: Field,
    {
        let eps = <T as Scalar>::epsilon();
        let slice_size = self.slice_size;
        let nrows = self.nrows;

        (0..self.num_slices).flat_map(move |s| {
            let slice_start_row = s * slice_size;
            let slice_end_row = ((s + 1) * slice_size).min(nrows);
            let rows_in_slice = slice_end_row - slice_start_row;

            let data_start = self.slice_ptrs[s];
            let width = self.slice_widths[s];

            (0..width).flat_map(move |k| {
                let col_offset = data_start + k * slice_size;
                (0..rows_in_slice).filter_map(move |row_in_slice| {
                    let idx = col_offset + row_in_slice;
                    let val = self.values[idx].clone();
                    if Scalar::abs(val.clone()) > eps {
                        let col = self.col_indices[idx];
                        let global_row = slice_start_row + row_in_slice;
                        Some((global_row, col, val))
                    } else {
                        None
                    }
                })
            })
        })
    }

    /// Returns statistics about the SELL matrix.
    pub fn stats(&self) -> SellStats
    where
        T: Field,
    {
        let max_width = self.slice_widths.iter().max().copied().unwrap_or(0);
        let min_width = self.slice_widths.iter().min().copied().unwrap_or(0);
        let avg_width = if self.num_slices > 0 {
            self.slice_widths.iter().sum::<usize>() as f64 / self.num_slices as f64
        } else {
            0.0
        };

        SellStats {
            nrows: self.nrows,
            ncols: self.ncols,
            slice_size: self.slice_size,
            num_slices: self.num_slices,
            nnz: self.nnz(),
            nstored: self.nstored(),
            max_slice_width: max_width,
            min_slice_width: min_width,
            avg_slice_width: avg_width,
            storage_efficiency: self.storage_efficiency(),
        }
    }
}

/// Statistics about a SELL matrix.
#[derive(Debug, Clone, Copy)]
pub struct SellStats {
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Slice size.
    pub slice_size: usize,
    /// Number of slices.
    pub num_slices: usize,
    /// Number of non-zeros.
    pub nnz: usize,
    /// Number of stored values (including padding).
    pub nstored: usize,
    /// Maximum slice width.
    pub max_slice_width: usize,
    /// Minimum slice width.
    pub min_slice_width: usize,
    /// Average slice width.
    pub avg_slice_width: f64,
    /// Storage efficiency (nnz / stored).
    pub storage_efficiency: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_csr() -> CsrMatrix<f64> {
        // [1 0 2 0]
        // [0 3 0 0]
        // [4 0 5 6]
        // [0 0 0 7]
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let col_indices = vec![0, 2, 1, 0, 2, 3, 3];
        let row_ptrs = vec![0, 2, 3, 6, 7];
        CsrMatrix::new(4, 4, row_ptrs, col_indices, values).unwrap()
    }

    #[test]
    fn test_sell_from_csr() {
        let csr = make_test_csr();
        let sell = SellMatrix::from_csr(&csr, SliceSize::Custom(2));

        assert_eq!(sell.nrows(), 4);
        assert_eq!(sell.ncols(), 4);
        assert_eq!(sell.slice_size(), 2);
        assert_eq!(sell.num_slices(), 2);
        assert_eq!(sell.nnz(), 7);
    }

    #[test]
    fn test_sell_matvec() {
        let csr = make_test_csr();
        let sell = SellMatrix::from_csr(&csr, SliceSize::Custom(2));

        // [1 0 2 0]   [1]   [7]
        // [0 3 0 0] * [2] = [6]
        // [4 0 5 6]   [3]   [43]
        // [0 0 0 7]   [4]   [28]
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = sell.mul_vec(&x);

        assert!((y[0] - 7.0).abs() < 1e-10);
        assert!((y[1] - 6.0).abs() < 1e-10);
        assert!((y[2] - 43.0).abs() < 1e-10);
        assert!((y[3] - 28.0).abs() < 1e-10);
    }

    #[test]
    fn test_sell_matvec_transpose() {
        let csr = make_test_csr();
        let sell = SellMatrix::from_csr(&csr, SliceSize::Custom(2));

        let x = vec![1.0, 1.0, 1.0, 1.0];
        let mut y = vec![0.0; 4];
        sell.matvec_transpose(&x, &mut y);

        // A^T * [1,1,1,1] = column sums
        assert!((y[0] - 5.0).abs() < 1e-10);
        assert!((y[1] - 3.0).abs() < 1e-10);
        assert!((y[2] - 7.0).abs() < 1e-10);
        assert!((y[3] - 13.0).abs() < 1e-10);
    }

    #[test]
    fn test_sell_to_csr_roundtrip() {
        let csr1 = make_test_csr();
        let sell = SellMatrix::from_csr(&csr1, SliceSize::Custom(2));
        let csr2 = sell.to_csr();

        assert_eq!(csr1.nnz(), csr2.nnz());

        for row in 0..4 {
            for col in 0..4 {
                let v1 = csr1.get(row, col).cloned().unwrap_or(0.0);
                let v2 = csr2.get(row, col).cloned().unwrap_or(0.0);
                assert!((v1 - v2).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_sell_to_dense() {
        let csr = make_test_csr();
        let sell = SellMatrix::from_csr(&csr, SliceSize::Custom(2));
        let dense = sell.to_dense();

        assert!((dense[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((dense[(0, 2)] - 2.0).abs() < 1e-10);
        assert!((dense[(1, 1)] - 3.0).abs() < 1e-10);
        assert!((dense[(2, 0)] - 4.0).abs() < 1e-10);
        assert!((dense[(2, 2)] - 5.0).abs() < 1e-10);
        assert!((dense[(2, 3)] - 6.0).abs() < 1e-10);
        assert!((dense[(3, 3)] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_sell_get() {
        let csr = make_test_csr();
        let sell = SellMatrix::from_csr(&csr, SliceSize::Custom(2));

        assert_eq!(sell.get(0, 0), Some(1.0));
        assert_eq!(sell.get(0, 2), Some(2.0));
        assert_eq!(sell.get(1, 1), Some(3.0));
        assert_eq!(sell.get(2, 0), Some(4.0));
        assert_eq!(sell.get(2, 2), Some(5.0));
        assert_eq!(sell.get(2, 3), Some(6.0));
        assert_eq!(sell.get(3, 3), Some(7.0));

        assert_eq!(sell.get(0, 1), None);
    }

    #[test]
    fn test_sell_eye() {
        let sell: SellMatrix<f64> = SellMatrix::eye(4, SliceSize::Custom(2));

        assert_eq!(sell.nrows(), 4);
        assert_eq!(sell.ncols(), 4);
        assert_eq!(sell.nnz(), 4);

        for i in 0..4 {
            assert_eq!(sell.get(i, i), Some(1.0));
        }
    }

    #[test]
    fn test_sell_scale() {
        let csr = make_test_csr();
        let mut sell = SellMatrix::from_csr(&csr, SliceSize::Custom(2));

        sell.scale(2.0);

        assert_eq!(sell.get(0, 0), Some(2.0));
        assert_eq!(sell.get(2, 2), Some(10.0));
    }

    #[test]
    fn test_sell_stats() {
        let csr = make_test_csr();
        let sell = SellMatrix::from_csr(&csr, SliceSize::Custom(2));
        let stats = sell.stats();

        assert_eq!(stats.nrows, 4);
        assert_eq!(stats.ncols, 4);
        assert_eq!(stats.slice_size, 2);
        assert_eq!(stats.num_slices, 2);
        assert_eq!(stats.nnz, 7);
    }

    #[test]
    fn test_sell_iter() {
        let csr = make_test_csr();
        let sell = SellMatrix::from_csr(&csr, SliceSize::Custom(2));

        let entries: Vec<_> = sell.iter().collect();
        assert_eq!(entries.len(), 7);
    }

    #[test]
    fn test_slice_size_conversions() {
        assert_eq!(SliceSize::Warp32.value(), 32);
        assert_eq!(SliceSize::Wavefront64.value(), 64);
        assert_eq!(SliceSize::Simd256.value(), 256);
        assert_eq!(SliceSize::Custom(128).value(), 128);

        assert_eq!(SliceSize::from(32), SliceSize::Warp32);
        assert_eq!(SliceSize::from(64), SliceSize::Wavefront64);
        assert_eq!(SliceSize::from(100), SliceSize::Custom(100));
    }

    #[test]
    fn test_sell_different_slice_sizes() {
        let csr = make_test_csr();

        // Test with various slice sizes
        for &slice_size in &[1, 2, 3, 4, 5] {
            let sell = SellMatrix::from_csr(&csr, SliceSize::Custom(slice_size));
            assert_eq!(sell.nnz(), 7);

            // Verify matvec is correct
            let x = vec![1.0, 2.0, 3.0, 4.0];
            let y = sell.mul_vec(&x);
            assert!((y[0] - 7.0).abs() < 1e-10);
            assert!((y[1] - 6.0).abs() < 1e-10);
            assert!((y[2] - 43.0).abs() < 1e-10);
            assert!((y[3] - 28.0).abs() < 1e-10);
        }
    }
}
