//! Hybrid ELL+COO (HYB) matrix format.
//!
//! HYB stores sparse matrices using a combination of:
//! - **ELL part**: For rows with up to K non-zeros (K is the ELL width)
//! - **COO part**: For overflow entries beyond K per row
//!
//! This hybrid approach provides:
//! - Efficient vectorized operations for uniform parts (ELL)
//! - Flexibility for rows with many entries (COO overflow)
//! - Better GPU performance than pure ELL for irregular matrices
//!
//! # When to Use HYB
//!
//! HYB format is optimal for:
//! - Matrices with mostly uniform row lengths but some outliers
//! - GPU computation where ELL alone wastes too much memory
//! - Power-law degree distributions (social networks, web graphs)
//!
//! # K Selection
//!
//! The ELL width K can be:
//! - Automatic: Based on mean + stddev of row lengths
//! - Manual: User-specified threshold
//! - Median-based: Use median row length

use crate::coo::CooMatrix;
use crate::csr::CsrMatrix;
use crate::ell::EllMatrix;
use oxiblas_core::scalar::{Field, Scalar};

/// Error type for HYB matrix operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HybError {
    /// Invalid dimensions.
    InvalidDimensions {
        /// Number of rows.
        nrows: usize,
        /// Number of columns.
        ncols: usize,
    },
    /// ELL width is zero.
    ZeroEllWidth,
    /// Incompatible ELL and COO dimensions.
    IncompatibleParts {
        /// ELL dimensions.
        ell_shape: (usize, usize),
        /// COO dimensions.
        coo_shape: (usize, usize),
    },
}

impl core::fmt::Display for HybError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidDimensions { nrows, ncols } => {
                write!(f, "Invalid dimensions: {nrows}×{ncols}")
            }
            Self::ZeroEllWidth => {
                write!(f, "ELL width cannot be zero")
            }
            Self::IncompatibleParts {
                ell_shape,
                coo_shape,
            } => {
                write!(
                    f,
                    "ELL shape {:?} incompatible with COO shape {:?}",
                    ell_shape, coo_shape
                )
            }
        }
    }
}

impl std::error::Error for HybError {}

/// Strategy for determining ELL width in HYB format.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HybWidthStrategy {
    /// Fixed ELL width.
    Fixed(usize),
    /// Use mean row length.
    Mean,
    /// Use mean + k*stddev row length.
    MeanPlusStddev(f64),
    /// Use median row length.
    Median,
    /// Use a specific percentile of row lengths (0.0 to 1.0).
    Percentile(f64),
    /// Maximum row length (equivalent to pure ELL).
    Max,
}

impl Default for HybWidthStrategy {
    fn default() -> Self {
        // Default: mean + 1 stddev catches ~84% of rows in ELL
        Self::MeanPlusStddev(1.0)
    }
}

/// Hybrid ELL+COO matrix format.
///
/// Combines ELLPACK for regular entries with COO for overflow.
///
/// # Storage
///
/// - ELL part: Fixed-width storage for up to K entries per row
/// - COO part: Overflow entries beyond K per row
///
/// # Example
///
/// ```
/// use oxiblas_sparse::{CsrMatrix, HybMatrix, HybWidthStrategy};
///
/// // Create a sparse matrix
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let col_indices = vec![0, 2, 1, 0, 2];
/// let row_ptrs = vec![0, 2, 3, 5];
/// let csr = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();
///
/// // Convert to HYB with automatic width selection
/// let hyb = HybMatrix::from_csr(&csr, HybWidthStrategy::default());
/// assert_eq!(hyb.shape(), (3, 3));
/// ```
#[derive(Debug, Clone)]
pub struct HybMatrix<T: Scalar> {
    /// Number of rows.
    nrows: usize,
    /// Number of columns.
    ncols: usize,
    /// ELL width (max entries per row in ELL part).
    ell_width: usize,
    /// ELL data: nrows × ell_width values (row-major).
    ell_data: Vec<T>,
    /// ELL column indices: nrows × ell_width indices.
    ell_indices: Vec<usize>,
    /// COO row indices for overflow.
    coo_rows: Vec<usize>,
    /// COO column indices for overflow.
    coo_cols: Vec<usize>,
    /// COO values for overflow.
    coo_data: Vec<T>,
}

impl<T: Scalar + Clone> HybMatrix<T> {
    /// Creates a new HYB matrix from raw components.
    ///
    /// # Arguments
    ///
    /// * `nrows` - Number of rows
    /// * `ncols` - Number of columns
    /// * `ell_width` - Maximum entries per row in ELL part
    /// * `ell_data` - ELL values (nrows × ell_width)
    /// * `ell_indices` - ELL column indices (nrows × ell_width)
    /// * `coo_rows` - COO row indices
    /// * `coo_cols` - COO column indices
    /// * `coo_data` - COO values
    ///
    /// # Errors
    ///
    /// Returns an error if the input is invalid.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        nrows: usize,
        ncols: usize,
        ell_width: usize,
        ell_data: Vec<T>,
        ell_indices: Vec<usize>,
        coo_rows: Vec<usize>,
        coo_cols: Vec<usize>,
        coo_data: Vec<T>,
    ) -> Result<Self, HybError> {
        if nrows == 0 || ncols == 0 {
            return Err(HybError::InvalidDimensions { nrows, ncols });
        }

        let expected_ell_size = nrows * ell_width;
        if ell_data.len() != expected_ell_size || ell_indices.len() != expected_ell_size {
            return Err(HybError::InvalidDimensions { nrows, ncols });
        }

        if coo_rows.len() != coo_cols.len() || coo_rows.len() != coo_data.len() {
            return Err(HybError::InvalidDimensions { nrows, ncols });
        }

        Ok(Self {
            nrows,
            ncols,
            ell_width,
            ell_data,
            ell_indices,
            coo_rows,
            coo_cols,
            coo_data,
        })
    }

    /// Creates an empty HYB matrix.
    pub fn zeros(nrows: usize, ncols: usize, ell_width: usize) -> Self
    where
        T: Field,
    {
        let size = nrows * ell_width;
        Self {
            nrows,
            ncols,
            ell_width,
            ell_data: vec![T::zero(); size],
            ell_indices: vec![0; size],
            coo_rows: Vec::new(),
            coo_cols: Vec::new(),
            coo_data: Vec::new(),
        }
    }

    /// Creates an identity matrix in HYB format.
    pub fn eye(n: usize) -> Self
    where
        T: Field,
    {
        let ell_width = 1;
        let mut ell_data = Vec::with_capacity(n);
        let mut ell_indices = Vec::with_capacity(n);

        for i in 0..n {
            ell_data.push(T::one());
            ell_indices.push(i);
        }

        Self {
            nrows: n,
            ncols: n,
            ell_width,
            ell_data,
            ell_indices,
            coo_rows: Vec::new(),
            coo_cols: Vec::new(),
            coo_data: Vec::new(),
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

    /// Returns the ELL width.
    #[inline]
    pub fn ell_width(&self) -> usize {
        self.ell_width
    }

    /// Returns the number of non-zeros in the ELL part.
    pub fn ell_nnz(&self) -> usize
    where
        T: Field,
    {
        let eps = <T as Scalar>::epsilon();
        self.ell_data
            .iter()
            .filter(|v| Scalar::abs((*v).clone()) > eps)
            .count()
    }

    /// Returns the number of non-zeros in the COO part.
    #[inline]
    pub fn coo_nnz(&self) -> usize {
        self.coo_data.len()
    }

    /// Returns the total number of non-zeros.
    pub fn nnz(&self) -> usize
    where
        T: Field,
    {
        self.ell_nnz() + self.coo_nnz()
    }

    /// Returns the ELL data.
    #[inline]
    pub fn ell_data(&self) -> &[T] {
        &self.ell_data
    }

    /// Returns the ELL column indices.
    #[inline]
    pub fn ell_indices(&self) -> &[usize] {
        &self.ell_indices
    }

    /// Returns the COO row indices.
    #[inline]
    pub fn coo_rows(&self) -> &[usize] {
        &self.coo_rows
    }

    /// Returns the COO column indices.
    #[inline]
    pub fn coo_cols(&self) -> &[usize] {
        &self.coo_cols
    }

    /// Returns the COO data.
    #[inline]
    pub fn coo_data(&self) -> &[T] {
        &self.coo_data
    }

    /// Returns the fraction of entries stored in ELL.
    pub fn ell_fraction(&self) -> f64
    where
        T: Field,
    {
        let total = self.nnz();
        if total == 0 {
            return 1.0;
        }
        self.ell_nnz() as f64 / total as f64
    }

    /// Returns the storage efficiency (actual nnz / stored values).
    pub fn storage_efficiency(&self) -> f64
    where
        T: Field,
    {
        let nnz = self.nnz();
        if nnz == 0 {
            return 0.0;
        }
        let stored = self.nrows * self.ell_width + self.coo_data.len();
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

        // Check ELL part
        let ell_start = row * self.ell_width;
        for k in 0..self.ell_width {
            let idx = ell_start + k;
            if self.ell_indices[idx] == col {
                let val = self.ell_data[idx].clone();
                if Scalar::abs(val.clone()) > eps {
                    return Some(val);
                }
            }
        }

        // Check COO part
        for (i, (&r, &c)) in self.coo_rows.iter().zip(self.coo_cols.iter()).enumerate() {
            if r == row && c == col {
                return Some(self.coo_data[i].clone());
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

        // ELL part
        for row in 0..self.nrows {
            let ell_start = row * self.ell_width;
            for k in 0..self.ell_width {
                let idx = ell_start + k;
                let val = &self.ell_data[idx];
                if Scalar::abs(val.clone()) > eps {
                    let col = self.ell_indices[idx];
                    y[row] = y[row].clone() + val.clone() * x[col].clone();
                }
            }
        }

        // COO part
        for (i, (&row, &col)) in self.coo_rows.iter().zip(self.coo_cols.iter()).enumerate() {
            y[row] = y[row].clone() + self.coo_data[i].clone() * x[col].clone();
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

        // ELL part
        for row in 0..self.nrows {
            let ell_start = row * self.ell_width;
            for k in 0..self.ell_width {
                let idx = ell_start + k;
                let val = &self.ell_data[idx];
                if Scalar::abs(val.clone()) > eps {
                    let col = self.ell_indices[idx];
                    y[col] = y[col].clone() + val.clone() * x[row].clone();
                }
            }
        }

        // COO part
        for (i, (&row, &col)) in self.coo_rows.iter().zip(self.coo_cols.iter()).enumerate() {
            y[col] = y[col].clone() + self.coo_data[i].clone() * x[row].clone();
        }
    }

    /// Creates HYB matrix from CSR with automatic width selection.
    pub fn from_csr(csr: &CsrMatrix<T>, strategy: HybWidthStrategy) -> Self
    where
        T: Field,
    {
        let (nrows, ncols) = csr.shape();
        let eps = <T as Scalar>::epsilon();

        // Compute row lengths (actual non-zeros)
        let mut row_lengths: Vec<usize> = Vec::with_capacity(nrows);
        for row in 0..nrows {
            let mut count = 0;
            for (_, val) in csr.row_iter(row) {
                if Scalar::abs(val.clone()) > eps {
                    count += 1;
                }
            }
            row_lengths.push(count);
        }

        // Determine ELL width based on strategy
        let ell_width = compute_ell_width(&row_lengths, strategy);
        let ell_width = ell_width.max(1); // At least 1

        // Build HYB from CSR
        let mut ell_data = vec![T::zero(); nrows * ell_width];
        let mut ell_indices = vec![0usize; nrows * ell_width];
        let mut coo_rows = Vec::new();
        let mut coo_cols = Vec::new();
        let mut coo_data = Vec::new();

        for row in 0..nrows {
            let ell_start = row * ell_width;
            let mut ell_count = 0;

            for (col, val) in csr.row_iter(row) {
                if Scalar::abs(val.clone()) <= eps {
                    continue;
                }

                if ell_count < ell_width {
                    ell_data[ell_start + ell_count] = val.clone();
                    ell_indices[ell_start + ell_count] = col;
                    ell_count += 1;
                } else {
                    // Overflow to COO
                    coo_rows.push(row);
                    coo_cols.push(col);
                    coo_data.push(val.clone());
                }
            }
        }

        Self {
            nrows,
            ncols,
            ell_width,
            ell_data,
            ell_indices,
            coo_rows,
            coo_cols,
            coo_data,
        }
    }

    /// Creates HYB matrix from COO format.
    pub fn from_coo(coo: &CooMatrix<T>, strategy: HybWidthStrategy) -> Self
    where
        T: Scalar<Real = T> + Field + oxiblas_core::Real,
    {
        let csr = coo.to_csr();
        Self::from_csr(&csr, strategy)
    }

    /// Creates HYB matrix from ELL format (no COO overflow).
    pub fn from_ell(ell: &EllMatrix<T>) -> Self
    where
        T: Field,
    {
        let (nrows, ncols) = ell.shape();
        let ell_width = ell.width();

        // Copy ELL data - convert from Vec<Vec<T>> to flat layout
        let mut ell_data = Vec::with_capacity(nrows * ell_width);
        let mut ell_indices = Vec::with_capacity(nrows * ell_width);

        let data = ell.data();
        let indices = ell.indices();

        for row in 0..nrows {
            for k in 0..ell_width {
                ell_data.push(data[row][k].clone());
                ell_indices.push(indices[row][k]);
            }
        }

        Self {
            nrows,
            ncols,
            ell_width,
            ell_data,
            ell_indices,
            coo_rows: Vec::new(),
            coo_cols: Vec::new(),
            coo_data: Vec::new(),
        }
    }

    /// Converts to CSR format.
    pub fn to_csr(&self) -> CsrMatrix<T>
    where
        T: Field,
    {
        let eps = <T as Scalar>::epsilon();

        // Collect all entries
        let mut entries: Vec<(usize, usize, T)> = Vec::new();

        // ELL entries
        for row in 0..self.nrows {
            let ell_start = row * self.ell_width;
            for k in 0..self.ell_width {
                let idx = ell_start + k;
                let val = self.ell_data[idx].clone();
                if Scalar::abs(val.clone()) > eps {
                    entries.push((row, self.ell_indices[idx], val));
                }
            }
        }

        // COO entries
        for (i, (&row, &col)) in self.coo_rows.iter().zip(self.coo_cols.iter()).enumerate() {
            entries.push((row, col, self.coo_data[i].clone()));
        }

        // Sort by row, then column
        entries.sort_by_key(|(r, c, _)| (*r, *c));

        // Build CSR
        let mut row_ptrs = vec![0usize; self.nrows + 1];
        let mut col_indices = Vec::with_capacity(entries.len());
        let mut values = Vec::with_capacity(entries.len());

        for (row, col, val) in entries {
            col_indices.push(col);
            values.push(val);
            row_ptrs[row + 1] += 1;
        }

        // Cumulative sum
        for i in 1..=self.nrows {
            row_ptrs[i] += row_ptrs[i - 1];
        }

        // Safety: we constructed valid CSR data
        unsafe { CsrMatrix::new_unchecked(self.nrows, self.ncols, row_ptrs, col_indices, values) }
    }

    /// Converts to COO format.
    pub fn to_coo(&self) -> CooMatrix<T>
    where
        T: Field,
    {
        let eps = <T as Scalar>::epsilon();

        let mut builder = crate::coo::CooMatrixBuilder::new(self.nrows, self.ncols);

        // ELL entries
        for row in 0..self.nrows {
            let ell_start = row * self.ell_width;
            for k in 0..self.ell_width {
                let idx = ell_start + k;
                let val = self.ell_data[idx].clone();
                if Scalar::abs(val.clone()) > eps {
                    builder.add(row, self.ell_indices[idx], val);
                }
            }
        }

        // COO entries
        for (i, (&row, &col)) in self.coo_rows.iter().zip(self.coo_cols.iter()).enumerate() {
            builder.add(row, col, self.coo_data[i].clone());
        }

        builder.build()
    }

    /// Converts to ELL format.
    ///
    /// Note: This may require increasing the ELL width to accommodate all entries.
    pub fn to_ell(&self) -> EllMatrix<T>
    where
        T: Field,
    {
        let eps = <T as Scalar>::epsilon();

        // Compute actual max row length including COO
        let mut row_lengths = vec![0usize; self.nrows];

        // Count ELL entries
        for row in 0..self.nrows {
            let ell_start = row * self.ell_width;
            for k in 0..self.ell_width {
                let idx = ell_start + k;
                if Scalar::abs(self.ell_data[idx].clone()) > eps {
                    row_lengths[row] += 1;
                }
            }
        }

        // Count COO entries
        for &row in &self.coo_rows {
            row_lengths[row] += 1;
        }

        let max_width = row_lengths.iter().max().copied().unwrap_or(0).max(1);

        // Build ELL with Vec<Vec<T>> format
        let mut data: Vec<Vec<T>> = vec![vec![T::zero(); max_width]; self.nrows];
        let mut indices: Vec<Vec<usize>> = vec![vec![0usize; max_width]; self.nrows];
        let mut current_counts = vec![0usize; self.nrows];

        // Add ELL entries
        for row in 0..self.nrows {
            let ell_start = row * self.ell_width;
            for k in 0..self.ell_width {
                let idx = ell_start + k;
                let val = self.ell_data[idx].clone();
                if Scalar::abs(val.clone()) > eps {
                    let pos = current_counts[row];
                    data[row][pos] = val;
                    indices[row][pos] = self.ell_indices[idx];
                    current_counts[row] += 1;
                }
            }
        }

        // Add COO entries
        for (i, (&row, &col)) in self.coo_rows.iter().zip(self.coo_cols.iter()).enumerate() {
            let pos = current_counts[row];
            data[row][pos] = self.coo_data[i].clone();
            indices[row][pos] = col;
            current_counts[row] += 1;
        }

        // Safety: we constructed valid ELL data
        unsafe { EllMatrix::new_unchecked(self.nrows, self.ncols, max_width, data, indices) }
    }

    /// Converts to dense matrix.
    pub fn to_dense(&self) -> oxiblas_matrix::Mat<T>
    where
        T: Field + bytemuck::Zeroable,
    {
        let eps = <T as Scalar>::epsilon();
        let mut dense = oxiblas_matrix::Mat::zeros(self.nrows, self.ncols);

        // ELL entries
        for row in 0..self.nrows {
            let ell_start = row * self.ell_width;
            for k in 0..self.ell_width {
                let idx = ell_start + k;
                let val = self.ell_data[idx].clone();
                if Scalar::abs(val.clone()) > eps {
                    let col = self.ell_indices[idx];
                    dense[(row, col)] = val;
                }
            }
        }

        // COO entries
        for (i, (&row, &col)) in self.coo_rows.iter().zip(self.coo_cols.iter()).enumerate() {
            dense[(row, col)] = self.coo_data[i].clone();
        }

        dense
    }

    /// Scales all values by a scalar.
    pub fn scale(&mut self, alpha: T) {
        for val in &mut self.ell_data {
            *val = val.clone() * alpha.clone();
        }
        for val in &mut self.coo_data {
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
        let ell_width = self.ell_width;

        // ELL entries
        let ell_iter = (0..self.nrows).flat_map(move |row| {
            let ell_start = row * ell_width;
            (0..ell_width).filter_map(move |k| {
                let idx = ell_start + k;
                let val = self.ell_data[idx].clone();
                if Scalar::abs(val.clone()) > eps {
                    Some((row, self.ell_indices[idx], val))
                } else {
                    None
                }
            })
        });

        // COO entries
        let coo_iter = self
            .coo_rows
            .iter()
            .zip(self.coo_cols.iter())
            .zip(self.coo_data.iter())
            .map(|((&row, &col), val)| (row, col, val.clone()));

        ell_iter.chain(coo_iter)
    }

    /// Rebalances the HYB matrix with a new ELL width.
    ///
    /// This redistributes entries between ELL and COO based on the new width.
    pub fn rebalance(&mut self, new_width: usize)
    where
        T: Field,
    {
        if new_width == self.ell_width {
            return;
        }

        let eps = <T as Scalar>::epsilon();

        // Collect all entries
        let mut entries: Vec<(usize, usize, T)> = Vec::new();

        // ELL entries
        for row in 0..self.nrows {
            let ell_start = row * self.ell_width;
            for k in 0..self.ell_width {
                let idx = ell_start + k;
                let val = self.ell_data[idx].clone();
                if Scalar::abs(val.clone()) > eps {
                    entries.push((row, self.ell_indices[idx], val));
                }
            }
        }

        // COO entries
        for (i, (&row, &col)) in self.coo_rows.iter().zip(self.coo_cols.iter()).enumerate() {
            entries.push((row, col, self.coo_data[i].clone()));
        }

        // Sort by row, then column
        entries.sort_by_key(|(r, c, _)| (*r, *c));

        // Rebuild with new width
        self.ell_width = new_width;
        self.ell_data = vec![T::zero(); self.nrows * new_width];
        self.ell_indices = vec![0usize; self.nrows * new_width];
        self.coo_rows.clear();
        self.coo_cols.clear();
        self.coo_data.clear();

        let mut current_row = 0;
        let mut count_in_row = 0;

        for (row, col, val) in entries {
            if row != current_row {
                current_row = row;
                count_in_row = 0;
            }

            if count_in_row < new_width {
                let idx = row * new_width + count_in_row;
                self.ell_data[idx] = val;
                self.ell_indices[idx] = col;
                count_in_row += 1;
            } else {
                self.coo_rows.push(row);
                self.coo_cols.push(col);
                self.coo_data.push(val);
            }
        }
    }

    /// Returns statistics about the HYB matrix.
    pub fn stats(&self) -> HybStats
    where
        T: Field,
    {
        let eps = <T as Scalar>::epsilon();

        // Compute actual row lengths
        let mut row_lengths = vec![0usize; self.nrows];

        for row in 0..self.nrows {
            let ell_start = row * self.ell_width;
            for k in 0..self.ell_width {
                let idx = ell_start + k;
                if Scalar::abs(self.ell_data[idx].clone()) > eps {
                    row_lengths[row] += 1;
                }
            }
        }

        for &row in &self.coo_rows {
            row_lengths[row] += 1;
        }

        let ell_nnz = self.ell_nnz();
        let coo_nnz = self.coo_nnz();
        let total_nnz = ell_nnz + coo_nnz;

        let max_row_len = row_lengths.iter().max().copied().unwrap_or(0);
        let min_row_len = row_lengths.iter().min().copied().unwrap_or(0);
        let avg_row_len = if self.nrows > 0 {
            total_nnz as f64 / self.nrows as f64
        } else {
            0.0
        };

        HybStats {
            nrows: self.nrows,
            ncols: self.ncols,
            ell_width: self.ell_width,
            ell_nnz,
            coo_nnz,
            total_nnz,
            ell_fraction: if total_nnz > 0 {
                ell_nnz as f64 / total_nnz as f64
            } else {
                1.0
            },
            max_row_length: max_row_len,
            min_row_length: min_row_len,
            avg_row_length: avg_row_len,
            storage_efficiency: self.storage_efficiency(),
        }
    }
}

/// Statistics about a HYB matrix.
#[derive(Debug, Clone, Copy)]
pub struct HybStats {
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// ELL width.
    pub ell_width: usize,
    /// Number of non-zeros in ELL part.
    pub ell_nnz: usize,
    /// Number of non-zeros in COO part.
    pub coo_nnz: usize,
    /// Total non-zeros.
    pub total_nnz: usize,
    /// Fraction of entries in ELL (0.0 to 1.0).
    pub ell_fraction: f64,
    /// Maximum row length.
    pub max_row_length: usize,
    /// Minimum row length.
    pub min_row_length: usize,
    /// Average row length.
    pub avg_row_length: f64,
    /// Storage efficiency (nnz / stored).
    pub storage_efficiency: f64,
}

/// Computes ELL width based on row length statistics.
fn compute_ell_width(row_lengths: &[usize], strategy: HybWidthStrategy) -> usize {
    if row_lengths.is_empty() {
        return 1;
    }

    match strategy {
        HybWidthStrategy::Fixed(k) => k,

        HybWidthStrategy::Mean => {
            let sum: usize = row_lengths.iter().sum();
            let mean = sum as f64 / row_lengths.len() as f64;
            mean.ceil() as usize
        }

        HybWidthStrategy::MeanPlusStddev(k) => {
            let n = row_lengths.len() as f64;
            let sum: usize = row_lengths.iter().sum();
            let mean = sum as f64 / n;

            let variance: f64 = row_lengths
                .iter()
                .map(|&x| {
                    let diff = x as f64 - mean;
                    diff * diff
                })
                .sum::<f64>()
                / n;
            let stddev = variance.sqrt();

            (mean + k * stddev).ceil() as usize
        }

        HybWidthStrategy::Median => {
            let mut sorted = row_lengths.to_vec();
            sorted.sort_unstable();
            let mid = sorted.len() / 2;
            if sorted.len() % 2 == 0 {
                (sorted[mid - 1] + sorted[mid]).div_ceil(2)
            } else {
                sorted[mid]
            }
        }

        HybWidthStrategy::Percentile(p) => {
            let p = p.clamp(0.0, 1.0);
            let mut sorted = row_lengths.to_vec();
            sorted.sort_unstable();
            let idx = ((sorted.len() - 1) as f64 * p) as usize;
            sorted[idx]
        }

        HybWidthStrategy::Max => row_lengths.iter().max().copied().unwrap_or(1),
    }
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
    fn test_hyb_from_csr_fixed() {
        let csr = make_test_csr();
        let hyb = HybMatrix::from_csr(&csr, HybWidthStrategy::Fixed(2));

        assert_eq!(hyb.nrows(), 4);
        assert_eq!(hyb.ncols(), 4);
        assert_eq!(hyb.ell_width(), 2);

        // Row 2 has 3 entries, so 1 should overflow to COO
        assert_eq!(hyb.coo_nnz(), 1);
        assert_eq!(hyb.nnz(), 7);
    }

    #[test]
    fn test_hyb_from_csr_max() {
        let csr = make_test_csr();
        let hyb = HybMatrix::from_csr(&csr, HybWidthStrategy::Max);

        assert_eq!(hyb.ell_width(), 3); // Max row has 3 entries
        assert_eq!(hyb.coo_nnz(), 0); // No overflow
    }

    #[test]
    fn test_hyb_matvec() {
        let csr = make_test_csr();
        let hyb = HybMatrix::from_csr(&csr, HybWidthStrategy::Fixed(2));

        // [1 0 2 0]   [1]   [7]
        // [0 3 0 0] * [2] = [6]
        // [4 0 5 6]   [3]   [43]
        // [0 0 0 7]   [4]   [28]
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = hyb.mul_vec(&x);

        assert!((y[0] - 7.0).abs() < 1e-10);
        assert!((y[1] - 6.0).abs() < 1e-10);
        assert!((y[2] - 43.0).abs() < 1e-10);
        assert!((y[3] - 28.0).abs() < 1e-10);
    }

    #[test]
    fn test_hyb_matvec_transpose() {
        let csr = make_test_csr();
        let hyb = HybMatrix::from_csr(&csr, HybWidthStrategy::Fixed(2));

        let x = vec![1.0, 1.0, 1.0, 1.0];
        let mut y = vec![0.0; 4];
        hyb.matvec_transpose(&x, &mut y);

        // A^T * [1,1,1,1] = column sums
        assert!((y[0] - 5.0).abs() < 1e-10); // 1 + 4
        assert!((y[1] - 3.0).abs() < 1e-10); // 3
        assert!((y[2] - 7.0).abs() < 1e-10); // 2 + 5
        assert!((y[3] - 13.0).abs() < 1e-10); // 6 + 7
    }

    #[test]
    fn test_hyb_to_csr_roundtrip() {
        let csr1 = make_test_csr();
        let hyb = HybMatrix::from_csr(&csr1, HybWidthStrategy::Fixed(2));
        let csr2 = hyb.to_csr();

        assert_eq!(csr1.nnz(), csr2.nnz());

        // Check all values match
        for row in 0..4 {
            for col in 0..4 {
                let v1 = csr1.get(row, col).cloned().unwrap_or(0.0);
                let v2 = csr2.get(row, col).cloned().unwrap_or(0.0);
                assert!((v1 - v2).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_hyb_to_dense() {
        let csr = make_test_csr();
        let hyb = HybMatrix::from_csr(&csr, HybWidthStrategy::Fixed(2));
        let dense = hyb.to_dense();

        assert!((dense[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((dense[(0, 2)] - 2.0).abs() < 1e-10);
        assert!((dense[(1, 1)] - 3.0).abs() < 1e-10);
        assert!((dense[(2, 0)] - 4.0).abs() < 1e-10);
        assert!((dense[(2, 2)] - 5.0).abs() < 1e-10);
        assert!((dense[(2, 3)] - 6.0).abs() < 1e-10);
        assert!((dense[(3, 3)] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_hyb_get() {
        let csr = make_test_csr();
        let hyb = HybMatrix::from_csr(&csr, HybWidthStrategy::Fixed(2));

        assert_eq!(hyb.get(0, 0), Some(1.0));
        assert_eq!(hyb.get(0, 2), Some(2.0));
        assert_eq!(hyb.get(1, 1), Some(3.0));
        assert_eq!(hyb.get(2, 0), Some(4.0));
        assert_eq!(hyb.get(2, 2), Some(5.0));
        assert_eq!(hyb.get(2, 3), Some(6.0)); // This might be in COO
        assert_eq!(hyb.get(3, 3), Some(7.0));

        assert_eq!(hyb.get(0, 1), None);
    }

    #[test]
    fn test_hyb_eye() {
        let hyb: HybMatrix<f64> = HybMatrix::eye(4);

        assert_eq!(hyb.nrows(), 4);
        assert_eq!(hyb.ncols(), 4);
        assert_eq!(hyb.ell_width(), 1);
        assert_eq!(hyb.nnz(), 4);

        for i in 0..4 {
            assert_eq!(hyb.get(i, i), Some(1.0));
        }
    }

    #[test]
    fn test_hyb_scale() {
        let csr = make_test_csr();
        let mut hyb = HybMatrix::from_csr(&csr, HybWidthStrategy::Fixed(2));

        hyb.scale(2.0);

        assert_eq!(hyb.get(0, 0), Some(2.0));
        assert_eq!(hyb.get(2, 2), Some(10.0));
    }

    #[test]
    fn test_hyb_rebalance() {
        let csr = make_test_csr();
        let mut hyb = HybMatrix::from_csr(&csr, HybWidthStrategy::Fixed(1));

        // Initially most entries in COO
        assert!(hyb.coo_nnz() > 0);

        // Rebalance to width 3
        hyb.rebalance(3);

        assert_eq!(hyb.ell_width(), 3);
        assert_eq!(hyb.coo_nnz(), 0); // All fit in ELL now
        assert_eq!(hyb.nnz(), 7);
    }

    #[test]
    fn test_hyb_stats() {
        let csr = make_test_csr();
        let hyb = HybMatrix::from_csr(&csr, HybWidthStrategy::Fixed(2));
        let stats = hyb.stats();

        assert_eq!(stats.nrows, 4);
        assert_eq!(stats.ncols, 4);
        assert_eq!(stats.ell_width, 2);
        assert_eq!(stats.total_nnz, 7);
        assert_eq!(stats.max_row_length, 3);
        assert_eq!(stats.min_row_length, 1);
    }

    #[test]
    fn test_hyb_iter() {
        let csr = make_test_csr();
        let hyb = HybMatrix::from_csr(&csr, HybWidthStrategy::Fixed(2));

        let entries: Vec<_> = hyb.iter().collect();
        assert_eq!(entries.len(), 7);
    }

    #[test]
    fn test_compute_ell_width() {
        let row_lengths = vec![1, 2, 5, 3, 2];

        assert_eq!(
            compute_ell_width(&row_lengths, HybWidthStrategy::Fixed(4)),
            4
        );
        assert_eq!(compute_ell_width(&row_lengths, HybWidthStrategy::Max), 5);
        assert_eq!(compute_ell_width(&row_lengths, HybWidthStrategy::Median), 2);

        // Mean = 2.6, so ceil = 3
        assert_eq!(compute_ell_width(&row_lengths, HybWidthStrategy::Mean), 3);
    }

    #[test]
    fn test_hyb_from_ell() {
        // Create a simple ELL from CSR
        let csr = make_test_csr();
        let ell = crate::ell::EllMatrix::from_csr(&csr, None).unwrap();

        let hyb = HybMatrix::from_ell(&ell);

        assert_eq!(hyb.coo_nnz(), 0);
        assert_eq!(hyb.nnz(), 7);
    }
}
