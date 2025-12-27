//! Block Sparse Column (BSC) matrix format.
//!
//! BSC stores matrix data as dense blocks, using:
//! - `data`: Array of dense r×c blocks
//! - `indices`: Block row indices (like CSC but for block rows)
//! - `indptr`: Block column pointers (like CSC but for block columns)
//!
//! For an m×n matrix with r×c blocks:
//! - Number of block rows: mb = ceil(m/r)
//! - Number of block columns: nb = ceil(n/c)
//!
//! # When to Use BSC
//!
//! BSC format is optimal for:
//! - Column-oriented block-structured problems
//! - When column access is more frequent than row access
//! - Block-structured direct solvers
//! - Vectorized block operations on columns
//!
//! BSC is NOT efficient for:
//! - Matrices without block structure
//! - Row-oriented operations
//! - Very small matrices

use crate::bsr::{BsrMatrix, DenseBlock};
use oxiblas_core::scalar::{Field, Scalar};

/// Error type for BSC matrix operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BscError {
    /// Invalid block dimensions.
    InvalidBlockSize {
        /// Block rows.
        block_rows: usize,
        /// Block columns.
        block_cols: usize,
    },
    /// Matrix dimensions not compatible with block size.
    IncompatibleDimensions {
        /// Matrix rows.
        nrows: usize,
        /// Matrix columns.
        ncols: usize,
        /// Block rows.
        block_rows: usize,
        /// Block columns.
        block_cols: usize,
    },
    /// Invalid indptr array length.
    InvalidIndptr {
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },
    /// Mismatched data/indices counts.
    DataIndicesMismatch {
        /// Number of blocks in data.
        num_blocks: usize,
        /// Number of indices.
        num_indices: usize,
    },
    /// Block row index out of bounds.
    InvalidBlockIndex {
        /// The invalid index.
        index: usize,
        /// Number of block rows.
        mb_rows: usize,
    },
    /// Indptr not monotonically increasing.
    InvalidIndptrOrder,
    /// Block data has wrong size.
    InvalidBlockData {
        /// Block index.
        block_idx: usize,
        /// Expected size.
        expected: usize,
        /// Actual size.
        actual: usize,
    },
}

impl core::fmt::Display for BscError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidBlockSize {
                block_rows,
                block_cols,
            } => {
                write!(f, "Invalid block size: {block_rows}×{block_cols}")
            }
            Self::IncompatibleDimensions {
                nrows,
                ncols,
                block_rows,
                block_cols,
            } => {
                write!(
                    f,
                    "Matrix {nrows}×{ncols} incompatible with {block_rows}×{block_cols} blocks"
                )
            }
            Self::InvalidIndptr { expected, actual } => {
                write!(
                    f,
                    "Invalid indptr length: expected {expected}, got {actual}"
                )
            }
            Self::DataIndicesMismatch {
                num_blocks,
                num_indices,
            } => {
                write!(f, "Mismatch: {num_blocks} blocks but {num_indices} indices")
            }
            Self::InvalidBlockIndex { index, mb_rows } => {
                write!(
                    f,
                    "Block row index {index} out of bounds (mb_rows={mb_rows})"
                )
            }
            Self::InvalidIndptrOrder => {
                write!(f, "Indptr must be monotonically increasing")
            }
            Self::InvalidBlockData {
                block_idx,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Block {block_idx}: expected {expected} elements, got {actual}"
                )
            }
        }
    }
}

impl std::error::Error for BscError {}

/// Block Sparse Column matrix format.
///
/// Efficient for:
/// - Column-oriented block-structured problems
/// - Block-structured direct solvers
/// - Column access patterns
///
/// # Storage
///
/// Stores sparse matrices as a collection of dense blocks arranged in CSC-like
/// structure. Each block is r×c dense matrix.
///
/// # Example
///
/// ```
/// use oxiblas_sparse::{BscMatrix, DenseBlock};
///
/// // 4×4 matrix with 2×2 blocks (column-oriented):
/// // [1 2 | 0 0]
/// // [3 4 | 0 0]
/// // [----+----]
/// // [0 0 | 5 6]
/// // [0 0 | 7 8]
/// let block1 = DenseBlock::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
/// let block2 = DenseBlock::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
///
/// let bsc = BscMatrix::new(
///     4, 4,           // matrix dimensions
///     2, 2,           // block dimensions
///     vec![0, 1, 2],  // indptr (2 block columns)
///     vec![0, 1],     // indices (block row indices)
///     vec![block1, block2],
/// ).unwrap();
///
/// assert_eq!(bsc.nblocks(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct BscMatrix<T: Scalar> {
    /// Number of matrix rows.
    nrows: usize,
    /// Number of matrix columns.
    ncols: usize,
    /// Block row size.
    block_rows: usize,
    /// Block column size.
    block_cols: usize,
    /// Number of block rows.
    mb: usize,
    /// Number of block columns.
    nb: usize,
    /// Column pointers for blocks.
    indptr: Vec<usize>,
    /// Block row indices.
    indices: Vec<usize>,
    /// Dense blocks.
    data: Vec<DenseBlock<T>>,
}

impl<T: Scalar + Clone> BscMatrix<T> {
    /// Creates a new BSC matrix from raw components.
    ///
    /// # Arguments
    ///
    /// * `nrows` - Number of matrix rows
    /// * `ncols` - Number of matrix columns
    /// * `block_rows` - Number of rows per block
    /// * `block_cols` - Number of columns per block
    /// * `indptr` - Block column pointers (length nb + 1)
    /// * `indices` - Block row indices
    /// * `data` - Dense blocks
    ///
    /// # Errors
    ///
    /// Returns an error if the input is invalid.
    pub fn new(
        nrows: usize,
        ncols: usize,
        block_rows: usize,
        block_cols: usize,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<DenseBlock<T>>,
    ) -> Result<Self, BscError> {
        // Validate block size
        if block_rows == 0 || block_cols == 0 {
            return Err(BscError::InvalidBlockSize {
                block_rows,
                block_cols,
            });
        }

        // Calculate number of block rows/cols
        let mb = nrows.div_ceil(block_rows);
        let nb = ncols.div_ceil(block_cols);

        // Validate indptr length (nb + 1 for column pointers)
        if indptr.len() != nb + 1 {
            return Err(BscError::InvalidIndptr {
                expected: nb + 1,
                actual: indptr.len(),
            });
        }

        // Validate indptr is monotonically increasing
        for i in 1..indptr.len() {
            if indptr[i] < indptr[i - 1] {
                return Err(BscError::InvalidIndptrOrder);
            }
        }

        // Validate data and indices count
        let nnz_blocks = data.len();
        if indices.len() != nnz_blocks {
            return Err(BscError::DataIndicesMismatch {
                num_blocks: nnz_blocks,
                num_indices: indices.len(),
            });
        }

        // Validate indptr[nb] equals nnz_blocks
        if indptr[nb] != nnz_blocks {
            return Err(BscError::InvalidIndptr {
                expected: nnz_blocks,
                actual: indptr[nb],
            });
        }

        // Validate block row indices
        for &idx in &indices {
            if idx >= mb {
                return Err(BscError::InvalidBlockIndex {
                    index: idx,
                    mb_rows: mb,
                });
            }
        }

        // Validate block sizes
        let block_size = block_rows * block_cols;
        for (i, block) in data.iter().enumerate() {
            if block.data().len() != block_size {
                return Err(BscError::InvalidBlockData {
                    block_idx: i,
                    expected: block_size,
                    actual: block.data().len(),
                });
            }
        }

        Ok(Self {
            nrows,
            ncols,
            block_rows,
            block_cols,
            mb,
            nb,
            indptr,
            indices,
            data,
        })
    }

    /// Creates a BSC matrix without validation (unsafe but faster).
    ///
    /// # Safety
    ///
    /// The caller must ensure all invariants hold.
    #[inline]
    pub unsafe fn new_unchecked(
        nrows: usize,
        ncols: usize,
        block_rows: usize,
        block_cols: usize,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<DenseBlock<T>>,
    ) -> Self {
        let mb = nrows.div_ceil(block_rows);
        let nb = ncols.div_ceil(block_cols);
        Self {
            nrows,
            ncols,
            block_rows,
            block_cols,
            mb,
            nb,
            indptr,
            indices,
            data,
        }
    }

    /// Creates an empty BSC matrix with given dimensions.
    pub fn zeros(nrows: usize, ncols: usize, block_rows: usize, block_cols: usize) -> Self {
        let nb = ncols.div_ceil(block_cols);
        Self {
            nrows,
            ncols,
            block_rows,
            block_cols,
            mb: nrows.div_ceil(block_rows),
            nb,
            indptr: vec![0; nb + 1],
            indices: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Creates an identity matrix in BSC format.
    pub fn eye(n: usize, block_size: usize) -> Self
    where
        T: Field,
    {
        let nb = n.div_ceil(block_size);
        let mut indptr = Vec::with_capacity(nb + 1);
        let mut indices = Vec::with_capacity(nb);
        let mut data = Vec::with_capacity(nb);

        indptr.push(0);

        for bj in 0..nb {
            indices.push(bj); // Diagonal block: block row = block col

            // Create identity block
            let mut block_data = vec![T::zero(); block_size * block_size];
            for i in 0..block_size {
                let global_idx = bj * block_size + i;
                if global_idx < n {
                    block_data[i * block_size + i] = T::one();
                }
            }
            data.push(DenseBlock::new(block_size, block_size, block_data));

            indptr.push(data.len());
        }

        Self {
            nrows: n,
            ncols: n,
            block_rows: block_size,
            block_cols: block_size,
            mb: nb,
            nb,
            indptr,
            indices,
            data,
        }
    }

    /// Returns the number of matrix rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Returns the number of matrix columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Returns the shape (nrows, ncols).
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Returns the block dimensions (block_rows, block_cols).
    #[inline]
    pub fn block_shape(&self) -> (usize, usize) {
        (self.block_rows, self.block_cols)
    }

    /// Returns the number of block rows.
    #[inline]
    pub fn nblock_rows(&self) -> usize {
        self.mb
    }

    /// Returns the number of block columns.
    #[inline]
    pub fn nblock_cols(&self) -> usize {
        self.nb
    }

    /// Returns the number of non-zero blocks.
    #[inline]
    pub fn nblocks(&self) -> usize {
        self.data.len()
    }

    /// Returns the number of non-zero scalar elements.
    ///
    /// Note: This counts actual non-zeros within blocks, not stored values.
    pub fn nnz(&self) -> usize
    where
        T: Field,
    {
        let eps = <T as Scalar>::epsilon();
        let mut count = 0;

        for block in &self.data {
            for val in block.data() {
                if Scalar::abs(val.clone()) > eps {
                    count += 1;
                }
            }
        }

        count
    }

    /// Returns the total stored values.
    #[inline]
    pub fn nstored(&self) -> usize {
        self.data.len() * self.block_rows * self.block_cols
    }

    /// Returns the block column pointers.
    #[inline]
    pub fn indptr(&self) -> &[usize] {
        &self.indptr
    }

    /// Returns the block row indices.
    #[inline]
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Returns the block data.
    #[inline]
    pub fn data(&self) -> &[DenseBlock<T>] {
        &self.data
    }

    /// Returns mutable block data.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [DenseBlock<T>] {
        &mut self.data
    }

    /// Gets the block at block position (bi, bj), if present.
    pub fn get_block(&self, bi: usize, bj: usize) -> Option<&DenseBlock<T>> {
        if bi >= self.mb || bj >= self.nb {
            return None;
        }

        let start = self.indptr[bj];
        let end = self.indptr[bj + 1];

        for k in start..end {
            if self.indices[k] == bi {
                return Some(&self.data[k]);
            }
        }

        None
    }

    /// Gets the scalar value at (row, col).
    pub fn get(&self, row: usize, col: usize) -> Option<T>
    where
        T: Field,
    {
        if row >= self.nrows || col >= self.ncols {
            return None;
        }

        let bi = row / self.block_rows;
        let bj = col / self.block_cols;
        let local_i = row % self.block_rows;
        let local_j = col % self.block_cols;

        self.get_block(bi, bj)
            .map(|block| block.get(local_i, local_j).clone())
    }

    /// Gets the scalar value at (row, col), returning zero if not present.
    pub fn get_or_zero(&self, row: usize, col: usize) -> T
    where
        T: Field,
    {
        self.get(row, col).unwrap_or_else(T::zero)
    }

    /// Returns an iterator over non-zero blocks as (block_row, block_col, &block).
    pub fn block_iter(&self) -> impl Iterator<Item = (usize, usize, &DenseBlock<T>)> + '_ {
        (0..self.nb).flat_map(move |bj| {
            let start = self.indptr[bj];
            let end = self.indptr[bj + 1];

            (start..end).map(move |k| (self.indices[k], bj, &self.data[k]))
        })
    }

    /// Returns an iterator over blocks in column bj.
    pub fn col_block_iter(&self, bj: usize) -> impl Iterator<Item = (usize, &DenseBlock<T>)> + '_ {
        let start = if bj < self.nb { self.indptr[bj] } else { 0 };
        let end = if bj < self.nb { self.indptr[bj + 1] } else { 0 };

        (start..end).map(move |k| (self.indices[k], &self.data[k]))
    }

    /// Returns an iterator over all non-zero scalars as (row, col, value).
    pub fn iter(&self) -> impl Iterator<Item = (usize, usize, T)> + '_
    where
        T: Field,
    {
        let eps = <T as Scalar>::epsilon();
        let br = self.block_rows;
        let bc = self.block_cols;
        let nrows = self.nrows;
        let ncols = self.ncols;

        self.block_iter().flat_map(move |(bi, bj, block)| {
            let base_row = bi * br;
            let base_col = bj * bc;

            (0..br).flat_map(move |i| {
                (0..bc).filter_map(move |j| {
                    let global_row = base_row + i;
                    let global_col = base_col + j;

                    if global_row < nrows && global_col < ncols {
                        let val = block.get(i, j).clone();
                        if Scalar::abs(val.clone()) > eps {
                            return Some((global_row, global_col, val));
                        }
                    }
                    None
                })
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

        // Process each block column
        for bj in 0..self.nb {
            let start = self.indptr[bj];
            let end = self.indptr[bj + 1];
            let col_start = bj * self.block_cols;
            let col_end = (col_start + self.block_cols).min(self.ncols);

            for k in start..end {
                let bi = self.indices[k];
                let block = &self.data[k];
                let row_start = bi * self.block_rows;
                let row_end = (row_start + self.block_rows).min(self.nrows);

                // Block matrix-vector product
                for (i, yi) in y[row_start..row_end].iter_mut().enumerate() {
                    for j in 0..(col_end - col_start) {
                        *yi = yi.clone() + block.get(i, j).clone() * x[col_start + j].clone();
                    }
                }
            }
        }
    }

    /// Transposed matrix-vector product: y = A^T * x.
    pub fn matvec_transpose(&self, x: &[T], y: &mut [T])
    where
        T: Field,
    {
        assert_eq!(x.len(), self.nrows, "x length must equal nrows");
        assert_eq!(y.len(), self.ncols, "y length must equal ncols");

        // Initialize y to zero
        for yi in y.iter_mut() {
            *yi = T::zero();
        }

        // Process each block column
        for bj in 0..self.nb {
            let start = self.indptr[bj];
            let end = self.indptr[bj + 1];
            let col_start = bj * self.block_cols;
            let col_end = (col_start + self.block_cols).min(self.ncols);

            for k in start..end {
                let bi = self.indices[k];
                let block = &self.data[k];
                let row_start = bi * self.block_rows;
                let row_end = (row_start + self.block_rows).min(self.nrows);

                // Block transpose matrix-vector product: y_col += block^T * x_row
                for j in 0..(col_end - col_start) {
                    for i in 0..(row_end - row_start) {
                        y[col_start + j] = y[col_start + j].clone()
                            + block.get(i, j).clone() * x[row_start + i].clone();
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

    /// Converts to BSR format.
    pub fn to_bsr(&self) -> BsrMatrix<T>
    where
        T: Field,
    {
        let mut block_entries: Vec<(usize, usize, DenseBlock<T>)> =
            Vec::with_capacity(self.nblocks());

        // Collect all blocks with their positions
        for bj in 0..self.nb {
            let start = self.indptr[bj];
            let end = self.indptr[bj + 1];
            for k in start..end {
                let bi = self.indices[k];
                block_entries.push((bi, bj, self.data[k].clone()));
            }
        }

        // Sort by block row, then block column
        block_entries.sort_by_key(|(bi, bj, _)| (*bi, *bj));

        // Build BSR structure
        let mut indptr = Vec::with_capacity(self.mb + 1);
        let mut indices = Vec::with_capacity(block_entries.len());
        let mut data = Vec::with_capacity(block_entries.len());

        indptr.push(0);

        let mut entry_idx = 0;
        for bi in 0..self.mb {
            while entry_idx < block_entries.len() && block_entries[entry_idx].0 == bi {
                let (_, bj, block) = &block_entries[entry_idx];
                indices.push(*bj);
                data.push(block.clone());
                entry_idx += 1;
            }
            indptr.push(data.len());
        }

        // Safety: we constructed valid BSR data
        unsafe {
            BsrMatrix::new_unchecked(
                self.nrows,
                self.ncols,
                self.block_rows,
                self.block_cols,
                indptr,
                indices,
                data,
            )
        }
    }

    /// Creates a BSC matrix from BSR format.
    pub fn from_bsr(bsr: &BsrMatrix<T>) -> Self
    where
        T: Field,
    {
        let (nrows, ncols) = bsr.shape();
        let (block_rows, block_cols) = bsr.block_shape();
        let mb = bsr.nblock_rows();
        let nb = bsr.nblock_cols();

        // Collect all blocks with their positions
        let mut block_entries: Vec<(usize, usize, DenseBlock<T>)> =
            Vec::with_capacity(bsr.nblocks());
        for (bi, bj, block) in bsr.block_iter() {
            block_entries.push((bi, bj, block.clone()));
        }

        // Sort by block column, then block row
        block_entries.sort_by_key(|(bi, bj, _)| (*bj, *bi));

        // Build BSC structure
        let mut indptr = Vec::with_capacity(nb + 1);
        let mut indices = Vec::with_capacity(block_entries.len());
        let mut data = Vec::with_capacity(block_entries.len());

        indptr.push(0);

        let mut entry_idx = 0;
        for bj in 0..nb {
            while entry_idx < block_entries.len() && block_entries[entry_idx].1 == bj {
                let (bi, _, block) = &block_entries[entry_idx];
                indices.push(*bi);
                data.push(block.clone());
                entry_idx += 1;
            }
            indptr.push(data.len());
        }

        Self {
            nrows,
            ncols,
            block_rows,
            block_cols,
            mb,
            nb,
            indptr,
            indices,
            data,
        }
    }

    /// Converts to CSC format.
    pub fn to_csc(&self) -> crate::csc::CscMatrix<T>
    where
        T: Field,
    {
        let eps = <T as Scalar>::epsilon();

        let mut col_ptrs = vec![0usize; self.ncols + 1];
        let mut row_indices = Vec::new();
        let mut values = Vec::new();

        for col in 0..self.ncols {
            let bj = col / self.block_cols;
            let local_j = col % self.block_cols;

            let block_start = self.indptr[bj];
            let block_end = self.indptr[bj + 1];

            let mut col_entries: Vec<(usize, T)> = Vec::new();

            for k in block_start..block_end {
                let bi = self.indices[k];
                let block = &self.data[k];

                for i in 0..self.block_rows {
                    let global_row = bi * self.block_rows + i;
                    if global_row < self.nrows {
                        let val = block.get(i, local_j).clone();
                        if Scalar::abs(val.clone()) > eps {
                            col_entries.push((global_row, val));
                        }
                    }
                }
            }

            // Sort by row
            col_entries.sort_by_key(|(row, _)| *row);

            for (row, val) in col_entries {
                row_indices.push(row);
                values.push(val);
            }
            col_ptrs[col + 1] = values.len();
        }

        // Safety: we constructed valid CSC data
        unsafe {
            crate::csc::CscMatrix::new_unchecked(
                self.nrows,
                self.ncols,
                col_ptrs,
                row_indices,
                values,
            )
        }
    }

    /// Creates a BSC matrix from CSC format.
    pub fn from_csc(csc: &crate::csc::CscMatrix<T>, block_rows: usize, block_cols: usize) -> Self
    where
        T: Field,
    {
        let (nrows, ncols) = csc.shape();
        let eps = <T as Scalar>::epsilon();

        let mb = nrows.div_ceil(block_rows);
        let nb = ncols.div_ceil(block_cols);

        let mut indptr = Vec::with_capacity(nb + 1);
        let mut indices = Vec::new();
        let mut data = Vec::new();

        indptr.push(0);

        for bj in 0..nb {
            let col_start = bj * block_cols;
            let col_end = (col_start + block_cols).min(ncols);

            // Collect all block rows that have entries in this block column
            let mut block_rows_present = std::collections::HashSet::new();
            for col in col_start..col_end {
                for (row, _) in csc.col_iter(col) {
                    let bi = row / block_rows;
                    block_rows_present.insert(bi);
                }
            }

            // Sort block rows
            let mut sorted_bis: Vec<_> = block_rows_present.into_iter().collect();
            sorted_bis.sort_unstable();

            for bi in sorted_bis {
                let row_start = bi * block_rows;

                // Extract block data
                let mut block_data = vec![T::zero(); block_rows * block_cols];
                let mut has_nonzero = false;

                for col in col_start..col_end {
                    let local_j = col - col_start;
                    for (row, val) in csc.col_iter(col) {
                        if row >= row_start && row < row_start + block_rows {
                            let local_i = row - row_start;
                            if Scalar::abs(val.clone()) > eps {
                                block_data[local_i * block_cols + local_j] = val.clone();
                                has_nonzero = true;
                            }
                        }
                    }
                }

                if has_nonzero {
                    indices.push(bi);
                    data.push(DenseBlock::new(block_rows, block_cols, block_data));
                }
            }

            indptr.push(data.len());
        }

        Self {
            nrows,
            ncols,
            block_rows,
            block_cols,
            mb,
            nb,
            indptr,
            indices,
            data,
        }
    }

    /// Converts to dense matrix.
    pub fn to_dense(&self) -> oxiblas_matrix::Mat<T>
    where
        T: Field + bytemuck::Zeroable,
    {
        let mut dense = oxiblas_matrix::Mat::zeros(self.nrows, self.ncols);

        for bj in 0..self.nb {
            let start = self.indptr[bj];
            let end = self.indptr[bj + 1];
            let col_start = bj * self.block_cols;

            for k in start..end {
                let bi = self.indices[k];
                let block = &self.data[k];
                let row_start = bi * self.block_rows;

                for i in 0..self.block_rows {
                    let global_row = row_start + i;
                    if global_row >= self.nrows {
                        break;
                    }

                    for j in 0..self.block_cols {
                        let global_col = col_start + j;
                        if global_col >= self.ncols {
                            break;
                        }

                        dense[(global_row, global_col)] = block.get(i, j).clone();
                    }
                }
            }
        }

        dense
    }

    /// Creates a BSC matrix from a dense matrix.
    ///
    /// # Arguments
    ///
    /// * `dense` - Source dense matrix
    /// * `block_rows` - Block row size
    /// * `block_cols` - Block column size
    pub fn from_dense(
        dense: &oxiblas_matrix::MatRef<'_, T>,
        block_rows: usize,
        block_cols: usize,
    ) -> Self
    where
        T: Field,
    {
        let (nrows, ncols) = dense.shape();
        let eps = <T as Scalar>::epsilon();

        let mb = nrows.div_ceil(block_rows);
        let nb = ncols.div_ceil(block_cols);

        let mut indptr = Vec::with_capacity(nb + 1);
        let mut indices = Vec::new();
        let mut data = Vec::new();

        indptr.push(0);

        for bj in 0..nb {
            let col_start = bj * block_cols;
            let col_end = (col_start + block_cols).min(ncols);

            for bi in 0..mb {
                let row_start = bi * block_rows;
                let row_end = (row_start + block_rows).min(nrows);

                // Check if block has any non-zeros
                let mut has_nonzero = false;
                'outer: for i in row_start..row_end {
                    for j in col_start..col_end {
                        if Scalar::abs(dense[(i, j)].clone()) > eps {
                            has_nonzero = true;
                            break 'outer;
                        }
                    }
                }

                if has_nonzero {
                    // Extract block
                    let mut block_data = vec![T::zero(); block_rows * block_cols];
                    for i in 0..block_rows {
                        let global_row = row_start + i;
                        if global_row >= nrows {
                            break;
                        }
                        for j in 0..block_cols {
                            let global_col = col_start + j;
                            if global_col >= ncols {
                                break;
                            }
                            block_data[i * block_cols + j] =
                                dense[(global_row, global_col)].clone();
                        }
                    }

                    indices.push(bi);
                    data.push(DenseBlock::new(block_rows, block_cols, block_data));
                }
            }

            indptr.push(data.len());
        }

        Self {
            nrows,
            ncols,
            block_rows,
            block_cols,
            mb,
            nb,
            indptr,
            indices,
            data,
        }
    }

    /// Scales all values by a scalar.
    pub fn scale(&mut self, alpha: T) {
        for block in &mut self.data {
            block.scale(alpha.clone());
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
    /// The transpose swaps row and column indices and transposes each block.
    pub fn transpose(&self) -> Self
    where
        T: Field,
    {
        // Collect all blocks with transposed positions
        let mut block_entries: Vec<(usize, usize, DenseBlock<T>)> =
            Vec::with_capacity(self.nblocks());

        for bj in 0..self.nb {
            let start = self.indptr[bj];
            let end = self.indptr[bj + 1];
            for k in start..end {
                let bi = self.indices[k];
                let block = &self.data[k];

                // Transpose the block
                let mut transposed_data = vec![T::zero(); self.block_rows * self.block_cols];
                for i in 0..self.block_rows {
                    for j in 0..self.block_cols {
                        transposed_data[j * self.block_rows + i] = block.get(i, j).clone();
                    }
                }
                let transposed_block =
                    DenseBlock::new(self.block_cols, self.block_rows, transposed_data);

                // Swap bi and bj
                block_entries.push((bj, bi, transposed_block));
            }
        }

        // Sort by new block column, then block row
        block_entries.sort_by_key(|(bi, bj, _)| (*bj, *bi));

        // Build new BSC structure
        // Transposed matrix: ncols becomes nrows, nrows becomes ncols
        // block_cols becomes block_rows, block_rows becomes block_cols
        // nb becomes mb, mb becomes nb
        let new_nb = self.mb;
        let new_mb = self.nb;

        let mut indptr = Vec::with_capacity(new_nb + 1);
        let mut indices = Vec::with_capacity(block_entries.len());
        let mut data = Vec::with_capacity(block_entries.len());

        indptr.push(0);

        let mut entry_idx = 0;
        for bj in 0..new_nb {
            while entry_idx < block_entries.len() && block_entries[entry_idx].1 == bj {
                let (bi, _, block) = &block_entries[entry_idx];
                indices.push(*bi);
                data.push(block.clone());
                entry_idx += 1;
            }
            indptr.push(data.len());
        }

        Self {
            nrows: self.ncols,
            ncols: self.nrows,
            block_rows: self.block_cols,
            block_cols: self.block_rows,
            mb: new_mb,
            nb: new_nb,
            indptr,
            indices,
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bsc_new() {
        // 4×4 matrix with 2×2 blocks:
        // [1 2 | 0 0]
        // [3 4 | 0 0]
        // [----+----]
        // [0 0 | 5 6]
        // [0 0 | 7 8]
        let block1 = DenseBlock::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let block2 = DenseBlock::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);

        let bsc =
            BscMatrix::new(4, 4, 2, 2, vec![0, 1, 2], vec![0, 1], vec![block1, block2]).unwrap();

        assert_eq!(bsc.nrows(), 4);
        assert_eq!(bsc.ncols(), 4);
        assert_eq!(bsc.nblocks(), 2);
        assert_eq!(bsc.block_shape(), (2, 2));
    }

    #[test]
    fn test_bsc_get() {
        let block1 = DenseBlock::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let block2 = DenseBlock::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);

        let bsc =
            BscMatrix::new(4, 4, 2, 2, vec![0, 1, 2], vec![0, 1], vec![block1, block2]).unwrap();

        // Block 1
        assert_eq!(bsc.get(0, 0), Some(1.0));
        assert_eq!(bsc.get(0, 1), Some(2.0));
        assert_eq!(bsc.get(1, 0), Some(3.0));
        assert_eq!(bsc.get(1, 1), Some(4.0));

        // Block 2
        assert_eq!(bsc.get(2, 2), Some(5.0));
        assert_eq!(bsc.get(2, 3), Some(6.0));
        assert_eq!(bsc.get(3, 2), Some(7.0));
        assert_eq!(bsc.get(3, 3), Some(8.0));

        // Zero blocks
        assert_eq!(bsc.get(0, 2), None);
        assert_eq!(bsc.get(2, 0), None);
    }

    #[test]
    fn test_bsc_matvec() {
        // [1 2 0 0]   [1]   [3]
        // [3 4 0 0] * [1] = [7]
        // [0 0 5 6]   [1]   [11]
        // [0 0 7 8]   [1]   [15]
        let block1 = DenseBlock::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let block2 = DenseBlock::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);

        let bsc =
            BscMatrix::new(4, 4, 2, 2, vec![0, 1, 2], vec![0, 1], vec![block1, block2]).unwrap();

        let x = vec![1.0, 1.0, 1.0, 1.0];
        let y = bsc.mul_vec(&x);

        assert!((y[0] - 3.0).abs() < 1e-10);
        assert!((y[1] - 7.0).abs() < 1e-10);
        assert!((y[2] - 11.0).abs() < 1e-10);
        assert!((y[3] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_bsc_matvec_transpose() {
        // [1 2 0 0]^T   [1]   [1*1 + 3*1]   [4]
        // [3 4 0 0]   * [1] = [2*1 + 4*1] = [6]
        // [0 0 5 6]     [1]   [5*1 + 7*1]   [12]
        // [0 0 7 8]     [1]   [6*1 + 8*1]   [14]
        let block1 = DenseBlock::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let block2 = DenseBlock::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);

        let bsc =
            BscMatrix::new(4, 4, 2, 2, vec![0, 1, 2], vec![0, 1], vec![block1, block2]).unwrap();

        let x = vec![1.0, 1.0, 1.0, 1.0];
        let mut y = vec![0.0; 4];
        bsc.matvec_transpose(&x, &mut y);

        assert!((y[0] - 4.0).abs() < 1e-10);
        assert!((y[1] - 6.0).abs() < 1e-10);
        assert!((y[2] - 12.0).abs() < 1e-10);
        assert!((y[3] - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_bsc_eye() {
        let bsc: BscMatrix<f64> = BscMatrix::eye(4, 2);

        assert_eq!(bsc.nrows(), 4);
        assert_eq!(bsc.ncols(), 4);
        assert_eq!(bsc.nblocks(), 2);

        for i in 0..4 {
            assert_eq!(bsc.get(i, i), Some(1.0));
        }
    }

    #[test]
    fn test_bsc_to_dense() {
        let block1 = DenseBlock::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let block2 = DenseBlock::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);

        let bsc =
            BscMatrix::new(4, 4, 2, 2, vec![0, 1, 2], vec![0, 1], vec![block1, block2]).unwrap();

        let dense = bsc.to_dense();

        assert!((dense[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((dense[(0, 1)] - 2.0).abs() < 1e-10);
        assert!((dense[(1, 0)] - 3.0).abs() < 1e-10);
        assert!((dense[(1, 1)] - 4.0).abs() < 1e-10);
        assert!((dense[(0, 2)] - 0.0).abs() < 1e-10);
        assert!((dense[(2, 2)] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_bsc_to_bsr_roundtrip() {
        let block1 = DenseBlock::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let block2 = DenseBlock::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);

        let bsc =
            BscMatrix::new(4, 4, 2, 2, vec![0, 1, 2], vec![0, 1], vec![block1, block2]).unwrap();

        let bsr = bsc.to_bsr();
        let bsc2 = BscMatrix::from_bsr(&bsr);

        // Check values match
        for i in 0..4 {
            for j in 0..4 {
                let v1 = bsc.get_or_zero(i, j);
                let v2 = bsc2.get_or_zero(i, j);
                assert!((v1 - v2).abs() < 1e-10, "Mismatch at ({i}, {j})");
            }
        }
    }

    #[test]
    fn test_bsc_from_dense() {
        use oxiblas_matrix::Mat;

        let dense = Mat::from_rows(&[
            &[1.0f64, 2.0, 0.0, 0.0],
            &[3.0, 4.0, 0.0, 0.0],
            &[0.0, 0.0, 5.0, 6.0],
            &[0.0, 0.0, 7.0, 8.0],
        ]);

        let bsc = BscMatrix::from_dense(&dense.as_ref(), 2, 2);

        assert_eq!(bsc.nblocks(), 2);
        assert_eq!(bsc.get(0, 0), Some(1.0));
        assert_eq!(bsc.get(2, 2), Some(5.0));
    }

    #[test]
    fn test_bsc_scale() {
        let block = DenseBlock::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let mut bsc = BscMatrix::new(2, 2, 2, 2, vec![0, 1], vec![0], vec![block]).unwrap();

        bsc.scale(2.0);

        assert_eq!(bsc.get(0, 0), Some(2.0));
        assert_eq!(bsc.get(1, 1), Some(8.0));
    }

    #[test]
    fn test_bsc_transpose() {
        // [1 2]       [1 3]
        // [3 4]  ->   [2 4]
        let block = DenseBlock::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let bsc = BscMatrix::new(2, 2, 2, 2, vec![0, 1], vec![0], vec![block]).unwrap();

        let bsc_t = bsc.transpose();
        let dense = bsc.to_dense();
        let dense_t = bsc_t.to_dense();

        for i in 0..2 {
            for j in 0..2 {
                assert!((dense[(i, j)] - dense_t[(j, i)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_bsc_zeros() {
        let bsc: BscMatrix<f64> = BscMatrix::zeros(6, 8, 2, 4);

        assert_eq!(bsc.nrows(), 6);
        assert_eq!(bsc.ncols(), 8);
        assert_eq!(bsc.nblocks(), 0);
        assert_eq!(bsc.nblock_rows(), 3);
        assert_eq!(bsc.nblock_cols(), 2);
    }

    #[test]
    fn test_bsc_non_square_blocks() {
        // 4×6 matrix with 2×3 blocks
        let block1 = DenseBlock::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let block2 = DenseBlock::new(2, 3, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

        let bsc =
            BscMatrix::new(4, 6, 2, 3, vec![0, 1, 2], vec![0, 1], vec![block1, block2]).unwrap();

        assert_eq!(bsc.nrows(), 4);
        assert_eq!(bsc.ncols(), 6);
        assert_eq!(bsc.nblock_rows(), 2);
        assert_eq!(bsc.nblock_cols(), 2);

        // Check values
        assert_eq!(bsc.get(0, 0), Some(1.0));
        assert_eq!(bsc.get(1, 2), Some(6.0));
        assert_eq!(bsc.get(2, 3), Some(7.0));
    }

    #[test]
    fn test_bsc_block_iter() {
        let block1 = DenseBlock::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let block2 = DenseBlock::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);

        let bsc =
            BscMatrix::new(4, 4, 2, 2, vec![0, 1, 2], vec![0, 1], vec![block1, block2]).unwrap();

        let blocks: Vec<_> = bsc.block_iter().map(|(bi, bj, _)| (bi, bj)).collect();
        assert_eq!(blocks, vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn test_bsc_col_block_iter() {
        let block1 = DenseBlock::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let block2 = DenseBlock::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);

        let bsc =
            BscMatrix::new(4, 4, 2, 2, vec![0, 1, 2], vec![0, 1], vec![block1, block2]).unwrap();

        // Block column 0 contains block (0, 0)
        let col0_blocks: Vec<_> = bsc.col_block_iter(0).map(|(bi, _)| bi).collect();
        assert_eq!(col0_blocks, vec![0]);

        // Block column 1 contains block (1, 1)
        let col1_blocks: Vec<_> = bsc.col_block_iter(1).map(|(bi, _)| bi).collect();
        assert_eq!(col1_blocks, vec![1]);
    }
}
