//! Block Sparse Row (BSR) matrix format.
//!
//! BSR stores matrix data as dense blocks, using:
//! - `data`: Array of dense r×c blocks
//! - `indices`: Block column indices (like CSR but for block columns)
//! - `indptr`: Block row pointers (like CSR but for block rows)
//!
//! For an m×n matrix with r×c blocks:
//! - Number of block rows: mb = ceil(m/r)
//! - Number of block columns: nb = ceil(n/c)
//!
//! # When to Use BSR
//!
//! BSR format is optimal for:
//! - Block-structured matrices (FEM, structural mechanics)
//! - Matrices with dense subblocks
//! - When block size matches SIMD register width
//! - Matrix-vector products with vectorized block operations
//!
//! BSR is NOT efficient for:
//! - Matrices without block structure
//! - Irregular sparsity patterns
//! - Very small matrices

use oxiblas_core::scalar::{Field, Scalar};

/// Error type for BSR matrix operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BsrError {
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
    /// Block column index out of bounds.
    InvalidBlockIndex {
        /// The invalid index.
        index: usize,
        /// Number of block columns.
        nb_cols: usize,
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

impl core::fmt::Display for BsrError {
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
            Self::InvalidBlockIndex { index, nb_cols } => {
                write!(
                    f,
                    "Block column index {index} out of bounds (nb_cols={nb_cols})"
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

impl std::error::Error for BsrError {}

/// A dense block stored in row-major order.
#[derive(Debug, Clone)]
pub struct DenseBlock<T: Scalar> {
    /// Block data in row-major order.
    data: Vec<T>,
    /// Number of rows in block.
    rows: usize,
    /// Number of columns in block.
    cols: usize,
}

impl<T: Scalar + Clone> DenseBlock<T> {
    /// Creates a new dense block.
    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> Self {
        debug_assert_eq!(data.len(), rows * cols);
        Self { data, rows, cols }
    }

    /// Creates a zero block.
    pub fn zeros(rows: usize, cols: usize) -> Self
    where
        T: Field,
    {
        Self {
            data: vec![T::zero(); rows * cols],
            rows,
            cols,
        }
    }

    /// Gets element at (i, j) within the block.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> &T {
        &self.data[i * self.cols + j]
    }

    /// Gets mutable element at (i, j) within the block.
    #[inline]
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut T {
        &mut self.data[i * self.cols + j]
    }

    /// Returns block dimensions.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Returns the data as a slice.
    #[inline]
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Returns mutable data slice.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Block matrix-vector product: y += A * x.
    pub fn matvec_add(&self, x: &[T], y: &mut [T])
    where
        T: Field,
    {
        for i in 0..self.rows {
            for j in 0..self.cols {
                y[i] = y[i].clone() + self.get(i, j).clone() * x[j].clone();
            }
        }
    }

    /// Scales the block by a scalar.
    pub fn scale(&mut self, alpha: T) {
        for val in &mut self.data {
            *val = val.clone() * alpha.clone();
        }
    }

    /// Returns the Frobenius norm squared.
    pub fn frobenius_norm_sq(&self) -> T
    where
        T: Field,
    {
        self.data
            .iter()
            .fold(T::zero(), |acc, val| acc + val.clone() * val.clone())
    }
}

/// Block Sparse Row matrix format.
///
/// Efficient for:
/// - Block-structured problems (FEM, etc.)
/// - Vectorized block operations
/// - Dense subblocks
///
/// # Storage
///
/// Stores sparse matrices as a collection of dense blocks arranged in CSR-like
/// structure. Each block is r×c dense matrix.
///
/// # Example
///
/// ```
/// use oxiblas_sparse::{BsrMatrix, DenseBlock};
///
/// // 4×4 matrix with 2×2 blocks:
/// // [1 2 | 0 0]
/// // [3 4 | 0 0]
/// // [----+----]
/// // [0 0 | 5 6]
/// // [0 0 | 7 8]
/// let block1 = DenseBlock::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
/// let block2 = DenseBlock::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
///
/// let bsr = BsrMatrix::new(
///     4, 4,           // matrix dimensions
///     2, 2,           // block dimensions
///     vec![0, 1, 2],  // indptr (2 block rows)
///     vec![0, 1],     // indices (block columns)
///     vec![block1, block2],
/// ).unwrap();
///
/// assert_eq!(bsr.nblocks(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct BsrMatrix<T: Scalar> {
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
    /// Row pointers for blocks.
    indptr: Vec<usize>,
    /// Block column indices.
    indices: Vec<usize>,
    /// Dense blocks.
    data: Vec<DenseBlock<T>>,
}

impl<T: Scalar + Clone> BsrMatrix<T> {
    /// Creates a new BSR matrix from raw components.
    ///
    /// # Arguments
    ///
    /// * `nrows` - Number of matrix rows
    /// * `ncols` - Number of matrix columns
    /// * `block_rows` - Number of rows per block
    /// * `block_cols` - Number of columns per block
    /// * `indptr` - Block row pointers (length mb + 1)
    /// * `indices` - Block column indices
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
    ) -> Result<Self, BsrError> {
        // Validate block size
        if block_rows == 0 || block_cols == 0 {
            return Err(BsrError::InvalidBlockSize {
                block_rows,
                block_cols,
            });
        }

        // Calculate number of block rows/cols
        let mb = nrows.div_ceil(block_rows);
        let nb = ncols.div_ceil(block_cols);

        // Validate indptr length
        if indptr.len() != mb + 1 {
            return Err(BsrError::InvalidIndptr {
                expected: mb + 1,
                actual: indptr.len(),
            });
        }

        // Validate indptr is monotonically increasing
        for i in 1..indptr.len() {
            if indptr[i] < indptr[i - 1] {
                return Err(BsrError::InvalidIndptrOrder);
            }
        }

        // Validate data and indices count
        let nnz_blocks = data.len();
        if indices.len() != nnz_blocks {
            return Err(BsrError::DataIndicesMismatch {
                num_blocks: nnz_blocks,
                num_indices: indices.len(),
            });
        }

        // Validate indptr[mb] equals nnz_blocks
        if indptr[mb] != nnz_blocks {
            return Err(BsrError::InvalidIndptr {
                expected: nnz_blocks,
                actual: indptr[mb],
            });
        }

        // Validate block column indices
        for &idx in &indices {
            if idx >= nb {
                return Err(BsrError::InvalidBlockIndex {
                    index: idx,
                    nb_cols: nb,
                });
            }
        }

        // Validate block sizes
        let block_size = block_rows * block_cols;
        for (i, block) in data.iter().enumerate() {
            if block.data.len() != block_size {
                return Err(BsrError::InvalidBlockData {
                    block_idx: i,
                    expected: block_size,
                    actual: block.data.len(),
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

    /// Creates a BSR matrix without validation (unsafe but faster).
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

    /// Creates an empty BSR matrix with given dimensions.
    pub fn zeros(nrows: usize, ncols: usize, block_rows: usize, block_cols: usize) -> Self {
        let mb = nrows.div_ceil(block_rows);
        Self {
            nrows,
            ncols,
            block_rows,
            block_cols,
            mb,
            nb: ncols.div_ceil(block_cols),
            indptr: vec![0; mb + 1],
            indices: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Creates an identity matrix in BSR format.
    pub fn eye(n: usize, block_size: usize) -> Self
    where
        T: Field,
    {
        let mb = n.div_ceil(block_size);
        let mut indptr = Vec::with_capacity(mb + 1);
        let mut indices = Vec::with_capacity(mb);
        let mut data = Vec::with_capacity(mb);

        indptr.push(0);

        for bi in 0..mb {
            indices.push(bi);

            // Create identity block
            let mut block_data = vec![T::zero(); block_size * block_size];
            for i in 0..block_size {
                let global_row = bi * block_size + i;
                if global_row < n {
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
            mb,
            nb: mb,
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

    /// Returns the block row pointers.
    #[inline]
    pub fn indptr(&self) -> &[usize] {
        &self.indptr
    }

    /// Returns the block column indices.
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

        let start = self.indptr[bi];
        let end = self.indptr[bi + 1];

        for k in start..end {
            if self.indices[k] == bj {
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
        (0..self.mb).flat_map(move |bi| {
            let start = self.indptr[bi];
            let end = self.indptr[bi + 1];

            (start..end).map(move |k| (bi, self.indices[k], &self.data[k]))
        })
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

        // Process each block row
        for bi in 0..self.mb {
            let start = self.indptr[bi];
            let end = self.indptr[bi + 1];
            let row_start = bi * self.block_rows;
            let row_end = (row_start + self.block_rows).min(self.nrows);

            for k in start..end {
                let bj = self.indices[k];
                let block = &self.data[k];
                let col_start = bj * self.block_cols;
                let col_end = (col_start + self.block_cols).min(self.ncols);

                // Block matrix-vector product
                for (i, yi) in y[row_start..row_end].iter_mut().enumerate() {
                    for j in 0..(col_end - col_start) {
                        *yi = yi.clone() + block.get(i, j).clone() * x[col_start + j].clone();
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
            let bi = row / self.block_rows;
            let local_i = row % self.block_rows;

            let block_start = self.indptr[bi];
            let block_end = self.indptr[bi + 1];

            let mut row_entries: Vec<(usize, T)> = Vec::new();

            for k in block_start..block_end {
                let bj = self.indices[k];
                let block = &self.data[k];

                for j in 0..self.block_cols {
                    let global_col = bj * self.block_cols + j;
                    if global_col < self.ncols {
                        let val = block.get(local_i, j).clone();
                        if Scalar::abs(val.clone()) > eps {
                            row_entries.push((global_col, val));
                        }
                    }
                }
            }

            // Sort by column
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

        for bi in 0..self.mb {
            let start = self.indptr[bi];
            let end = self.indptr[bi + 1];
            let row_start = bi * self.block_rows;

            for k in start..end {
                let bj = self.indices[k];
                let block = &self.data[k];
                let col_start = bj * self.block_cols;

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

    /// Creates a BSR matrix from a dense matrix.
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

        let mut indptr = Vec::with_capacity(mb + 1);
        let mut indices = Vec::new();
        let mut data = Vec::new();

        indptr.push(0);

        for bi in 0..mb {
            let row_start = bi * block_rows;
            let row_end = (row_start + block_rows).min(nrows);

            for bj in 0..nb {
                let col_start = bj * block_cols;
                let col_end = (col_start + block_cols).min(ncols);

                // Check if block has any non-zeros
                let mut has_nonzero = false;
                for i in row_start..row_end {
                    for j in col_start..col_end {
                        if Scalar::abs(dense[(i, j)].clone()) > eps {
                            has_nonzero = true;
                            break;
                        }
                    }
                    if has_nonzero {
                        break;
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

                    indices.push(bj);
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

    /// Creates a BSR matrix from CSR format.
    ///
    /// # Arguments
    ///
    /// * `csr` - Source CSR matrix
    /// * `block_rows` - Block row size
    /// * `block_cols` - Block column size
    pub fn from_csr(csr: &crate::csr::CsrMatrix<T>, block_rows: usize, block_cols: usize) -> Self
    where
        T: Field,
    {
        let (nrows, ncols) = csr.shape();
        let eps = <T as Scalar>::epsilon();

        let mb = nrows.div_ceil(block_rows);
        let nb = ncols.div_ceil(block_cols);

        let mut indptr = Vec::with_capacity(mb + 1);
        let mut indices = Vec::new();
        let mut data = Vec::new();

        indptr.push(0);

        for bi in 0..mb {
            let row_start = bi * block_rows;
            let row_end = (row_start + block_rows).min(nrows);

            // Collect all block columns that have entries in this block row
            let mut block_cols_present = std::collections::HashSet::new();
            for row in row_start..row_end {
                for (col, _) in csr.row_iter(row) {
                    let bj = col / block_cols;
                    block_cols_present.insert(bj);
                }
            }

            // Sort block columns
            let mut sorted_bjs: Vec<_> = block_cols_present.into_iter().collect();
            sorted_bjs.sort();

            for bj in sorted_bjs {
                let col_start = bj * block_cols;

                // Extract block data
                let mut block_data = vec![T::zero(); block_rows * block_cols];
                let mut has_nonzero = false;

                for row in row_start..row_end {
                    let local_i = row - row_start;
                    for (col, val) in csr.row_iter(row) {
                        if col >= col_start && col < col_start + block_cols {
                            let local_j = col - col_start;
                            if Scalar::abs(val.clone()) > eps {
                                block_data[local_i * block_cols + local_j] = val.clone();
                                has_nonzero = true;
                            }
                        }
                    }
                }

                if has_nonzero {
                    indices.push(bj);
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
    pub fn transpose(&self) -> Self
    where
        T: Field,
    {
        // Convert to CSR, transpose, convert back
        let csr = self.to_csr();
        let csr_t = csr.transpose();
        Self::from_csr(&csr_t, self.block_cols, self.block_rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bsr_new() {
        // 4×4 matrix with 2×2 blocks:
        // [1 2 | 0 0]
        // [3 4 | 0 0]
        // [----+----]
        // [0 0 | 5 6]
        // [0 0 | 7 8]
        let block1 = DenseBlock::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let block2 = DenseBlock::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);

        let bsr =
            BsrMatrix::new(4, 4, 2, 2, vec![0, 1, 2], vec![0, 1], vec![block1, block2]).unwrap();

        assert_eq!(bsr.nrows(), 4);
        assert_eq!(bsr.ncols(), 4);
        assert_eq!(bsr.nblocks(), 2);
        assert_eq!(bsr.block_shape(), (2, 2));
    }

    #[test]
    fn test_bsr_get() {
        let block1 = DenseBlock::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let block2 = DenseBlock::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);

        let bsr =
            BsrMatrix::new(4, 4, 2, 2, vec![0, 1, 2], vec![0, 1], vec![block1, block2]).unwrap();

        // Block 1
        assert_eq!(bsr.get(0, 0), Some(1.0));
        assert_eq!(bsr.get(0, 1), Some(2.0));
        assert_eq!(bsr.get(1, 0), Some(3.0));
        assert_eq!(bsr.get(1, 1), Some(4.0));

        // Block 2
        assert_eq!(bsr.get(2, 2), Some(5.0));
        assert_eq!(bsr.get(2, 3), Some(6.0));
        assert_eq!(bsr.get(3, 2), Some(7.0));
        assert_eq!(bsr.get(3, 3), Some(8.0));

        // Zero blocks
        assert_eq!(bsr.get(0, 2), None);
        assert_eq!(bsr.get(2, 0), None);
    }

    #[test]
    fn test_bsr_matvec() {
        // [1 2 0 0]   [1]   [3]
        // [3 4 0 0] * [1] = [7]
        // [0 0 5 6]   [1]   [11]
        // [0 0 7 8]   [1]   [15]
        let block1 = DenseBlock::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let block2 = DenseBlock::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);

        let bsr =
            BsrMatrix::new(4, 4, 2, 2, vec![0, 1, 2], vec![0, 1], vec![block1, block2]).unwrap();

        let x = vec![1.0, 1.0, 1.0, 1.0];
        let y = bsr.mul_vec(&x);

        assert!((y[0] - 3.0).abs() < 1e-10);
        assert!((y[1] - 7.0).abs() < 1e-10);
        assert!((y[2] - 11.0).abs() < 1e-10);
        assert!((y[3] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_bsr_eye() {
        let bsr: BsrMatrix<f64> = BsrMatrix::eye(4, 2);

        assert_eq!(bsr.nrows(), 4);
        assert_eq!(bsr.ncols(), 4);
        assert_eq!(bsr.nblocks(), 2);

        for i in 0..4 {
            assert_eq!(bsr.get(i, i), Some(1.0));
        }
    }

    #[test]
    fn test_bsr_to_dense() {
        let block1 = DenseBlock::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let block2 = DenseBlock::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);

        let bsr =
            BsrMatrix::new(4, 4, 2, 2, vec![0, 1, 2], vec![0, 1], vec![block1, block2]).unwrap();

        let dense = bsr.to_dense();

        assert!((dense[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((dense[(0, 1)] - 2.0).abs() < 1e-10);
        assert!((dense[(1, 0)] - 3.0).abs() < 1e-10);
        assert!((dense[(1, 1)] - 4.0).abs() < 1e-10);
        assert!((dense[(0, 2)] - 0.0).abs() < 1e-10);
        assert!((dense[(2, 2)] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_bsr_to_csr() {
        let block1 = DenseBlock::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let block2 = DenseBlock::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);

        let bsr =
            BsrMatrix::new(4, 4, 2, 2, vec![0, 1, 2], vec![0, 1], vec![block1, block2]).unwrap();

        let csr = bsr.to_csr();

        assert_eq!(csr.nrows(), 4);
        assert_eq!(csr.ncols(), 4);
        assert_eq!(csr.get(0, 0), Some(&1.0));
        assert_eq!(csr.get(2, 2), Some(&5.0));
    }

    #[test]
    fn test_bsr_from_dense() {
        use oxiblas_matrix::Mat;

        let dense = Mat::from_rows(&[
            &[1.0f64, 2.0, 0.0, 0.0],
            &[3.0, 4.0, 0.0, 0.0],
            &[0.0, 0.0, 5.0, 6.0],
            &[0.0, 0.0, 7.0, 8.0],
        ]);

        let bsr = BsrMatrix::from_dense(&dense.as_ref(), 2, 2);

        assert_eq!(bsr.nblocks(), 2);
        assert_eq!(bsr.get(0, 0), Some(1.0));
        assert_eq!(bsr.get(2, 2), Some(5.0));
    }

    #[test]
    fn test_bsr_from_csr() {
        let values = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let col_indices = vec![0, 1, 0, 1, 2, 3, 2, 3];
        let row_ptrs = vec![0, 2, 4, 6, 8];

        let csr = crate::csr::CsrMatrix::new(4, 4, row_ptrs, col_indices, values).unwrap();
        let bsr = BsrMatrix::from_csr(&csr, 2, 2);

        assert_eq!(bsr.nblocks(), 2);
        assert_eq!(bsr.get(0, 0), Some(1.0));
        assert_eq!(bsr.get(3, 3), Some(8.0));
    }

    #[test]
    fn test_bsr_scale() {
        let block = DenseBlock::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let mut bsr = BsrMatrix::new(2, 2, 2, 2, vec![0, 1], vec![0], vec![block]).unwrap();

        bsr.scale(2.0);

        assert_eq!(bsr.get(0, 0), Some(2.0));
        assert_eq!(bsr.get(1, 1), Some(8.0));
    }

    #[test]
    fn test_bsr_transpose() {
        // [1 2]       [1 3]
        // [3 4]  ->   [2 4]
        let block = DenseBlock::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let bsr = BsrMatrix::new(2, 2, 2, 2, vec![0, 1], vec![0], vec![block]).unwrap();

        let bsr_t = bsr.transpose();
        let dense = bsr.to_dense();
        let dense_t = bsr_t.to_dense();

        for i in 0..2 {
            for j in 0..2 {
                assert!((dense[(i, j)] - dense_t[(j, i)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_bsr_zeros() {
        let bsr: BsrMatrix<f64> = BsrMatrix::zeros(6, 8, 2, 4);

        assert_eq!(bsr.nrows(), 6);
        assert_eq!(bsr.ncols(), 8);
        assert_eq!(bsr.nblocks(), 0);
        assert_eq!(bsr.nblock_rows(), 3);
        assert_eq!(bsr.nblock_cols(), 2);
    }

    #[test]
    fn test_bsr_non_square_blocks() {
        // 6×4 matrix with 3×2 blocks
        let block1 = DenseBlock::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let block2 = DenseBlock::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

        let bsr =
            BsrMatrix::new(6, 4, 3, 2, vec![0, 1, 2], vec![0, 1], vec![block1, block2]).unwrap();

        assert_eq!(bsr.nrows(), 6);
        assert_eq!(bsr.ncols(), 4);
        assert_eq!(bsr.nblock_rows(), 2);
        assert_eq!(bsr.nblock_cols(), 2);

        // Check values
        assert_eq!(bsr.get(0, 0), Some(1.0));
        assert_eq!(bsr.get(2, 1), Some(6.0));
        assert_eq!(bsr.get(3, 2), Some(7.0));
    }

    #[test]
    fn test_bsr_block_iter() {
        let block1 = DenseBlock::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let block2 = DenseBlock::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);

        let bsr =
            BsrMatrix::new(4, 4, 2, 2, vec![0, 1, 2], vec![0, 1], vec![block1, block2]).unwrap();

        let blocks: Vec<_> = bsr.block_iter().map(|(bi, bj, _)| (bi, bj)).collect();
        assert_eq!(blocks, vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn test_dense_block() {
        let mut block = DenseBlock::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        assert_eq!(block.shape(), (2, 3));
        assert_eq!(*block.get(0, 0), 1.0);
        assert_eq!(*block.get(0, 2), 3.0);
        assert_eq!(*block.get(1, 1), 5.0);

        *block.get_mut(1, 1) = 10.0;
        assert_eq!(*block.get(1, 1), 10.0);
    }
}
