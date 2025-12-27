//! Matrix builder patterns for ergonomic matrix creation.
//!
//! This module provides a fluent API for creating matrices with various
//! initialization strategies.
//!
//! # Examples
//!
//! ```
//! use oxiblas::builder::MatBuilder;
//!
//! // Create a 4x4 identity matrix
//! let eye = MatBuilder::<f64>::identity(4);
//!
//! // Create a 3x3 matrix of ones
//! let ones = MatBuilder::<f64>::ones(3, 3);
//!
//! // Create a 5x5 matrix from a function
//! let custom = MatBuilder::from_fn(5, 5, |i, j| (i + j) as f64);
//!
//! // Create a diagonal matrix
//! let diag = MatBuilder::diagonal(&[1.0, 2.0, 3.0, 4.0]);
//!
//! // Create a Hilbert matrix
//! let hilbert = MatBuilder::<f64>::hilbert(4);
//! ```

use oxiblas_core::scalar::Scalar;
use oxiblas_matrix::Mat;

/// Builder for creating matrices with various initialization strategies.
///
/// `MatBuilder` provides a fluent API for matrix creation, supporting:
/// - Standard matrices (zeros, ones, identity, filled)
/// - Function-based initialization
/// - Special matrices (diagonal, tridiagonal, banded)
/// - Famous test matrices (Hilbert, Vandermonde, Toeplitz)
///
/// # Type Parameters
///
/// * `T` - The scalar type of the matrix elements
///
/// # Examples
///
/// ```
/// use oxiblas::builder::MatBuilder;
///
/// // Create matrices with different initialization
/// let zeros = MatBuilder::<f64>::zeros(3, 4);
/// let ones = MatBuilder::<f64>::ones(3, 4);
/// let eye = MatBuilder::<f64>::identity(4);
/// let custom = MatBuilder::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
/// ```
pub struct MatBuilder<T: Scalar> {
    _marker: std::marker::PhantomData<T>,
}

impl<T: Scalar> MatBuilder<T> {
    /// Creates a new matrix filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `nrows` - Number of rows
    /// * `ncols` - Number of columns
    ///
    /// # Returns
    ///
    /// A new `Mat<T>` with all elements set to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiblas::builder::MatBuilder;
    ///
    /// let m = MatBuilder::<f64>::zeros(3, 4);
    /// assert_eq!(m.nrows(), 3);
    /// assert_eq!(m.ncols(), 4);
    /// assert_eq!(m[(0, 0)], 0.0);
    /// ```
    #[inline]
    pub fn zeros(nrows: usize, ncols: usize) -> Mat<T>
    where
        T: bytemuck::Zeroable,
    {
        Mat::zeros(nrows, ncols)
    }

    /// Creates a new matrix filled with ones.
    ///
    /// # Arguments
    ///
    /// * `nrows` - Number of rows
    /// * `ncols` - Number of columns
    ///
    /// # Returns
    ///
    /// A new `Mat<T>` with all elements set to one.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiblas::builder::MatBuilder;
    ///
    /// let m = MatBuilder::<f64>::ones(2, 3);
    /// assert_eq!(m[(1, 2)], 1.0);
    /// ```
    #[inline]
    pub fn ones(nrows: usize, ncols: usize) -> Mat<T>
    where
        T: num_traits::One,
    {
        Mat::filled(nrows, ncols, T::one())
    }

    /// Creates a new matrix filled with a specific value.
    ///
    /// # Arguments
    ///
    /// * `nrows` - Number of rows
    /// * `ncols` - Number of columns
    /// * `value` - The value to fill the matrix with
    ///
    /// # Returns
    ///
    /// A new `Mat<T>` with all elements set to `value`.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiblas::builder::MatBuilder;
    ///
    /// let m = MatBuilder::filled(2, 2, 3.14);
    /// assert_eq!(m[(0, 0)], 3.14);
    /// assert_eq!(m[(1, 1)], 3.14);
    /// ```
    #[inline]
    pub fn filled(nrows: usize, ncols: usize, value: T) -> Mat<T> {
        Mat::filled(nrows, ncols, value)
    }

    /// Creates an identity matrix of size n×n.
    ///
    /// # Arguments
    ///
    /// * `n` - The size of the square matrix
    ///
    /// # Returns
    ///
    /// A new `Mat<T>` with ones on the diagonal and zeros elsewhere.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiblas::builder::MatBuilder;
    ///
    /// let eye = MatBuilder::<f64>::identity(3);
    /// assert_eq!(eye[(0, 0)], 1.0);
    /// assert_eq!(eye[(0, 1)], 0.0);
    /// assert_eq!(eye[(1, 1)], 1.0);
    /// assert_eq!(eye[(2, 2)], 1.0);
    /// ```
    pub fn identity(n: usize) -> Mat<T>
    where
        T: bytemuck::Zeroable + num_traits::One,
    {
        let mut m = Mat::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = T::one();
        }
        m
    }

    /// Creates a rectangular identity matrix (eye).
    ///
    /// # Arguments
    ///
    /// * `nrows` - Number of rows
    /// * `ncols` - Number of columns
    ///
    /// # Returns
    ///
    /// A new `Mat<T>` with ones on the main diagonal and zeros elsewhere.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiblas::builder::MatBuilder;
    ///
    /// let eye = MatBuilder::<f64>::eye(3, 4);
    /// assert_eq!(eye[(0, 0)], 1.0);
    /// assert_eq!(eye[(1, 1)], 1.0);
    /// assert_eq!(eye[(2, 2)], 1.0);
    /// assert_eq!(eye[(2, 3)], 0.0);
    /// ```
    pub fn eye(nrows: usize, ncols: usize) -> Mat<T>
    where
        T: bytemuck::Zeroable + num_traits::One,
    {
        let mut m = Mat::zeros(nrows, ncols);
        let min_dim = nrows.min(ncols);
        for i in 0..min_dim {
            m[(i, i)] = T::one();
        }
        m
    }

    /// Creates a matrix from a function.
    ///
    /// The function is called with (row, column) indices and should return
    /// the value for that position.
    ///
    /// # Arguments
    ///
    /// * `nrows` - Number of rows
    /// * `ncols` - Number of columns
    /// * `f` - Function mapping (i, j) to a scalar value
    ///
    /// # Returns
    ///
    /// A new `Mat<T>` with elements initialized by the function.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiblas::builder::MatBuilder;
    ///
    /// // Create a matrix where each element is the sum of its indices
    /// let m = MatBuilder::from_fn(3, 3, |i, j| (i + j) as f64);
    /// assert_eq!(m[(0, 0)], 0.0);
    /// assert_eq!(m[(1, 2)], 3.0);
    /// assert_eq!(m[(2, 2)], 4.0);
    /// ```
    pub fn from_fn<F>(nrows: usize, ncols: usize, mut f: F) -> Mat<T>
    where
        F: FnMut(usize, usize) -> T,
    {
        let mut m = Mat::filled(nrows, ncols, f(0, 0));
        for j in 0..ncols {
            for i in 0..nrows {
                m[(i, j)] = f(i, j);
            }
        }
        m
    }

    /// Creates a diagonal matrix from a slice of values.
    ///
    /// # Arguments
    ///
    /// * `diag` - Slice containing the diagonal elements
    ///
    /// # Returns
    ///
    /// A new square `Mat<T>` with the given diagonal and zeros elsewhere.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiblas::builder::MatBuilder;
    ///
    /// let d = MatBuilder::diagonal(&[1.0, 2.0, 3.0]);
    /// assert_eq!(d[(0, 0)], 1.0);
    /// assert_eq!(d[(1, 1)], 2.0);
    /// assert_eq!(d[(2, 2)], 3.0);
    /// assert_eq!(d[(0, 1)], 0.0);
    /// ```
    pub fn diagonal(diag: &[T]) -> Mat<T>
    where
        T: bytemuck::Zeroable,
    {
        let n = diag.len();
        let mut m = Mat::zeros(n, n);
        for (i, &val) in diag.iter().enumerate() {
            m[(i, i)] = val;
        }
        m
    }

    /// Creates a tridiagonal matrix.
    ///
    /// # Arguments
    ///
    /// * `sub` - Sub-diagonal elements (length n-1)
    /// * `main` - Main diagonal elements (length n)
    /// * `sup` - Super-diagonal elements (length n-1)
    ///
    /// # Returns
    ///
    /// A new square `Mat<T>` with the specified diagonals.
    ///
    /// # Panics
    ///
    /// Panics if the lengths are inconsistent.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiblas::builder::MatBuilder;
    ///
    /// let t = MatBuilder::tridiagonal(&[1.0, 2.0], &[4.0, 5.0, 6.0], &[7.0, 8.0]);
    /// // [4, 7, 0]
    /// // [1, 5, 8]
    /// // [0, 2, 6]
    /// assert_eq!(t[(0, 0)], 4.0);
    /// assert_eq!(t[(0, 1)], 7.0);
    /// assert_eq!(t[(1, 0)], 1.0);
    /// assert_eq!(t[(1, 1)], 5.0);
    /// ```
    pub fn tridiagonal(sub: &[T], main: &[T], sup: &[T]) -> Mat<T>
    where
        T: bytemuck::Zeroable,
    {
        let n = main.len();
        assert_eq!(
            sub.len(),
            n.saturating_sub(1),
            "sub-diagonal must have length n-1"
        );
        assert_eq!(
            sup.len(),
            n.saturating_sub(1),
            "super-diagonal must have length n-1"
        );

        let mut m = Mat::zeros(n, n);

        // Main diagonal
        for (i, &val) in main.iter().enumerate() {
            m[(i, i)] = val;
        }

        // Sub-diagonal
        for (i, &val) in sub.iter().enumerate() {
            m[(i + 1, i)] = val;
        }

        // Super-diagonal
        for (i, &val) in sup.iter().enumerate() {
            m[(i, i + 1)] = val;
        }

        m
    }

    /// Creates a banded matrix with the specified diagonals.
    ///
    /// # Arguments
    ///
    /// * `n` - Size of the square matrix
    /// * `diagonals` - Iterator of (offset, values) pairs where offset is the
    ///   diagonal offset (0 = main, positive = super, negative = sub)
    ///
    /// # Returns
    ///
    /// A new square `Mat<T>` with the specified bands.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiblas::builder::MatBuilder;
    ///
    /// // Create a pentadiagonal matrix
    /// let b = MatBuilder::<f64>::banded(
    ///     5,
    ///     vec![
    ///         (0, vec![2.0; 5]),     // main diagonal
    ///         (1, vec![-1.0; 4]),    // super-diagonal
    ///         (-1, vec![-1.0; 4]),   // sub-diagonal
    ///         (2, vec![0.5; 3]),     // second super-diagonal
    ///         (-2, vec![0.5; 3]),    // second sub-diagonal
    ///     ],
    /// );
    /// assert_eq!(b[(0, 0)], 2.0);
    /// assert_eq!(b[(0, 1)], -1.0);
    /// ```
    pub fn banded<I>(n: usize, diagonals: I) -> Mat<T>
    where
        T: bytemuck::Zeroable,
        I: IntoIterator<Item = (isize, Vec<T>)>,
    {
        let mut m = Mat::zeros(n, n);

        for (offset, values) in diagonals {
            let (row_start, col_start) = if offset >= 0 {
                (0, offset as usize)
            } else {
                ((-offset) as usize, 0)
            };

            for (k, &val) in values.iter().enumerate() {
                let i = row_start + k;
                let j = col_start + k;
                if i < n && j < n {
                    m[(i, j)] = val;
                }
            }
        }

        m
    }

    /// Creates a column vector from a slice.
    ///
    /// # Arguments
    ///
    /// * `data` - Slice containing the vector elements
    ///
    /// # Returns
    ///
    /// A new `Mat<T>` with shape (n, 1).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiblas::builder::MatBuilder;
    ///
    /// let v = MatBuilder::col_vec(&[1.0, 2.0, 3.0]);
    /// assert_eq!(v.nrows(), 3);
    /// assert_eq!(v.ncols(), 1);
    /// assert_eq!(v[(1, 0)], 2.0);
    /// ```
    pub fn col_vec(data: &[T]) -> Mat<T>
    where
        T: bytemuck::Zeroable,
    {
        let n = data.len();
        let mut m = Mat::zeros(n, 1);
        for (i, &val) in data.iter().enumerate() {
            m[(i, 0)] = val;
        }
        m
    }

    /// Creates a row vector from a slice.
    ///
    /// # Arguments
    ///
    /// * `data` - Slice containing the vector elements
    ///
    /// # Returns
    ///
    /// A new `Mat<T>` with shape (1, n).
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiblas::builder::MatBuilder;
    ///
    /// let v = MatBuilder::row_vec(&[1.0, 2.0, 3.0]);
    /// assert_eq!(v.nrows(), 1);
    /// assert_eq!(v.ncols(), 3);
    /// assert_eq!(v[(0, 1)], 2.0);
    /// ```
    pub fn row_vec(data: &[T]) -> Mat<T>
    where
        T: bytemuck::Zeroable,
    {
        let n = data.len();
        let mut m = Mat::zeros(1, n);
        for (j, &val) in data.iter().enumerate() {
            m[(0, j)] = val;
        }
        m
    }

    /// Creates a block diagonal matrix from multiple matrices.
    ///
    /// # Arguments
    ///
    /// * `blocks` - Slice of matrices to place on the diagonal
    ///
    /// # Returns
    ///
    /// A new `Mat<T>` with the given matrices on the block diagonal.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiblas::builder::MatBuilder;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
    /// let b: Mat<f64> = Mat::from_rows(&[&[5.0]]);
    /// let block_diag = MatBuilder::block_diagonal(&[a.as_ref(), b.as_ref()]);
    /// // [1, 2, 0]
    /// // [3, 4, 0]
    /// // [0, 0, 5]
    /// assert_eq!(block_diag.nrows(), 3);
    /// assert_eq!(block_diag.ncols(), 3);
    /// assert_eq!(block_diag[(0, 0)], 1.0);
    /// assert_eq!(block_diag[(2, 2)], 5.0);
    /// ```
    pub fn block_diagonal(blocks: &[oxiblas_matrix::MatRef<'_, T>]) -> Mat<T>
    where
        T: bytemuck::Zeroable,
    {
        let total_rows: usize = blocks.iter().map(|b| b.nrows()).sum();
        let total_cols: usize = blocks.iter().map(|b| b.ncols()).sum();

        let mut m = Mat::zeros(total_rows, total_cols);

        let mut row_offset = 0;
        let mut col_offset = 0;

        for block in blocks {
            for j in 0..block.ncols() {
                for i in 0..block.nrows() {
                    m[(row_offset + i, col_offset + j)] = block[(i, j)];
                }
            }
            row_offset += block.nrows();
            col_offset += block.ncols();
        }

        m
    }

    /// Vertically stacks matrices (concatenates along rows).
    ///
    /// # Arguments
    ///
    /// * `matrices` - Slice of matrices to stack vertically
    ///
    /// # Panics
    ///
    /// Panics if matrices have different numbers of columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiblas::builder::MatBuilder;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0]]);
    /// let b: Mat<f64> = Mat::from_rows(&[&[3.0, 4.0], &[5.0, 6.0]]);
    /// let stacked = MatBuilder::vstack(&[a.as_ref(), b.as_ref()]);
    /// assert_eq!(stacked.nrows(), 3);
    /// assert_eq!(stacked.ncols(), 2);
    /// ```
    pub fn vstack(matrices: &[oxiblas_matrix::MatRef<'_, T>]) -> Mat<T>
    where
        T: bytemuck::Zeroable,
    {
        if matrices.is_empty() {
            return Mat::zeros(0, 0);
        }

        let ncols = matrices[0].ncols();
        for m in matrices {
            assert_eq!(
                m.ncols(),
                ncols,
                "All matrices must have the same number of columns"
            );
        }

        let total_rows: usize = matrices.iter().map(|m| m.nrows()).sum();
        let mut result = Mat::zeros(total_rows, ncols);

        let mut row_offset = 0;
        for mat in matrices {
            for j in 0..ncols {
                for i in 0..mat.nrows() {
                    result[(row_offset + i, j)] = mat[(i, j)];
                }
            }
            row_offset += mat.nrows();
        }

        result
    }

    /// Horizontally stacks matrices (concatenates along columns).
    ///
    /// # Arguments
    ///
    /// * `matrices` - Slice of matrices to stack horizontally
    ///
    /// # Panics
    ///
    /// Panics if matrices have different numbers of rows.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiblas::builder::MatBuilder;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a: Mat<f64> = Mat::from_rows(&[&[1.0], &[2.0]]);
    /// let b: Mat<f64> = Mat::from_rows(&[&[3.0, 4.0], &[5.0, 6.0]]);
    /// let stacked = MatBuilder::hstack(&[a.as_ref(), b.as_ref()]);
    /// assert_eq!(stacked.nrows(), 2);
    /// assert_eq!(stacked.ncols(), 3);
    /// ```
    pub fn hstack(matrices: &[oxiblas_matrix::MatRef<'_, T>]) -> Mat<T>
    where
        T: bytemuck::Zeroable,
    {
        if matrices.is_empty() {
            return Mat::zeros(0, 0);
        }

        let nrows = matrices[0].nrows();
        for m in matrices {
            assert_eq!(
                m.nrows(),
                nrows,
                "All matrices must have the same number of rows"
            );
        }

        let total_cols: usize = matrices.iter().map(|m| m.ncols()).sum();
        let mut result = Mat::zeros(nrows, total_cols);

        let mut col_offset = 0;
        for mat in matrices {
            for j in 0..mat.ncols() {
                for i in 0..nrows {
                    result[(i, col_offset + j)] = mat[(i, j)];
                }
            }
            col_offset += mat.ncols();
        }

        result
    }
}

// Special matrices for f64
impl MatBuilder<f64> {
    /// Creates a Hilbert matrix.
    ///
    /// The Hilbert matrix H has elements `H(i,j) = 1/(i+j+1)`.
    /// It is a famous ill-conditioned matrix used in numerical analysis.
    ///
    /// # Arguments
    ///
    /// * `n` - Size of the square matrix
    ///
    /// # Returns
    ///
    /// A new Hilbert matrix of size n×n.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiblas::builder::MatBuilder;
    ///
    /// let h = MatBuilder::<f64>::hilbert(3);
    /// assert!((h[(0, 0)] - 1.0).abs() < 1e-10);
    /// assert!((h[(0, 1)] - 0.5).abs() < 1e-10);
    /// assert!((h[(1, 1)] - 1.0/3.0).abs() < 1e-10);
    /// ```
    pub fn hilbert(n: usize) -> Mat<f64> {
        Self::from_fn(n, n, |i, j| 1.0 / ((i + j + 1) as f64))
    }

    /// Creates a Vandermonde matrix.
    ///
    /// For a vector v = (v0, v1, ..., vn-1), the Vandermonde matrix V has
    /// `V(i,j) = v_i^j` (or `v_i^(n-1-j)` for the alternate form).
    ///
    /// # Arguments
    ///
    /// * `v` - Vector of base values
    /// * `ncols` - Number of columns (defaults to length of v if None)
    ///
    /// # Returns
    ///
    /// A new Vandermonde matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiblas::builder::MatBuilder;
    ///
    /// let v = MatBuilder::<f64>::vandermonde(&[1.0, 2.0, 3.0], None);
    /// // [1, 1, 1]
    /// // [1, 2, 4]
    /// // [1, 3, 9]
    /// assert_eq!(v[(0, 0)], 1.0);
    /// assert_eq!(v[(1, 2)], 4.0);
    /// assert_eq!(v[(2, 2)], 9.0);
    /// ```
    pub fn vandermonde(v: &[f64], ncols: Option<usize>) -> Mat<f64> {
        let nrows = v.len();
        let ncols = ncols.unwrap_or(nrows);

        Self::from_fn(nrows, ncols, |i, j| v[i].powi(j as i32))
    }

    /// Creates a Toeplitz matrix.
    ///
    /// A Toeplitz matrix has constant diagonals. It is defined by its
    /// first column and first row.
    ///
    /// # Arguments
    ///
    /// * `col` - First column of the matrix
    /// * `row` - First row of the matrix (`col[0]` must equal `row[0]`)
    ///
    /// # Returns
    ///
    /// A new Toeplitz matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiblas::builder::MatBuilder;
    ///
    /// let t = MatBuilder::<f64>::toeplitz(&[1.0, 2.0, 3.0], &[1.0, 4.0, 5.0]);
    /// // [1, 4, 5]
    /// // [2, 1, 4]
    /// // [3, 2, 1]
    /// assert_eq!(t[(0, 0)], 1.0);
    /// assert_eq!(t[(1, 0)], 2.0);
    /// assert_eq!(t[(0, 1)], 4.0);
    /// ```
    pub fn toeplitz(col: &[f64], row: &[f64]) -> Mat<f64> {
        let nrows = col.len();
        let ncols = row.len();

        Self::from_fn(
            nrows,
            ncols,
            |i, j| {
                if j >= i { row[j - i] } else { col[i - j] }
            },
        )
    }

    /// Creates a Hankel matrix.
    ///
    /// A Hankel matrix has constant anti-diagonals. It is defined by its
    /// first column and last row.
    ///
    /// # Arguments
    ///
    /// * `col` - First column of the matrix
    /// * `row` - Last row of the matrix (`col[n-1]` must equal `row[0]` for consistency)
    ///
    /// # Returns
    ///
    /// A new Hankel matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiblas::builder::MatBuilder;
    ///
    /// let h = MatBuilder::<f64>::hankel(&[1.0, 2.0, 3.0], &[3.0, 4.0, 5.0]);
    /// // [1, 2, 3]
    /// // [2, 3, 4]
    /// // [3, 4, 5]
    /// assert_eq!(h[(0, 0)], 1.0);
    /// assert_eq!(h[(1, 1)], 3.0);
    /// assert_eq!(h[(2, 2)], 5.0);
    /// ```
    pub fn hankel(col: &[f64], row: &[f64]) -> Mat<f64> {
        let nrows = col.len();
        let ncols = row.len();

        Self::from_fn(nrows, ncols, |i, j| {
            let sum = i + j;
            if sum < nrows {
                col[sum]
            } else {
                row[sum - nrows + 1]
            }
        })
    }

    /// Creates a companion matrix for a polynomial.
    ///
    /// For a monic polynomial p(x) = x^n + c_{n-1}*x^(n-1) + ... + c_1*x + c_0,
    /// the companion matrix has eigenvalues equal to the polynomial roots.
    ///
    /// # Arguments
    ///
    /// * `coeffs` - Polynomial coefficients [c0, c1, ..., c_{n-1}] (excluding leading 1)
    ///
    /// # Returns
    ///
    /// The companion matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiblas::builder::MatBuilder;
    ///
    /// // Polynomial: x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
    /// let c = MatBuilder::<f64>::companion(&[-6.0, 11.0, -6.0]);
    /// // Eigenvalues should be 1, 2, 3
    /// assert_eq!(c.nrows(), 3);
    /// ```
    pub fn companion(coeffs: &[f64]) -> Mat<f64> {
        let n = coeffs.len();
        if n == 0 {
            return Mat::zeros(0, 0);
        }

        let mut m = Mat::zeros(n, n);

        // Sub-diagonal of ones
        for i in 1..n {
            m[(i, i - 1)] = 1.0;
        }

        // Last column is negative of coefficients
        for (i, &c) in coeffs.iter().enumerate() {
            m[(i, n - 1)] = -c;
        }

        m
    }

    /// Creates a circulant matrix.
    ///
    /// A circulant matrix is a special Toeplitz matrix where each row
    /// is a cyclic shift of the previous row.
    ///
    /// # Arguments
    ///
    /// * `v` - First row of the circulant matrix
    ///
    /// # Returns
    ///
    /// A new circulant matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiblas::builder::MatBuilder;
    ///
    /// let c = MatBuilder::<f64>::circulant(&[1.0, 2.0, 3.0]);
    /// // [1, 2, 3]
    /// // [3, 1, 2]
    /// // [2, 3, 1]
    /// assert_eq!(c[(0, 0)], 1.0);
    /// assert_eq!(c[(1, 0)], 3.0);
    /// assert_eq!(c[(2, 0)], 2.0);
    /// ```
    pub fn circulant(v: &[f64]) -> Mat<f64> {
        let n = v.len();
        Self::from_fn(n, n, |i, j| {
            let idx = (j + n - i) % n;
            v[idx]
        })
    }

    /// Creates a Cauchy matrix.
    ///
    /// The Cauchy matrix C has elements `C(i,j) = 1/(x_i - y_j)`.
    ///
    /// # Arguments
    ///
    /// * `x` - First vector
    /// * `y` - Second vector
    ///
    /// # Returns
    ///
    /// A new Cauchy matrix.
    ///
    /// # Panics
    ///
    /// May produce infinities if `x[i] == y[j]` for any i, j.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiblas::builder::MatBuilder;
    ///
    /// let c = MatBuilder::<f64>::cauchy(&[1.0, 2.0], &[3.0, 4.0, 5.0]);
    /// // C[0,0] = 1/(1-3) = -0.5
    /// assert!((c[(0, 0)] + 0.5).abs() < 1e-10);
    /// ```
    pub fn cauchy(x: &[f64], y: &[f64]) -> Mat<f64> {
        let nrows = x.len();
        let ncols = y.len();
        Self::from_fn(nrows, ncols, |i, j| 1.0 / (x[i] - y[j]))
    }

    /// Creates a random matrix with values uniformly distributed in [0, 1).
    ///
    /// Uses a simple linear congruential generator seeded with the given value.
    /// For production use, consider using a proper random number generator.
    ///
    /// # Arguments
    ///
    /// * `nrows` - Number of rows
    /// * `ncols` - Number of columns
    /// * `seed` - Random seed
    ///
    /// # Returns
    ///
    /// A new random matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiblas::builder::MatBuilder;
    ///
    /// let r = MatBuilder::<f64>::random(3, 3, 42);
    /// // Values should be in [0, 1)
    /// for i in 0..3 {
    ///     for j in 0..3 {
    ///         assert!(r[(i, j)] >= 0.0 && r[(i, j)] < 1.0);
    ///     }
    /// }
    /// ```
    pub fn random(nrows: usize, ncols: usize, seed: u64) -> Mat<f64> {
        let mut state = seed;
        Self::from_fn(nrows, ncols, |_, _| {
            // Simple LCG: state = (a * state + c) mod m
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Convert to [0, 1)
            (state >> 11) as f64 / (1u64 << 53) as f64
        })
    }

    /// Creates a random symmetric positive definite matrix.
    ///
    /// Constructs A * A^T + n*I where A is a random matrix.
    ///
    /// # Arguments
    ///
    /// * `n` - Size of the square matrix
    /// * `seed` - Random seed
    ///
    /// # Returns
    ///
    /// A new symmetric positive definite matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiblas::builder::MatBuilder;
    ///
    /// let spd = MatBuilder::<f64>::random_spd(4, 42);
    /// // Should be symmetric
    /// for i in 0..4 {
    ///     for j in 0..4 {
    ///         assert!((spd[(i, j)] - spd[(j, i)]).abs() < 1e-10);
    ///     }
    /// }
    /// ```
    pub fn random_spd(n: usize, seed: u64) -> Mat<f64> {
        let a = Self::random(n, n, seed);

        // Compute A * A^T + n*I
        let mut result = Mat::zeros(n, n);

        // A * A^T
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += a[(i, k)] * a[(j, k)];
                }
                result[(i, j)] = sum;
            }
        }

        // Add n*I for guaranteed positive definiteness
        for i in 0..n {
            result[(i, i)] += n as f64;
        }

        result
    }

    /// Creates a matrix with linearly spaced values.
    ///
    /// Fills the matrix in column-major order with values from start to end.
    ///
    /// # Arguments
    ///
    /// * `nrows` - Number of rows
    /// * `ncols` - Number of columns
    /// * `start` - Starting value
    /// * `end` - Ending value
    ///
    /// # Returns
    ///
    /// A new matrix with linearly spaced values.
    ///
    /// # Examples
    ///
    /// ```
    /// use oxiblas::builder::MatBuilder;
    ///
    /// let m = MatBuilder::<f64>::linspace(2, 3, 0.0, 5.0);
    /// // Column-major: 0, 1, 2, 3, 4, 5
    /// assert_eq!(m[(0, 0)], 0.0);
    /// assert_eq!(m[(1, 0)], 1.0);
    /// assert_eq!(m[(0, 1)], 2.0);
    /// ```
    pub fn linspace(nrows: usize, ncols: usize, start: f64, end: f64) -> Mat<f64> {
        let total = nrows * ncols;
        if total == 0 {
            return Mat::zeros(nrows, ncols);
        }
        if total == 1 {
            return Mat::filled(nrows, ncols, start);
        }

        let step = (end - start) / ((total - 1) as f64);
        let mut m = Mat::zeros(nrows, ncols);
        let mut idx = 0;
        for j in 0..ncols {
            for i in 0..nrows {
                m[(i, j)] = start + (idx as f64) * step;
                idx += 1;
            }
        }
        m
    }
}

// Special matrices for f32
impl MatBuilder<f32> {
    /// Creates a Hilbert matrix (f32 version).
    pub fn hilbert(n: usize) -> Mat<f32> {
        Self::from_fn(n, n, |i, j| 1.0 / ((i + j + 1) as f32))
    }

    /// Creates a random matrix (f32 version).
    pub fn random(nrows: usize, ncols: usize, seed: u64) -> Mat<f32> {
        let mut state = seed;
        Self::from_fn(nrows, ncols, |_, _| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 11) as f64 / (1u64 << 53) as f64) as f32
        })
    }

    /// Creates a random symmetric positive definite matrix (f32 version).
    pub fn random_spd(n: usize, seed: u64) -> Mat<f32> {
        let a = Self::random(n, n, seed);
        let mut result = Mat::zeros(n, n);

        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0f32;
                for k in 0..n {
                    sum += a[(i, k)] * a[(j, k)];
                }
                result[(i, j)] = sum;
            }
        }

        for i in 0..n {
            result[(i, i)] += n as f32;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let m = MatBuilder::<f64>::zeros(3, 4);
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 4);
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(m[(i, j)], 0.0);
            }
        }
    }

    #[test]
    fn test_ones() {
        let m = MatBuilder::<f64>::ones(2, 3);
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(m[(i, j)], 1.0);
            }
        }
    }

    #[test]
    fn test_identity() {
        let eye = MatBuilder::<f64>::identity(4);
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    assert_eq!(eye[(i, j)], 1.0);
                } else {
                    assert_eq!(eye[(i, j)], 0.0);
                }
            }
        }
    }

    #[test]
    fn test_from_fn() {
        let m = MatBuilder::from_fn(3, 3, |i, j| (i * 10 + j) as f64);
        assert_eq!(m[(0, 0)], 0.0);
        assert_eq!(m[(1, 2)], 12.0);
        assert_eq!(m[(2, 1)], 21.0);
    }

    #[test]
    fn test_diagonal() {
        let d = MatBuilder::diagonal(&[1.0, 2.0, 3.0]);
        assert_eq!(d[(0, 0)], 1.0);
        assert_eq!(d[(1, 1)], 2.0);
        assert_eq!(d[(2, 2)], 3.0);
        assert_eq!(d[(0, 1)], 0.0);
        assert_eq!(d[(1, 0)], 0.0);
    }

    #[test]
    fn test_tridiagonal() {
        let t = MatBuilder::tridiagonal(&[1.0, 2.0], &[4.0, 5.0, 6.0], &[7.0, 8.0]);
        assert_eq!(t[(0, 0)], 4.0);
        assert_eq!(t[(0, 1)], 7.0);
        assert_eq!(t[(1, 0)], 1.0);
        assert_eq!(t[(1, 1)], 5.0);
        assert_eq!(t[(1, 2)], 8.0);
        assert_eq!(t[(2, 1)], 2.0);
        assert_eq!(t[(2, 2)], 6.0);
    }

    #[test]
    fn test_hilbert() {
        let h = MatBuilder::<f64>::hilbert(4);
        assert!((h[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((h[(0, 1)] - 0.5).abs() < 1e-10);
        assert!((h[(1, 0)] - 0.5).abs() < 1e-10);
        assert!((h[(1, 1)] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_vandermonde() {
        let v = MatBuilder::<f64>::vandermonde(&[1.0, 2.0, 3.0], None);
        assert_eq!(v[(0, 0)], 1.0); // 1^0
        assert_eq!(v[(0, 1)], 1.0); // 1^1
        assert_eq!(v[(0, 2)], 1.0); // 1^2
        assert_eq!(v[(1, 0)], 1.0); // 2^0
        assert_eq!(v[(1, 1)], 2.0); // 2^1
        assert_eq!(v[(1, 2)], 4.0); // 2^2
        assert_eq!(v[(2, 2)], 9.0); // 3^2
    }

    #[test]
    fn test_toeplitz() {
        let t = MatBuilder::<f64>::toeplitz(&[1.0, 2.0, 3.0], &[1.0, 4.0, 5.0]);
        assert_eq!(t[(0, 0)], 1.0);
        assert_eq!(t[(0, 1)], 4.0);
        assert_eq!(t[(0, 2)], 5.0);
        assert_eq!(t[(1, 0)], 2.0);
        assert_eq!(t[(1, 1)], 1.0);
        assert_eq!(t[(1, 2)], 4.0);
        assert_eq!(t[(2, 0)], 3.0);
        assert_eq!(t[(2, 1)], 2.0);
        assert_eq!(t[(2, 2)], 1.0);
    }

    #[test]
    fn test_circulant() {
        let c = MatBuilder::<f64>::circulant(&[1.0, 2.0, 3.0]);
        assert_eq!(c[(0, 0)], 1.0);
        assert_eq!(c[(0, 1)], 2.0);
        assert_eq!(c[(0, 2)], 3.0);
        assert_eq!(c[(1, 0)], 3.0);
        assert_eq!(c[(1, 1)], 1.0);
        assert_eq!(c[(1, 2)], 2.0);
        assert_eq!(c[(2, 0)], 2.0);
        assert_eq!(c[(2, 1)], 3.0);
        assert_eq!(c[(2, 2)], 1.0);
    }

    #[test]
    fn test_random() {
        let r = MatBuilder::<f64>::random(5, 5, 42);
        for i in 0..5 {
            for j in 0..5 {
                assert!(r[(i, j)] >= 0.0 && r[(i, j)] < 1.0);
            }
        }
    }

    #[test]
    fn test_random_spd() {
        let spd = MatBuilder::<f64>::random_spd(4, 42);
        // Should be symmetric
        for i in 0..4 {
            for j in 0..4 {
                assert!((spd[(i, j)] - spd[(j, i)]).abs() < 1e-10);
            }
        }
        // Diagonal should be positive (for SPD)
        for i in 0..4 {
            assert!(spd[(i, i)] > 0.0);
        }
    }

    #[test]
    fn test_vstack() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[3.0, 4.0], &[5.0, 6.0]]);
        let stacked = MatBuilder::vstack(&[a.as_ref(), b.as_ref()]);
        assert_eq!(stacked.nrows(), 3);
        assert_eq!(stacked.ncols(), 2);
        assert_eq!(stacked[(0, 0)], 1.0);
        assert_eq!(stacked[(1, 0)], 3.0);
        assert_eq!(stacked[(2, 1)], 6.0);
    }

    #[test]
    fn test_hstack() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0], &[2.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[3.0, 4.0], &[5.0, 6.0]]);
        let stacked = MatBuilder::hstack(&[a.as_ref(), b.as_ref()]);
        assert_eq!(stacked.nrows(), 2);
        assert_eq!(stacked.ncols(), 3);
        assert_eq!(stacked[(0, 0)], 1.0);
        assert_eq!(stacked[(0, 1)], 3.0);
        assert_eq!(stacked[(1, 2)], 6.0);
    }

    #[test]
    fn test_block_diagonal() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[5.0]]);
        let block_diag = MatBuilder::block_diagonal(&[a.as_ref(), b.as_ref()]);
        assert_eq!(block_diag.nrows(), 3);
        assert_eq!(block_diag.ncols(), 3);
        assert_eq!(block_diag[(0, 0)], 1.0);
        assert_eq!(block_diag[(1, 1)], 4.0);
        assert_eq!(block_diag[(2, 2)], 5.0);
        assert_eq!(block_diag[(0, 2)], 0.0);
        assert_eq!(block_diag[(2, 0)], 0.0);
    }

    #[test]
    fn test_linspace() {
        let m = MatBuilder::<f64>::linspace(2, 3, 0.0, 5.0);
        assert!((m[(0, 0)] - 0.0).abs() < 1e-10);
        assert!((m[(1, 0)] - 1.0).abs() < 1e-10);
        assert!((m[(0, 1)] - 2.0).abs() < 1e-10);
        assert!((m[(1, 1)] - 3.0).abs() < 1e-10);
        assert!((m[(0, 2)] - 4.0).abs() < 1e-10);
        assert!((m[(1, 2)] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_companion() {
        // Polynomial: x^2 - 3x + 2 = (x-1)(x-2)
        let c = MatBuilder::<f64>::companion(&[2.0, -3.0]);
        assert_eq!(c.nrows(), 2);
        assert_eq!(c[(0, 1)], -2.0);
        assert_eq!(c[(1, 0)], 1.0);
        assert_eq!(c[(1, 1)], 3.0);
    }

    #[test]
    fn test_col_vec() {
        let v = MatBuilder::col_vec(&[1.0, 2.0, 3.0]);
        assert_eq!(v.nrows(), 3);
        assert_eq!(v.ncols(), 1);
        assert_eq!(v[(0, 0)], 1.0);
        assert_eq!(v[(1, 0)], 2.0);
        assert_eq!(v[(2, 0)], 3.0);
    }

    #[test]
    fn test_row_vec() {
        let v = MatBuilder::row_vec(&[1.0, 2.0, 3.0]);
        assert_eq!(v.nrows(), 1);
        assert_eq!(v.ncols(), 3);
        assert_eq!(v[(0, 0)], 1.0);
        assert_eq!(v[(0, 1)], 2.0);
        assert_eq!(v[(0, 2)], 3.0);
    }
}
