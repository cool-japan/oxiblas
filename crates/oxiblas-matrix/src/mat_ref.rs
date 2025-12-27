//! Immutable matrix view type.
//!
//! `MatRef<'a, T>` is a borrowed, immutable view into a matrix with arbitrary strides.

use oxiblas_core::scalar::Scalar;

/// An immutable view into a matrix.
///
/// This type does not own its data and can represent a view into any
/// contiguous or strided matrix data. It supports arbitrary row and
/// column strides, making it suitable for submatrices and transposed views.
///
/// # Lifetime
///
/// The lifetime `'a` ensures that the view does not outlive the data it
/// references.
///
/// # Example
///
/// ```
/// use oxiblas_matrix::{Mat, MatRef};
///
/// let m: Mat<f64> = Mat::zeros(4, 4);
/// let view: MatRef<'_, f64> = m.as_ref();
///
/// // Create a submatrix view
/// let sub = view.submatrix(1, 1, 2, 2);
/// assert_eq!(sub.shape(), (2, 2));
/// ```
#[derive(Copy, Clone)]
pub struct MatRef<'a, T: Scalar> {
    /// Pointer to the first element.
    ptr: *const T,
    /// Number of rows.
    nrows: usize,
    /// Number of columns.
    ncols: usize,
    /// Stride between consecutive rows (in elements).
    row_stride: usize,
    /// Lifetime marker.
    _marker: core::marker::PhantomData<&'a T>,
}

impl<'a, T: Scalar> MatRef<'a, T> {
    /// Creates a new matrix view from raw components.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` points to valid, initialized data
    /// - The data remains valid for the lifetime `'a`
    /// - The strides are correct for the given dimensions
    #[inline]
    pub fn new(ptr: *const T, nrows: usize, ncols: usize, row_stride: usize) -> Self {
        MatRef {
            ptr,
            nrows,
            ncols,
            row_stride,
            _marker: core::marker::PhantomData,
        }
    }

    /// Creates a view from a slice (single column vector).
    #[inline]
    pub fn from_slice(slice: &'a [T]) -> Self {
        MatRef::new(slice.as_ptr(), slice.len(), 1, 1)
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

    /// Returns the shape as (nrows, ncols).
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Returns the row stride.
    #[inline]
    pub fn row_stride(&self) -> usize {
        self.row_stride
    }

    /// Returns a pointer to the first element.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns a pointer to the element at (row, col).
    #[inline]
    pub fn ptr_at(&self, row: usize, col: usize) -> *const T {
        debug_assert!(row < self.nrows && col < self.ncols);
        unsafe { self.ptr.add(row + col * self.row_stride) }
    }

    /// Returns a reference to the element at (row, col).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.nrows && col < self.ncols {
            Some(unsafe { &*self.ptr_at(row, col) })
        } else {
            None
        }
    }

    /// Returns a reference to the element at (row, col) without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `row < nrows` and `col < ncols`.
    #[inline]
    pub unsafe fn get_unchecked(&self, row: usize, col: usize) -> &T {
        &*self.ptr_at(row, col)
    }

    /// Returns a submatrix view.
    ///
    /// # Panics
    ///
    /// Panics if the submatrix extends beyond the matrix bounds.
    #[inline]
    pub fn submatrix(
        &self,
        row_start: usize,
        col_start: usize,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        assert!(
            row_start + nrows <= self.nrows && col_start + ncols <= self.ncols,
            "Submatrix out of bounds"
        );

        MatRef::new(
            self.ptr_at(row_start, col_start),
            nrows,
            ncols,
            self.row_stride,
        )
    }

    /// Returns a column view.
    #[inline]
    pub fn col(&self, j: usize) -> Self {
        assert!(j < self.ncols, "Column index out of bounds");
        self.submatrix(0, j, self.nrows, 1)
    }

    /// Returns a row view.
    #[inline]
    pub fn row(&self, i: usize) -> Self {
        assert!(i < self.nrows, "Row index out of bounds");
        self.submatrix(i, 0, 1, self.ncols)
    }

    /// Returns the diagonal as a column vector view.
    ///
    /// For non-square matrices, returns the shorter diagonal.
    #[inline]
    pub fn diagonal(&self) -> DiagRef<'a, T> {
        let len = self.nrows.min(self.ncols);
        DiagRef {
            ptr: self.ptr,
            len,
            stride: self.row_stride + 1, // Move down and right
            _marker: core::marker::PhantomData,
        }
    }

    /// Returns a transposed view (logical transpose, no data movement).
    #[inline]
    pub fn transpose(&self) -> TransposeRef<'a, T> {
        TransposeRef { inner: *self }
    }

    /// Returns true if the matrix is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nrows == 0 || self.ncols == 0
    }

    /// Returns true if this is a column vector.
    #[inline]
    pub fn is_column(&self) -> bool {
        self.ncols == 1
    }

    /// Returns true if this is a row vector.
    #[inline]
    pub fn is_row(&self) -> bool {
        self.nrows == 1
    }

    /// Returns true if this is a square matrix.
    #[inline]
    pub fn is_square(&self) -> bool {
        self.nrows == self.ncols
    }

    /// Returns a slice of column `j` if the column is contiguous.
    ///
    /// For column-major matrices with row_stride == nrows, columns are contiguous.
    #[inline]
    pub fn col_as_slice(&self, j: usize) -> Option<&'a [T]> {
        if j >= self.ncols {
            return None;
        }

        // Columns are always contiguous in column-major storage
        let start = unsafe { self.ptr.add(j * self.row_stride) };
        Some(unsafe { core::slice::from_raw_parts(start, self.nrows) })
    }

    /// Iterates over columns.
    #[inline]
    pub fn cols(&self) -> impl Iterator<Item = MatRef<'a, T>> + '_ {
        (0..self.ncols).map(move |j| self.col(j))
    }

    /// Iterates over rows.
    #[inline]
    pub fn rows(&self) -> impl Iterator<Item = MatRef<'a, T>> + '_ {
        (0..self.nrows).map(move |i| self.row(i))
    }

    /// Splits the matrix horizontally at column `mid`.
    #[inline]
    pub fn split_cols(&self, mid: usize) -> (Self, Self) {
        assert!(mid <= self.ncols, "Split point out of bounds");
        (
            self.submatrix(0, 0, self.nrows, mid),
            self.submatrix(0, mid, self.nrows, self.ncols - mid),
        )
    }

    /// Splits the matrix vertically at row `mid`.
    #[inline]
    pub fn split_rows(&self, mid: usize) -> (Self, Self) {
        assert!(mid <= self.nrows, "Split point out of bounds");
        (
            self.submatrix(0, 0, mid, self.ncols),
            self.submatrix(mid, 0, self.nrows - mid, self.ncols),
        )
    }
}

// Safety: MatRef is Send/Sync if T is
unsafe impl<'a, T: Scalar + Send> Send for MatRef<'a, T> {}
unsafe impl<'a, T: Scalar + Sync> Sync for MatRef<'a, T> {}

impl<'a, T: Scalar> core::ops::Index<(usize, usize)> for MatRef<'a, T> {
    type Output = T;

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < self.nrows && col < self.ncols, "Index out of bounds");
        unsafe { &*self.ptr_at(row, col) }
    }
}

impl<'a, T: Scalar + core::fmt::Debug> core::fmt::Debug for MatRef<'a, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "MatRef {}x{} {{", self.nrows, self.ncols)?;
        for i in 0..self.nrows.min(10) {
            write!(f, "  [")?;
            for j in 0..self.ncols.min(10) {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:?}", self[(i, j)])?;
            }
            if self.ncols > 10 {
                write!(f, ", ...")?;
            }
            writeln!(f, "]")?;
        }
        if self.nrows > 10 {
            writeln!(f, "  ...")?;
        }
        write!(f, "}}")
    }
}

/// A view into the diagonal of a matrix.
#[derive(Copy, Clone)]
pub struct DiagRef<'a, T: Scalar> {
    ptr: *const T,
    len: usize,
    stride: usize,
    _marker: core::marker::PhantomData<&'a T>,
}

impl<'a, T: Scalar> DiagRef<'a, T> {
    /// Returns the length of the diagonal.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the diagonal is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the element at index `i`.
    #[inline]
    pub fn get(&self, i: usize) -> Option<&T> {
        if i < self.len {
            Some(unsafe { &*self.ptr.add(i * self.stride) })
        } else {
            None
        }
    }
}

impl<'a, T: Scalar> core::ops::Index<usize> for DiagRef<'a, T> {
    type Output = T;

    #[inline]
    fn index(&self, i: usize) -> &Self::Output {
        assert!(i < self.len, "Index out of bounds");
        unsafe { &*self.ptr.add(i * self.stride) }
    }
}

/// A transposed view of a matrix.
#[derive(Copy, Clone)]
pub struct TransposeRef<'a, T: Scalar> {
    inner: MatRef<'a, T>,
}

impl<'a, T: Scalar> TransposeRef<'a, T> {
    /// Returns the number of rows (columns of the original).
    #[inline]
    #[allow(clippy::misnamed_getters)] // Intentional: transpose swaps rows/cols
    pub fn nrows(&self) -> usize {
        self.inner.ncols
    }

    /// Returns the number of columns (rows of the original).
    #[inline]
    #[allow(clippy::misnamed_getters)] // Intentional: transpose swaps rows/cols
    pub fn ncols(&self) -> usize {
        self.inner.nrows
    }

    /// Returns the shape.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.inner.ncols, self.inner.nrows)
    }

    /// Returns the element at (row, col) in the transposed view.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.inner.get(col, row)
    }
}

impl<'a, T: Scalar> core::ops::Index<(usize, usize)> for TransposeRef<'a, T> {
    type Output = T;

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.inner[(col, row)]
    }
}

#[cfg(test)]
mod tests {
    use crate::Mat;

    #[test]
    fn test_mat_ref_basic() {
        let m: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let view = m.as_ref();

        assert_eq!(view.shape(), (2, 3));
        assert_eq!(view[(0, 0)], 1.0);
        assert_eq!(view[(1, 2)], 6.0);
    }

    #[test]
    fn test_mat_ref_submatrix() {
        let m: Mat<f64> = Mat::from_rows(&[
            &[1.0, 2.0, 3.0, 4.0],
            &[5.0, 6.0, 7.0, 8.0],
            &[9.0, 10.0, 11.0, 12.0],
        ]);

        let sub = m.as_ref().submatrix(1, 1, 2, 2);
        assert_eq!(sub.shape(), (2, 2));
        assert_eq!(sub[(0, 0)], 6.0);
        assert_eq!(sub[(1, 1)], 11.0);
    }

    #[test]
    fn test_mat_ref_col_row() {
        let m: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let view = m.as_ref();

        let col1 = view.col(1);
        assert_eq!(col1.shape(), (2, 1));
        assert_eq!(col1[(0, 0)], 2.0);
        assert_eq!(col1[(1, 0)], 5.0);

        let row0 = view.row(0);
        assert_eq!(row0.shape(), (1, 3));
        assert_eq!(row0[(0, 1)], 2.0);
    }

    #[test]
    fn test_mat_ref_diagonal() {
        let m: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        let diag = m.as_ref().diagonal();
        assert_eq!(diag.len(), 3);
        assert_eq!(diag[0], 1.0);
        assert_eq!(diag[1], 5.0);
        assert_eq!(diag[2], 9.0);
    }

    #[test]
    fn test_mat_ref_transpose() {
        let m: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let t = m.as_ref().transpose();

        assert_eq!(t.shape(), (3, 2));
        assert_eq!(t[(0, 0)], 1.0);
        assert_eq!(t[(1, 0)], 2.0);
        assert_eq!(t[(0, 1)], 4.0);
        assert_eq!(t[(2, 1)], 6.0);
    }

    #[test]
    fn test_mat_ref_split() {
        let m: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]]);
        let view = m.as_ref();

        let (left, right) = view.split_cols(2);
        assert_eq!(left.shape(), (2, 2));
        assert_eq!(right.shape(), (2, 2));
        assert_eq!(left[(0, 1)], 2.0);
        assert_eq!(right[(0, 0)], 3.0);

        let (top, bottom) = view.split_rows(1);
        assert_eq!(top.shape(), (1, 4));
        assert_eq!(bottom.shape(), (1, 4));
        assert_eq!(top[(0, 2)], 3.0);
        assert_eq!(bottom[(0, 2)], 7.0);
    }
}
