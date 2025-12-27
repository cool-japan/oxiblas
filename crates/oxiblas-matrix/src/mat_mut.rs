//! Mutable matrix view type.
//!
//! `MatMut<'a, T>` is a borrowed, mutable view into a matrix with arbitrary strides.

use crate::mat_ref::MatRef;
use oxiblas_core::scalar::Scalar;

/// A mutable view into a matrix.
///
/// This type does not own its data and can represent a mutable view into any
/// contiguous or strided matrix data. It uses the reborrow pattern to prevent
/// aliasing issues.
///
/// # Reborrow Pattern
///
/// To prevent multiple mutable references to the same data, `MatMut` uses
/// the reborrow pattern. Use `rb()` for immutable reborrow and `rb_mut()`
/// for mutable reborrow.
///
/// # Example
///
/// ```
/// use oxiblas_matrix::{Mat, MatMut};
///
/// let mut m: Mat<f64> = Mat::zeros(3, 3);
/// let mut view: MatMut<'_, f64> = m.as_mut();
///
/// // Modify through the view
/// view[(1, 1)] = 5.0;
///
/// // Reborrow for nested operations
/// let sub = view.rb_mut().submatrix(0, 0, 2, 2);
/// ```
pub struct MatMut<'a, T: Scalar> {
    /// Pointer to the first element.
    ptr: *mut T,
    /// Number of rows.
    nrows: usize,
    /// Number of columns.
    ncols: usize,
    /// Stride between consecutive rows (in elements).
    row_stride: usize,
    /// Lifetime marker.
    _marker: core::marker::PhantomData<&'a mut T>,
}

impl<'a, T: Scalar> MatMut<'a, T> {
    /// Creates a new mutable matrix view from raw components.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` points to valid, initialized data
    /// - The data remains valid for the lifetime `'a`
    /// - No other references to the data exist
    /// - The strides are correct for the given dimensions
    #[inline]
    pub fn new(ptr: *mut T, nrows: usize, ncols: usize, row_stride: usize) -> Self {
        MatMut {
            ptr,
            nrows,
            ncols,
            row_stride,
            _marker: core::marker::PhantomData,
        }
    }

    /// Creates a view from a mutable slice (single column vector).
    #[inline]
    pub fn from_slice(slice: &'a mut [T]) -> Self {
        let len = slice.len();
        MatMut::new(slice.as_mut_ptr(), len, 1, 1)
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

    /// Returns a mutable pointer to the first element.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Returns a pointer to the element at (row, col).
    #[inline]
    pub fn ptr_at(&self, row: usize, col: usize) -> *const T {
        debug_assert!(row < self.nrows && col < self.ncols);
        unsafe { self.ptr.add(row + col * self.row_stride) }
    }

    /// Returns a mutable pointer to the element at (row, col).
    #[inline]
    pub fn ptr_at_mut(&mut self, row: usize, col: usize) -> *mut T {
        debug_assert!(row < self.nrows && col < self.ncols);
        unsafe { self.ptr.add(row + col * self.row_stride) }
    }

    /// Immutable reborrow - creates an immutable view.
    ///
    /// This is the key to the reborrow pattern: it creates a new reference
    /// with a shorter lifetime, allowing temporary immutable access.
    #[inline]
    pub fn rb(&self) -> MatRef<'_, T> {
        MatRef::new(self.ptr, self.nrows, self.ncols, self.row_stride)
    }

    /// Mutable reborrow - creates a new mutable view with a shorter lifetime.
    ///
    /// This allows passing the view to functions while retaining ownership.
    #[inline]
    pub fn rb_mut(&mut self) -> MatMut<'_, T> {
        MatMut::new(self.ptr, self.nrows, self.ncols, self.row_stride)
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

    /// Returns a mutable reference to the element at (row, col).
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row < self.nrows && col < self.ncols {
            Some(unsafe { &mut *self.ptr_at_mut(row, col) })
        } else {
            None
        }
    }

    /// Sets the element at (row, col).
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        assert!(row < self.nrows && col < self.ncols, "Index out of bounds");
        unsafe {
            *self.ptr_at_mut(row, col) = value;
        }
    }

    /// Returns an immutable submatrix view.
    #[inline]
    pub fn submatrix_ref(
        &self,
        row_start: usize,
        col_start: usize,
        nrows: usize,
        ncols: usize,
    ) -> MatRef<'_, T> {
        self.rb().submatrix(row_start, col_start, nrows, ncols)
    }

    /// Returns a mutable submatrix view.
    #[inline]
    pub fn submatrix(
        mut self,
        row_start: usize,
        col_start: usize,
        nrows: usize,
        ncols: usize,
    ) -> Self {
        assert!(
            row_start + nrows <= self.nrows && col_start + ncols <= self.ncols,
            "Submatrix out of bounds"
        );

        MatMut::new(
            self.ptr_at_mut(row_start, col_start),
            nrows,
            ncols,
            self.row_stride,
        )
    }

    /// Returns an immutable column view.
    #[inline]
    pub fn col_ref(&self, j: usize) -> MatRef<'_, T> {
        assert!(j < self.ncols, "Column index out of bounds");
        self.rb().col(j)
    }

    /// Returns a mutable column view.
    #[inline]
    pub fn col_mut(self, j: usize) -> Self {
        let nrows = self.nrows;
        assert!(j < self.ncols, "Column index out of bounds");
        self.submatrix(0, j, nrows, 1)
    }

    /// Returns an immutable row view.
    #[inline]
    pub fn row_ref(&self, i: usize) -> MatRef<'_, T> {
        assert!(i < self.nrows, "Row index out of bounds");
        self.rb().row(i)
    }

    /// Returns a mutable row view.
    #[inline]
    pub fn row_mut(self, i: usize) -> Self {
        let ncols = self.ncols;
        assert!(i < self.nrows, "Row index out of bounds");
        self.submatrix(i, 0, 1, ncols)
    }

    /// Fills the matrix with a value.
    pub fn fill(&mut self, value: T) {
        for j in 0..self.ncols {
            for i in 0..self.nrows {
                self.set(i, j, value);
            }
        }
    }

    /// Fills the matrix with zeros.
    pub fn fill_zero(&mut self)
    where
        T: num_traits::Zero,
    {
        self.fill(T::zero());
    }

    /// Scales the matrix by a scalar.
    pub fn scale(&mut self, alpha: T) {
        for j in 0..self.ncols {
            for i in 0..self.nrows {
                let val = unsafe { *self.ptr_at(i, j) };
                self.set(i, j, val * alpha);
            }
        }
    }

    /// Copies data from another matrix view.
    pub fn copy_from(&mut self, src: &MatRef<'_, T>) {
        assert_eq!(
            self.shape(),
            src.shape(),
            "Matrix shapes must match for copy"
        );

        for j in 0..self.ncols {
            for i in 0..self.nrows {
                self.set(i, j, src[(i, j)]);
            }
        }
    }

    /// Adds another matrix: self += alpha * other
    pub fn add_scaled(&mut self, alpha: T, other: &MatRef<'_, T>) {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Matrix shapes must match for addition"
        );

        for j in 0..self.ncols {
            for i in 0..self.nrows {
                let val = unsafe { *self.ptr_at(i, j) };
                self.set(i, j, val + alpha * other[(i, j)]);
            }
        }
    }

    /// Splits the matrix horizontally at column `mid`.
    #[inline]
    pub fn split_cols(self, mid: usize) -> (Self, Self) {
        assert!(mid <= self.ncols, "Split point out of bounds");

        let left = MatMut::new(self.ptr, self.nrows, mid, self.row_stride);

        let right_ptr = unsafe { self.ptr.add(mid * self.row_stride) };
        let right = MatMut::new(right_ptr, self.nrows, self.ncols - mid, self.row_stride);

        (left, right)
    }

    /// Splits the matrix vertically at row `mid`.
    #[inline]
    pub fn split_rows(self, mid: usize) -> (Self, Self) {
        assert!(mid <= self.nrows, "Split point out of bounds");

        let top = MatMut::new(self.ptr, mid, self.ncols, self.row_stride);

        let bottom_ptr = unsafe { self.ptr.add(mid) };
        let bottom = MatMut::new(bottom_ptr, self.nrows - mid, self.ncols, self.row_stride);

        (top, bottom)
    }

    /// Returns a mutable slice of column `j` if contiguous.
    #[inline]
    pub fn col_as_slice_mut(&mut self, j: usize) -> Option<&mut [T]> {
        if j >= self.ncols {
            return None;
        }

        let start = unsafe { self.ptr.add(j * self.row_stride) };
        Some(unsafe { core::slice::from_raw_parts_mut(start, self.nrows) })
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

    /// Swaps two rows.
    pub fn swap_rows(&mut self, i1: usize, i2: usize) {
        assert!(
            i1 < self.nrows && i2 < self.nrows,
            "Row index out of bounds"
        );
        if i1 == i2 {
            return;
        }

        for j in 0..self.ncols {
            let ptr1 = self.ptr_at_mut(i1, j);
            let ptr2 = self.ptr_at_mut(i2, j);
            unsafe {
                core::ptr::swap(ptr1, ptr2);
            }
        }
    }

    /// Swaps two columns.
    pub fn swap_cols(&mut self, j1: usize, j2: usize) {
        assert!(
            j1 < self.ncols && j2 < self.ncols,
            "Column index out of bounds"
        );
        if j1 == j2 {
            return;
        }

        for i in 0..self.nrows {
            let ptr1 = self.ptr_at_mut(i, j1);
            let ptr2 = self.ptr_at_mut(i, j2);
            unsafe {
                core::ptr::swap(ptr1, ptr2);
            }
        }
    }
}

// Safety: MatMut is Send/Sync if T is
unsafe impl<'a, T: Scalar + Send> Send for MatMut<'a, T> {}
unsafe impl<'a, T: Scalar + Sync> Sync for MatMut<'a, T> {}

impl<'a, T: Scalar> core::ops::Index<(usize, usize)> for MatMut<'a, T> {
    type Output = T;

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < self.nrows && col < self.ncols, "Index out of bounds");
        unsafe { &*self.ptr_at(row, col) }
    }
}

impl<'a, T: Scalar> core::ops::IndexMut<(usize, usize)> for MatMut<'a, T> {
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        assert!(row < self.nrows && col < self.ncols, "Index out of bounds");
        unsafe { &mut *self.ptr_at_mut(row, col) }
    }
}

impl<'a, T: Scalar + core::fmt::Debug> core::fmt::Debug for MatMut<'a, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "MatMut {}x{} {{", self.nrows, self.ncols)?;
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

#[cfg(test)]
mod tests {
    use crate::Mat;

    #[test]
    fn test_mat_mut_basic() {
        let mut m: Mat<f64> = Mat::zeros(3, 3);
        let mut view = m.as_mut();

        view[(1, 1)] = 5.0;
        view.set(0, 2, 3.0);

        assert_eq!(view[(1, 1)], 5.0);
        assert_eq!(view[(0, 2)], 3.0);
    }

    #[test]
    fn test_mat_mut_reborrow() {
        let mut m: Mat<f64> = Mat::zeros(3, 3);
        let mut view = m.as_mut();

        // Reborrow for reading
        {
            let immut = view.rb();
            assert_eq!(immut[(0, 0)], 0.0);
        }

        // Reborrow for writing
        {
            let mut sub = view.rb_mut().submatrix(0, 0, 2, 2);
            sub[(0, 0)] = 1.0;
        }

        assert_eq!(view[(0, 0)], 1.0);
    }

    #[test]
    fn test_mat_mut_fill() {
        let mut m: Mat<f64> = Mat::zeros(3, 3);
        let mut view = m.as_mut();

        view.fill(7.0);

        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(view[(i, j)], 7.0);
            }
        }
    }

    #[test]
    fn test_mat_mut_copy_from() {
        let src: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let mut dst: Mat<f64> = Mat::zeros(2, 2);

        dst.as_mut().copy_from(&src.as_ref());

        assert_eq!(dst[(0, 0)], 1.0);
        assert_eq!(dst[(1, 1)], 4.0);
    }

    #[test]
    fn test_mat_mut_swap() {
        let mut m: Mat<f64> =
            Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);
        let mut view = m.as_mut();

        view.swap_rows(0, 2);
        assert_eq!(view[(0, 0)], 7.0);
        assert_eq!(view[(2, 0)], 1.0);

        view.swap_cols(0, 1);
        assert_eq!(view[(0, 0)], 8.0);
        assert_eq!(view[(0, 1)], 7.0);
    }

    #[test]
    fn test_mat_mut_split() {
        let mut m: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]]);

        let (mut left, mut right) = m.as_mut().split_cols(2);

        left[(0, 0)] = 10.0;
        right[(0, 0)] = 20.0;

        // Check original matrix was modified
        assert_eq!(m[(0, 0)], 10.0);
        assert_eq!(m[(0, 2)], 20.0);
    }

    #[test]
    fn test_mat_mut_scale() {
        let mut m: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        m.as_mut().scale(2.0);

        assert_eq!(m[(0, 0)], 2.0);
        assert_eq!(m[(1, 1)], 8.0);
    }
}
