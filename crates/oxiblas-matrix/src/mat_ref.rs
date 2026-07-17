//! Immutable matrix view type.
//!
//! `MatRef<'a, T>` is a borrowed, immutable view into a matrix with arbitrary
//! strides. It can represent contiguous column-major matrices, submatrices, and
//! genuine zero-copy transposed / strided views.
//!
//! # Layout invariants
//!
//! A [`MatRef`] locates the element at logical position `(row, col)` at the
//! byte-agnostic element offset
//!
//! ```text
//! offset(row, col) = row * col_stride + col * row_stride
//! ```
//!
//! relative to `ptr`. The two stride fields have a fixed, load-bearing meaning:
//!
//! - `row_stride` is the number of elements between `(i, j)` and `(i, j + 1)`
//!   — i.e. how far you advance to move **one column to the right**. For
//!   standard column-major storage this equals the *leading dimension* (`lda`),
//!   which is why [`MatRef::row_stride`] returns it and the whole BLAS layer
//!   relies on that value being the `lda`.
//! - `col_stride` is the number of elements between `(i, j)` and `(i + 1, j)`
//!   — i.e. how far you advance to move **one row down**. For standard
//!   column-major storage this is `1`.
//!
//! This naming matches the sibling owned/borrowed types (`Mat`, `MatMut`,
//! `MmapMat`) where the `row_stride` field is likewise the leading dimension.
//! The apparent asymmetry (the `row` index is scaled by `col_stride` and the
//! `col` index by `row_stride`) is intentional and must be preserved: swapping
//! `(nrows, ncols)` together with `(row_stride, col_stride)` yields a valid
//! transposed view over exactly the same memory, which is how
//! [`MatRef::transpose`] achieves an O(1), allocation-free transpose. Downstream
//! crates (notably the GEMM path in `oxiblas-blas`) can therefore read both
//! strides and honor transposed operands without materializing a copy.
//!
//! Because a `MatRef` only ever hands out shared references, overlapping strides
//! (e.g. `col_stride == 0`) are memory-safe even though they are rarely useful.
//! The mutable counterpart `MatMut` has stricter aliasing requirements.

use oxiblas_core::scalar::Scalar;

/// An immutable view into a matrix.
///
/// This type does not own its data and can represent a view into any
/// contiguous or strided matrix data. It supports independent row and
/// column strides, making it suitable for submatrices and zero-copy
/// transposed views. See the [module documentation](self) for the exact
/// layout invariants.
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
    /// Elements between `(i, j)` and `(i, j + 1)` (advance one column).
    ///
    /// For column-major storage this is the leading dimension (`lda`).
    row_stride: usize,
    /// Elements between `(i, j)` and `(i + 1, j)` (advance one row).
    ///
    /// For column-major storage this is `1`.
    col_stride: usize,
    /// Lifetime marker.
    _marker: core::marker::PhantomData<&'a T>,
}

impl<'a, T: Scalar> MatRef<'a, T> {
    /// Creates a new column-major matrix view from raw components.
    ///
    /// The resulting view uses `row_stride` as the column-to-column distance
    /// (the leading dimension) and an implicit row-to-row distance of `1`,
    /// matching standard column-major (Fortran) storage. Use
    /// [`MatRef::new_strided`] to supply both strides explicitly.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` is non-null, well-aligned, and points to valid, initialized data
    /// - The data remains valid and immutable for the lifetime `'a`
    /// - For every `0 <= i < nrows` and `0 <= j < ncols`, the offset
    ///   `i + j * row_stride` is in bounds of the underlying allocation
    ///
    /// Prefer the safe [`MatRef::from_column_major`] / [`MatRef::from_strided`]
    /// constructors whenever a backing slice is available.
    #[inline]
    pub unsafe fn new(ptr: *const T, nrows: usize, ncols: usize, row_stride: usize) -> Self {
        MatRef {
            ptr,
            nrows,
            ncols,
            row_stride,
            col_stride: 1,
            _marker: core::marker::PhantomData,
        }
    }

    /// Creates a new matrix view from raw components with explicit strides.
    ///
    /// See the [module documentation](self) for the meaning of `row_stride`
    /// (advance one column) and `col_stride` (advance one row).
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` is non-null, well-aligned, and points to valid, initialized data
    /// - The data remains valid and immutable for the lifetime `'a`
    /// - For every `0 <= i < nrows` and `0 <= j < ncols`, the offset
    ///   `i * col_stride + j * row_stride` is in bounds of the underlying
    ///   allocation
    #[inline]
    pub unsafe fn new_strided(
        ptr: *const T,
        nrows: usize,
        ncols: usize,
        row_stride: usize,
        col_stride: usize,
    ) -> Self {
        MatRef {
            ptr,
            nrows,
            ncols,
            row_stride,
            col_stride,
            _marker: core::marker::PhantomData,
        }
    }

    /// Creates a view from a slice (single column vector).
    ///
    /// The view has shape `(slice.len(), 1)`. This never fails: a column vector
    /// always fits within its backing slice.
    #[inline]
    pub fn from_slice(slice: &'a [T]) -> Self {
        // A `len x 1` column vector accesses offsets `0..len`, all within
        // `slice`, so the validated constructor can never reject it.
        Self::from_strided(slice, slice.len(), 1, 1, 1)
            .expect("column-vector layout always fits its backing slice")
    }

    /// Creates a contiguous column-major view from a slice, validating that the
    /// slice is large enough.
    ///
    /// The leading dimension is taken to be `nrows` (no inter-column padding).
    /// Returns `None` if `slice.len() < nrows * ncols`.
    #[inline]
    pub fn from_column_major(slice: &'a [T], nrows: usize, ncols: usize) -> Option<Self> {
        Self::from_strided(slice, nrows, ncols, nrows, 1)
    }

    /// Creates a strided view from a slice, validating that every element the
    /// view can address lies within the slice.
    ///
    /// `row_stride` is the column-to-column distance and `col_stride` the
    /// row-to-row distance (see the [module documentation](self)). Returns
    /// `None` if any addressable element would fall outside `slice` (or if the
    /// offset computation overflows).
    ///
    /// Because the returned view is immutable, overlapping strides are
    /// permitted; only the largest reachable offset is bounds-checked.
    #[inline]
    pub fn from_strided(
        slice: &'a [T],
        nrows: usize,
        ncols: usize,
        row_stride: usize,
        col_stride: usize,
    ) -> Option<Self> {
        if nrows == 0 || ncols == 0 {
            // No element is ever dereferenced; the pointer only needs valid
            // provenance, which `slice.as_ptr()` provides.
            // SAFETY: empty view; `ptr` has valid provenance and no offset is
            // ever formed for an out-of-range index.
            return Some(unsafe {
                Self::new_strided(slice.as_ptr(), nrows, ncols, row_stride, col_stride)
            });
        }

        // Strides are non-negative, so the maximum reachable offset is at the
        // far corner `(nrows - 1, ncols - 1)`. Checking it bounds every access.
        let max_row_offset = (nrows - 1).checked_mul(col_stride)?;
        let max_col_offset = (ncols - 1).checked_mul(row_stride)?;
        let max_offset = max_row_offset.checked_add(max_col_offset)?;
        if max_offset < slice.len() {
            // SAFETY: `max_offset < slice.len()` and offsets are monotonic in
            // both indices, so every `(i, j)` maps inside `slice`.
            Some(unsafe {
                Self::new_strided(slice.as_ptr(), nrows, ncols, row_stride, col_stride)
            })
        } else {
            None
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

    /// Returns the shape as (nrows, ncols).
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Returns the row stride: the number of elements between consecutive
    /// columns (`(i, j)` to `(i, j + 1)`).
    ///
    /// For column-major storage this is the leading dimension (`lda`).
    #[inline]
    pub fn row_stride(&self) -> usize {
        self.row_stride
    }

    /// Returns the column stride: the number of elements between consecutive
    /// rows (`(i, j)` to `(i + 1, j)`).
    ///
    /// For column-major storage this is `1`; for a transposed view it is the
    /// original leading dimension.
    #[inline]
    pub fn col_stride(&self) -> usize {
        self.col_stride
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
        unsafe { self.ptr.add(row * self.col_stride + col * self.row_stride) }
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
    /// Both strides are preserved, so submatrices of transposed / strided views
    /// remain correct. Empty submatrices at the boundary (e.g.
    /// `submatrix(nrows, 0, 0, ncols)`) are legal and do not panic.
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
        // Overflow-safe bounds check: each `<= self.nrows - nrows` term is only
        // evaluated after the preceding `nrows <= self.nrows` guard (via `&&`
        // short-circuiting), so the subtraction can never underflow and
        // `row_start + nrows` can never wrap around `usize`.
        assert!(
            nrows <= self.nrows
                && ncols <= self.ncols
                && row_start <= self.nrows - nrows
                && col_start <= self.ncols - ncols,
            "Submatrix out of bounds"
        );

        // For an empty view no element is addressable, so reuse `self.ptr`
        // rather than forming a (possibly one-past-the-end) offset pointer that
        // would trip `ptr_at`'s debug assertion.
        let ptr = if nrows == 0 || ncols == 0 {
            self.ptr
        } else {
            self.ptr_at(row_start, col_start)
        };

        // SAFETY: bounds were validated above; the offset pointer stays within
        // the parent's allocation and the parent's strides remain valid for the
        // narrowed dimensions.
        unsafe { MatRef::new_strided(ptr, nrows, ncols, self.row_stride, self.col_stride) }
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
            // Moving to the next diagonal entry advances one row and one column.
            stride: self.row_stride + self.col_stride,
            _marker: core::marker::PhantomData,
        }
    }

    /// Returns a transposed view (logical transpose, no data movement).
    ///
    /// This is a genuine O(1), allocation-free view: the returned `MatRef`
    /// shares `self`'s data pointer and swaps `(nrows, ncols)` together with
    /// `(row_stride, col_stride)`. Indexing the transpose at `(i, j)` therefore
    /// returns the same element as indexing `self` at `(j, i)`.
    #[inline]
    pub fn transpose(&self) -> MatRef<'a, T> {
        // SAFETY: swapping the dimensions and strides re-labels the identical
        // set of in-bounds offsets: every `(i, j)` of the transpose maps to the
        // in-bounds `(j, i)` of `self`, so all reachable offsets stay valid.
        unsafe {
            MatRef::new_strided(
                self.ptr,
                self.ncols,
                self.nrows,
                self.col_stride,
                self.row_stride,
            )
        }
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

    /// Returns column `j` as a contiguous slice, or `None` if the column is not
    /// contiguous in memory.
    ///
    /// A column is contiguous exactly when the row-to-row distance is `1`, i.e.
    /// `col_stride() == 1`. This holds for standard column-major matrices and
    /// their column-major submatrices, but not for transposed or otherwise
    /// row-strided views, for which this method returns `None`.
    #[inline]
    pub fn col_as_slice(&self, j: usize) -> Option<&'a [T]> {
        if j >= self.ncols {
            return None;
        }

        // The elements of a column are contiguous only when advancing one row
        // moves exactly one element.
        if self.col_stride != 1 {
            return None;
        }

        // SAFETY: `j < ncols` and `col_stride == 1`, so the `nrows` elements
        // starting at `ptr + j * row_stride` are contiguous and in bounds by the
        // type's construction invariant.
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

#[cfg(test)]
mod tests {
    use super::MatRef;
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

    // ------------------------------------------------------------------
    // Regression tests for the audited findings.
    // ------------------------------------------------------------------

    /// Finding #2: legal empty boundary submatrices must not panic (this test
    /// runs in debug where the old `ptr_at` debug-assert would fire).
    #[test]
    fn test_submatrix_empty_boundary_no_panic() {
        let m: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let view = m.as_ref();
        let (nr, nc) = view.shape();

        // Zero rows at the bottom edge.
        let bottom = view.submatrix(nr, 0, 0, nc);
        assert_eq!(bottom.shape(), (0, nc));
        assert!(bottom.is_empty());
        assert!(bottom.get(0, 0).is_none());

        // Zero cols at the right edge.
        let right = view.submatrix(0, nc, nr, 0);
        assert_eq!(right.shape(), (nr, 0));
        assert!(right.is_empty());

        // Fully empty corner.
        let corner = view.submatrix(nr, nc, 0, 0);
        assert_eq!(corner.shape(), (0, 0));
        assert!(corner.is_empty());
    }

    /// Finding #3: the bounds check must not be defeatable by `usize` wraparound.
    /// With the wrapping `row_start + nrows` check this call slipped through and
    /// (in release) produced an out-of-bounds view; the overflow-safe check
    /// rejects it with the documented message.
    #[test]
    #[should_panic(expected = "Submatrix out of bounds")]
    fn test_submatrix_wraparound_rejected() {
        let m: Mat<f64> = Mat::zeros(4, 4);
        let view = m.as_ref();
        // row_start + nrows == usize::MAX + 1 wraps to 0 (<= 4) under the old
        // check but is rejected by the underflow-safe reformulation.
        let _ = view.submatrix(usize::MAX - 3, 0, 4, 1);
    }

    /// Finding #4: `col_as_slice` must report non-contiguous columns as `None`.
    #[test]
    fn test_col_as_slice_contiguity() {
        // Contiguous column-major view -> Some with correct values.
        let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2, col-major
        let cm = MatRef::from_column_major(&data, 3, 2).expect("valid layout");
        assert_eq!(cm.col_as_slice(0), Some(&data[0..3]));
        assert_eq!(cm.col_as_slice(1), Some(&data[3..6]));
        assert_eq!(cm.col_as_slice(2), None); // out-of-range column

        // Transposed view is row-strided -> columns are not contiguous.
        let t = cm.transpose();
        assert_eq!(t.shape(), (2, 3));
        assert!(t.col_stride() != 1);
        assert_eq!(t.col_as_slice(0), None);
        assert_eq!(t.col_as_slice(1), None);
    }

    /// Finding #5(a): transpose is a zero-copy view (shares the data pointer,
    /// no allocation).
    #[test]
    fn test_transpose_is_zero_copy() {
        let m: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let view = m.as_ref();
        let t = view.transpose();

        // Same backing pointer => no data was copied.
        assert_eq!(t.as_ptr(), view.as_ptr());
        // Strides and dims are swapped.
        assert_eq!(t.shape(), (view.ncols(), view.nrows()));
        assert_eq!(t.row_stride(), view.col_stride());
        assert_eq!(t.col_stride(), view.row_stride());
    }

    /// Finding #5(b): indexing a transposed view matches the original with
    /// swapped coordinates, and transpose is an involution.
    #[test]
    fn test_transpose_indexing_matches_original() {
        let m: Mat<f64> = Mat::from_rows(&[
            &[1.0, 2.0, 3.0, 4.0],
            &[5.0, 6.0, 7.0, 8.0],
            &[9.0, 10.0, 11.0, 12.0],
        ]);
        let view = m.as_ref();
        let t = view.transpose();

        for i in 0..view.nrows() {
            for j in 0..view.ncols() {
                assert_eq!(t[(j, i)], view[(i, j)]);
            }
        }

        // Transpose of transpose recovers the original view exactly.
        let tt = t.transpose();
        assert_eq!(tt.shape(), view.shape());
        assert_eq!(tt.as_ptr(), view.as_ptr());
        assert_eq!(tt.row_stride(), view.row_stride());
        assert_eq!(tt.col_stride(), view.col_stride());
    }

    /// Finding #5(c): GEMM-style row/column iteration over a transposed view
    /// produces correct results (each column of the transpose is a row of the
    /// original), including a submatrix taken from the transposed view.
    #[test]
    fn test_transpose_gemm_style_iteration() {
        let m: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let view = m.as_ref();
        let t = view.transpose(); // 3x2

        // Column j of the transpose == row j of the original.
        let expected_cols = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        for (j, tcol) in t.cols().enumerate() {
            assert_eq!(tcol.shape(), (3, 1));
            for i in 0..3 {
                assert_eq!(tcol[(i, 0)], expected_cols[j][i]);
            }
        }

        // Submatrix of a transposed view must honor both strides.
        let sub = t.submatrix(1, 0, 2, 2); // rows 1..3, all cols of the 3x2 transpose
        assert_eq!(sub.shape(), (2, 2));
        assert_eq!(sub[(0, 0)], view[(0, 1)]);
        assert_eq!(sub[(0, 1)], view[(1, 1)]);
        assert_eq!(sub[(1, 0)], view[(0, 2)]);
        assert_eq!(sub[(1, 1)], view[(1, 2)]);
    }

    /// Safe constructors validate their backing slice.
    #[test]
    fn test_safe_constructors_validate() {
        let data = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];

        assert!(MatRef::from_column_major(&data, 3, 2).is_some());
        assert!(MatRef::from_column_major(&data, 3, 3).is_none()); // needs 9 elems

        // Padded leading dimension: 2x2 with lda=3 needs offset (1)+ (1*3)=4 < 6.
        assert!(MatRef::from_strided(&data, 2, 2, 3, 1).is_some());
        // lda=4 pushes the far corner to offset 1 + 3 = 4? -> (1)+(1*4)=5 < 6 ok.
        assert!(MatRef::from_strided(&data, 2, 2, 4, 1).is_some());
        // lda=5 -> far corner offset 1 + 5 = 6, not < 6 -> rejected.
        assert!(MatRef::from_strided(&data, 2, 2, 5, 1).is_none());

        // Empty views are always valid regardless of strides.
        assert!(MatRef::from_strided(&data, 0, 5, 999, 999).is_some());
        assert!(MatRef::<f64>::from_strided(&[], 0, 0, 1, 1).is_some());
    }

    /// `from_slice` yields a column vector view over the whole slice.
    #[test]
    fn test_from_slice_column_vector() {
        let data = [10.0_f64, 20.0, 30.0];
        let v = MatRef::from_slice(&data);
        assert_eq!(v.shape(), (3, 1));
        assert_eq!(v[(0, 0)], 10.0);
        assert_eq!(v[(2, 0)], 30.0);
        assert_eq!(v.col_as_slice(0), Some(&data[..]));
    }
}
