//! Mutable matrix view type.
//!
//! `MatMut<'a, T>` is a borrowed, mutable view into a column-major matrix: the
//! element `(row, col)` lives at element offset `row + col * row_stride` from
//! the view's pointer (rows within a column are contiguous, and `row_stride`
//! is the column-to-column distance, i.e. the leading dimension).
//!
//! # Pointer arithmetic
//!
//! As for [`MatRef`], the only offsets a `MatMut` method forms from its
//! pointer with in-bounds pointer arithmetic are the element offsets for
//! `row < nrows` and `col < ncols`. An empty view addresses no element, so its
//! stride is unconstrained: sub-views of it reuse its base pointer.
//! [`MatMut::ptr_at`] and [`MatMut::ptr_at_mut`] accept any indices because
//! they use wrapping arithmetic and never dereference.
//!
//! # Aliasing
//!
//! When `row_stride < nrows` and `ncols > 1`, distinct indices address the
//! same element. Every safe constructor rejects such a layout, but
//! [`MatMut::new`] permits it; [`MatMut::split_cols`] and
//! [`MatMut::split_rows`] therefore check for it and panic instead of
//! returning two mutable views that share an element.

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
    /// Creates a new column-major mutable matrix view from raw components.
    ///
    /// The resulting view uses `row_stride` as the column-to-column distance
    /// (the leading dimension) and an implicit row-to-row distance of `1`,
    /// matching standard column-major (Fortran) storage.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` is non-null and well-aligned (even when the view is empty), and
    ///   points to valid, initialized data
    /// - The data remains valid and *exclusively* borrowed for the lifetime `'a`
    /// - No other reference to the data exists for that lifetime
    /// - For every `0 <= i < nrows` and `0 <= j < ncols`, the offset
    ///   `i + j * row_stride` is in bounds of the underlying allocation
    ///
    /// These element offsets are the only ones the view's methods form from
    /// `ptr` (see the [module documentation](self#pointer-arithmetic)): an
    /// empty view places no constraint on `row_stride`, and
    /// [`ptr_at`](Self::ptr_at) / [`ptr_at_mut`](Self::ptr_at_mut) only compute
    /// a pointer (with wrapping arithmetic) without dereferencing it. The
    /// element accessors (`get`, `get_mut`, `set`, `Index`, `IndexMut`,
    /// `col_as_slice_mut`, ...) read and **write** through `ptr` at those
    /// offsets, so violating the last bullet is an out-of-bounds write.
    ///
    /// `row_stride < nrows` is permitted, although distinct indices then share
    /// an element when `ncols > 1`; [`split_cols`](Self::split_cols) and
    /// [`split_rows`](Self::split_rows) panic rather than split such a view
    /// into two overlapping halves (see the [module documentation](self#aliasing)).
    ///
    /// Prefer the safe [`MatMut::from_column_major`] / [`MatMut::from_strided`]
    /// constructors whenever a backing slice is available.
    #[inline]
    pub unsafe fn new(ptr: *mut T, nrows: usize, ncols: usize, row_stride: usize) -> Self {
        MatMut {
            ptr,
            nrows,
            ncols,
            row_stride,
            _marker: core::marker::PhantomData,
        }
    }

    /// Creates a view from a mutable slice (single column vector).
    ///
    /// The view has shape `(slice.len(), 1)`. This never fails: a column vector
    /// always fits within its backing slice.
    #[inline]
    pub fn from_slice(slice: &'a mut [T]) -> Self {
        let len = slice.len();
        // SAFETY: a `len x 1` column vector addresses offsets `0..len`, all
        // within the exclusive slice, whose pointer is non-null, aligned and
        // initialized for `'a`.
        unsafe { MatMut::new(slice.as_mut_ptr(), len, 1, 1) }
    }

    /// Creates a contiguous column-major mutable view over a slice, validating
    /// that the slice is large enough.
    ///
    /// The leading dimension is taken to be `nrows` (no inter-column padding).
    /// Returns `None` if `slice.len() < nrows * ncols`.
    #[inline]
    pub fn from_column_major(slice: &'a mut [T], nrows: usize, ncols: usize) -> Option<Self> {
        Self::from_strided(slice, nrows, ncols, nrows)
    }

    /// Creates a column-major mutable view with an explicit leading dimension,
    /// validating that every element the view can address lies within `slice`.
    ///
    /// For a non-empty view, returns `None` if any addressable element would
    /// fall outside `slice` (or if the offset computation overflows), and also
    /// if `row_stride` is smaller than `nrows`: a mutable view must not alias
    /// itself, and a short stride would make distinct `(i, j)` pairs resolve to
    /// the same element.
    ///
    /// An empty view (`nrows == 0` or `ncols == 0`) addresses no element, so
    /// it is accepted with any `row_stride`. That is sound because no method of
    /// `MatMut` forms a stride-derived offset for an empty view (see the
    /// [module documentation](self#pointer-arithmetic)).
    #[inline]
    pub fn from_strided(
        slice: &'a mut [T],
        nrows: usize,
        ncols: usize,
        row_stride: usize,
    ) -> Option<Self> {
        if nrows == 0 || ncols == 0 {
            // SAFETY: an empty view addresses no element, so the element-offset
            // requirement of `MatMut::new` holds vacuously for any stride;
            // `slice.as_mut_ptr()` is non-null and aligned even for an empty
            // slice, and `slice` is borrowed exclusively for `'a`.
            return Some(unsafe { Self::new(slice.as_mut_ptr(), nrows, ncols, row_stride) });
        }

        if row_stride < nrows {
            return None;
        }

        // The maximum reachable offset is the far corner `(nrows-1, ncols-1)`.
        let max_offset = (ncols - 1)
            .checked_mul(row_stride)?
            .checked_add(nrows - 1)?;
        if max_offset < slice.len() {
            // SAFETY: `max_offset < slice.len()` and offsets are monotonic in
            // both indices, so every `(i, j)` maps inside `slice`; `row_stride
            // >= nrows` makes that mapping injective, so no two indices alias.
            Some(unsafe { Self::new(slice.as_mut_ptr(), nrows, ncols, row_stride) })
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

    /// Returns the row stride: the number of elements between `(i, j)` and
    /// `(i, j + 1)` (the leading dimension; rows within a column are
    /// contiguous).
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
    ///
    /// Computing the pointer is safe for any `row` and `col`: it uses wrapping
    /// arithmetic and never dereferences. The result points at element
    /// `(row, col)` only when `row < nrows` and `col < ncols`, and only then may
    /// it be dereferenced (in `unsafe` code, while the view's borrow rules
    /// allow the access). For out-of-range indices the returned pointer is
    /// unspecified and must not be dereferenced.
    #[inline]
    pub fn ptr_at(&self, row: usize, col: usize) -> *const T {
        self.ptr.wrapping_add(self.wrapping_offset(row, col))
    }

    /// Returns a mutable pointer to the element at (row, col).
    ///
    /// As with [`ptr_at`](Self::ptr_at), computing the pointer is safe for any
    /// indices (wrapping arithmetic, no dereference); reading or writing
    /// through it requires `row < nrows` and `col < ncols`.
    #[inline]
    pub fn ptr_at_mut(&mut self, row: usize, col: usize) -> *mut T {
        self.ptr.wrapping_add(self.wrapping_offset(row, col))
    }

    /// The element offset of `(row, col)`, computed with wrapping arithmetic so
    /// that it never panics, whatever the indices.
    #[inline]
    fn wrapping_offset(&self, row: usize, col: usize) -> usize {
        row.wrapping_add(col.wrapping_mul(self.row_stride))
    }

    /// Returns a pointer to the element at (row, col), computed with in-bounds
    /// pointer arithmetic.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `row < nrows` and `col < ncols`.
    #[inline]
    unsafe fn ptr_at_in_bounds(&self, row: usize, col: usize) -> *mut T {
        debug_assert!(
            row < self.nrows && col < self.ncols,
            "MatMut index ({row}, {col}) out of bounds for a {}x{} view",
            self.nrows,
            self.ncols
        );
        // SAFETY: the caller guarantees `row < nrows` and `col < ncols`, so
        // `row + col * row_stride` is an element offset, which the
        // `MatMut::new` contract keeps inside the allocation (so the
        // arithmetic cannot overflow either).
        unsafe { self.ptr.add(row + col * self.row_stride) }
    }

    /// Immutable reborrow - creates an immutable view.
    ///
    /// This is the key to the reborrow pattern: it creates a new reference
    /// with a shorter lifetime, allowing temporary immutable access.
    #[inline]
    pub fn rb(&self) -> MatRef<'_, T> {
        // SAFETY: `MatRef::new` requires the same element offsets
        // (`i + j * row_stride`) to be in bounds as `MatMut::new`, which `self`
        // guarantees, and the same non-null, aligned, initialized `ptr`. The
        // data stays immutable for the reborrow: it borrows `self` immutably,
        // and the view's data is exclusive to `self` for `'a`.
        unsafe { MatRef::new(self.ptr, self.nrows, self.ncols, self.row_stride) }
    }

    /// Mutable reborrow - creates a new mutable view with a shorter lifetime.
    ///
    /// This allows passing the view to functions while retaining ownership.
    #[inline]
    pub fn rb_mut(&mut self) -> MatMut<'_, T> {
        // SAFETY: `self` already upholds the `MatMut::new` contract; the
        // reborrow only narrows the lifetime, and `&mut self` guarantees it is
        // the sole live handle while it exists.
        unsafe { MatMut::new(self.ptr, self.nrows, self.ncols, self.row_stride) }
    }

    /// Returns a reference to the element at (row, col).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.nrows && col < self.ncols {
            // SAFETY: the indices were bounds-checked just above, so the
            // pointer is in bounds, aligned, and points to an initialized
            // element (`MatMut::new` contract). The `&T` borrows `self`
            // immutably, so this exclusive view cannot hand out a `&mut` to
            // the element while it lives.
            Some(unsafe { &*self.ptr_at_in_bounds(row, col) })
        } else {
            None
        }
    }

    /// Returns a mutable reference to the element at (row, col).
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row < self.nrows && col < self.ncols {
            // SAFETY: the indices were bounds-checked just above, so the
            // pointer is in bounds, aligned, and points to an initialized
            // element. The view borrows its data exclusively for `'a`
            // (`MatMut::new` contract) and the `&mut T` borrows `self`
            // mutably, so it is the only live reference to that element.
            Some(unsafe { &mut *self.ptr_at_in_bounds(row, col) })
        } else {
            None
        }
    }

    /// Sets the element at (row, col).
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        assert!(row < self.nrows && col < self.ncols, "Index out of bounds");
        // SAFETY: the `assert!` above (active in release) bounds-checks the
        // indices, so the pointer addresses an in-bounds, initialized element
        // that this view borrows exclusively (`MatMut::new` contract);
        // `&mut self` rules out other references through the view, and
        // `T: Copy` makes the implicit drop of the old value a no-op.
        unsafe {
            *self.ptr_at_in_bounds(row, col) = value;
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
    ///
    /// Empty submatrices at the boundary (e.g. `submatrix(nrows, 0, 0, ncols)`)
    /// are legal and do not panic.
    ///
    /// # Panics
    ///
    /// Panics if the submatrix extends beyond the matrix bounds.
    #[inline]
    pub fn submatrix(self, row_start: usize, col_start: usize, nrows: usize, ncols: usize) -> Self {
        // Overflow-safe bounds check (as in `MatRef::submatrix`): each
        // `<= self.nrows - nrows` term is only evaluated after the preceding
        // `nrows <= self.nrows` guard (via `&&` short-circuiting), so the
        // subtraction can never underflow and nothing can wrap around `usize`.
        assert!(
            nrows <= self.nrows
                && ncols <= self.ncols
                && row_start <= self.nrows - nrows
                && col_start <= self.ncols - ncols,
            "Submatrix out of bounds"
        );

        // An empty sub-view addresses no element, so reuse `self.ptr` rather
        // than forming an offset that may lie past the end of the allocation.
        let ptr = if nrows == 0 || ncols == 0 {
            self.ptr
        } else {
            // SAFETY: the sub-view is non-empty, so the assertion above gives
            // `row_start < row_start + nrows <= self.nrows` and likewise
            // `col_start < self.ncols`.
            unsafe { self.ptr_at_in_bounds(row_start, col_start) }
        };

        // SAFETY: `ptr` is non-null and aligned (it is `self.ptr` or an element
        // of `self`). An empty sub-view addresses no element. Otherwise element
        // `(i, j)` of the sub-view is element `(row_start + i, col_start + j)`
        // of `self`, in range by the assertion, at offset
        // `i + j * row_stride` from `ptr`, so it lies in bounds. `self` is
        // consumed by value, so the sub-view is the only handle to these
        // elements for `'a`.
        unsafe { MatMut::new(ptr, nrows, ncols, self.row_stride) }
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
                // SAFETY: the loop bounds keep `i < nrows` and `j < ncols`, so
                // the pointer addresses an in-bounds, initialized element
                // (`MatMut::new` contract); `T: Copy`, so it is read by value.
                let val = unsafe { *self.ptr_at_in_bounds(i, j) };
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
                // SAFETY: the loop bounds keep `i < nrows` and `j < ncols`, so
                // the pointer addresses an in-bounds, initialized element
                // (`MatMut::new` contract); `T: Copy`, so it is read by value.
                let val = unsafe { *self.ptr_at_in_bounds(i, j) };
                self.set(i, j, val + alpha * other[(i, j)]);
            }
        }
    }

    /// Splits the matrix horizontally at column `mid`.
    ///
    /// # Panics
    ///
    /// Panics if `mid > ncols`, or if both halves are non-empty and
    /// `row_stride < nrows` (possible only for a view built with
    /// [`MatMut::new`]): the halves would then share elements.
    #[inline]
    pub fn split_cols(self, mid: usize) -> (Self, Self) {
        assert!(mid <= self.ncols, "Split point out of bounds");
        // Column `j` occupies the offsets `j * row_stride .. j * row_stride +
        // nrows`; these ranges are pairwise disjoint exactly when
        // `row_stride >= nrows`. With an empty half nothing can overlap.
        assert!(
            mid == 0 || mid == self.ncols || self.row_stride >= self.nrows,
            "split_cols would create overlapping mutable views (row_stride < nrows)"
        );

        let right_ncols = self.ncols - mid;
        // An empty right half addresses no element, so reuse `self.ptr` rather
        // than forming `mid * row_stride`, which may lie past the end of the
        // allocation (e.g. a padded submatrix ending at the parent's last
        // column) or overflow (an empty view with an arbitrary stride).
        let right_ptr = if self.nrows == 0 || right_ncols == 0 {
            self.ptr
        } else {
            // SAFETY: `nrows > 0` and `mid < ncols`, so this is element
            // `(0, mid)`.
            unsafe { self.ptr_at_in_bounds(0, mid) }
        };

        // SAFETY: element `(i, j)` of the left half (`j < mid`) is element
        // `(i, j)` of `self`, at the same offset from the same pointer, so it is
        // in bounds. The halves cover disjoint column ranges, and the assertion
        // above makes their elements disjoint in memory too. `self` is consumed
        // by value, so each half is the only handle to its elements.
        let left = unsafe { MatMut::new(self.ptr, self.nrows, mid, self.row_stride) };
        // SAFETY: as above: `right_ptr` is `self.ptr` (empty right half, which
        // addresses no element) or element `(0, mid)` of `self`, and element
        // `(i, j)` of the right half is element `(i, mid + j)` of `self`, at
        // offset `i + j * row_stride` from `right_ptr`.
        let right = unsafe { MatMut::new(right_ptr, self.nrows, right_ncols, self.row_stride) };

        (left, right)
    }

    /// Splits the matrix vertically at row `mid`.
    ///
    /// # Panics
    ///
    /// Panics if `mid > nrows`, or if both halves are non-empty, `ncols > 1`
    /// and `row_stride < nrows` (possible only for a view built with
    /// [`MatMut::new`]): the halves would then share elements.
    #[inline]
    pub fn split_rows(self, mid: usize) -> (Self, Self) {
        assert!(mid <= self.nrows, "Split point out of bounds");
        // Within one column the halves are disjoint row ranges; across columns
        // they stay disjoint when the column ranges themselves are disjoint
        // (`row_stride >= nrows`) or when there is at most one column.
        assert!(
            mid == 0 || mid == self.nrows || self.ncols <= 1 || self.row_stride >= self.nrows,
            "split_rows would create overlapping mutable views (row_stride < nrows)"
        );

        let bottom_nrows = self.nrows - mid;
        // An empty bottom half addresses no element, so reuse `self.ptr` rather
        // than forming `mid`, which need not be in bounds (e.g. a zero-column
        // view over an empty slice).
        let bottom_ptr = if self.ncols == 0 || bottom_nrows == 0 {
            self.ptr
        } else {
            // SAFETY: `ncols > 0` and `mid < nrows`, so this is element
            // `(mid, 0)`.
            unsafe { self.ptr_at_in_bounds(mid, 0) }
        };

        // SAFETY: element `(i, j)` of the top half (`i < mid`) is element
        // `(i, j)` of `self`, at the same offset from the same pointer, so it is
        // in bounds. The halves cover disjoint row ranges, and the assertion
        // above makes their elements disjoint in memory too. `self` is consumed
        // by value, so each half is the only handle to its elements.
        let top = unsafe { MatMut::new(self.ptr, mid, self.ncols, self.row_stride) };
        // SAFETY: as above: `bottom_ptr` is `self.ptr` (empty bottom half,
        // which addresses no element) or element `(mid, 0)` of `self`, and
        // element `(i, j)` of the bottom half is element `(mid + i, j)` of
        // `self`, at offset `i + j * row_stride` from `bottom_ptr`.
        let bottom = unsafe { MatMut::new(bottom_ptr, bottom_nrows, self.ncols, self.row_stride) };

        (top, bottom)
    }

    /// Returns a mutable slice of column `j` if contiguous.
    ///
    /// Columns of a `MatMut` are always contiguous, so this returns `Some` for
    /// every `j < ncols` (an empty slice when `nrows == 0`) and `None` for
    /// `j >= ncols`.
    #[inline]
    pub fn col_as_slice_mut(&mut self, j: usize) -> Option<&mut [T]> {
        if j >= self.ncols {
            return None;
        }

        // A zero-row view addresses no element and may carry any stride, so
        // `j * row_stride` must not be formed for it.
        if self.nrows == 0 {
            return Some(&mut []);
        }

        // SAFETY: `nrows > 0` and `j < ncols`, so this is element `(0, j)`.
        let start = unsafe { self.ptr_at_in_bounds(0, j) };
        // SAFETY: a `MatMut` column is contiguous (implicit row-to-row stride
        // of 1), so the `nrows` elements at `start` are the in-bounds,
        // initialized elements `(0..nrows, j)`. They lie in one allocation, so
        // their total size fits in `isize`. The slice borrows `self` mutably and
        // the view's data is exclusive to it for `'a`, so nothing else accesses
        // these elements while the slice lives.
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
            // SAFETY: `i1, i2 < nrows` (asserted above, also in release) and
            // `j < ncols` (loop bound), so both pointers address in-bounds,
            // initialized elements that this view borrows exclusively
            // (`MatMut::new` contract, `&mut self`); `ptr::swap` permits
            // overlapping pointers, so no distinctness is required.
            unsafe {
                let ptr1 = self.ptr_at_in_bounds(i1, j);
                let ptr2 = self.ptr_at_in_bounds(i2, j);
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
            // SAFETY: `j1, j2 < ncols` (asserted above, also in release) and
            // `i < nrows` (loop bound), so both pointers address in-bounds,
            // initialized elements that this view borrows exclusively
            // (`MatMut::new` contract, `&mut self`); `ptr::swap` permits
            // overlapping pointers (e.g. `row_stride == 0`), so no
            // distinctness is required.
            unsafe {
                let ptr1 = self.ptr_at_in_bounds(i, j1);
                let ptr2 = self.ptr_at_in_bounds(i, j2);
                core::ptr::swap(ptr1, ptr2);
            }
        }
    }
}

// Safety: MatMut is Send/Sync if T is
unsafe impl<'a, T: Scalar + Send> Send for MatMut<'a, T> {}
// SAFETY: `MatMut` behaves like `&'a mut [T]`, which is `Sync` when `T: Sync`.
// A shared `&MatMut` only reaches the `&self` methods (`get`, `Index`, `rb`,
// `submatrix_ref`, `ptr_at`, ...), which hand out shared `&T` / `MatRef`;
// every mutation needs `&mut self`. Concurrent shared reads of `T` are sound
// because `T: Sync`.
unsafe impl<'a, T: Scalar + Sync> Sync for MatMut<'a, T> {}

impl<'a, T: Scalar> core::ops::Index<(usize, usize)> for MatMut<'a, T> {
    type Output = T;

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(row < self.nrows && col < self.ncols, "Index out of bounds");
        // SAFETY: the `assert!` above bounds-checks the indices, so the pointer
        // addresses an in-bounds, initialized element (`MatMut::new`
        // contract); the `&T` borrows `self` immutably, so this exclusive view
        // cannot hand out a `&mut` to the element meanwhile.
        unsafe { &*self.ptr_at_in_bounds(row, col) }
    }
}

impl<'a, T: Scalar> core::ops::IndexMut<(usize, usize)> for MatMut<'a, T> {
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        assert!(row < self.nrows && col < self.ncols, "Index out of bounds");
        // SAFETY: the `assert!` above bounds-checks the indices, so the pointer
        // addresses an in-bounds, initialized element; the view borrows its
        // data exclusively for `'a` (`MatMut::new` contract) and the `&mut T`
        // borrows `self` mutably, so it is the only live reference to that
        // element.
        unsafe { &mut *self.ptr_at_in_bounds(row, col) }
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

    /// Regression tests for undefined behaviour that used to be reachable from
    /// safe code. They are small enough to run under Miri:
    /// `cargo +nightly miri test -p oxiblas-matrix --lib -- soundness`.
    mod soundness {
        use crate::{Mat, MatMut};

        /// `split_cols(ncols)` on a padded submatrix that ends at the parent's
        /// last column used to form `ptr + ncols * row_stride`, past the end of
        /// the allocation.
        #[test]
        fn split_cols_at_end_of_padded_submatrix() {
            let mut m: Mat<f64> = Mat::zeros(4, 4);
            {
                let (mut left, right) = m.as_mut().submatrix(1, 0, 2, 4).split_cols(4);
                assert_eq!(left.shape(), (2, 4));
                assert_eq!(right.shape(), (2, 0));
                assert!(right.is_empty());
                left[(1, 3)] = 7.0;
            }
            assert_eq!(m[(2, 3)], 7.0);
        }

        /// Same for a view whose slice ends exactly at its last element.
        #[test]
        fn split_cols_at_end_of_tight_strided_slice() {
            let mut data = [0.0f64; 5];
            {
                let view = MatMut::from_strided(&mut data, 2, 2, 3).expect("fits");
                let (mut left, right) = view.split_cols(2);
                assert_eq!(left.shape(), (2, 2));
                assert_eq!(right.shape(), (2, 0));
                left[(1, 1)] = 1.0;
            }
            assert_eq!(data[4], 1.0);
        }

        /// `split_cols` on a view over an empty slice offsets a dangling pointer.
        #[test]
        fn split_cols_of_empty_slice_view() {
            let mut data: [f64; 0] = [];
            let view = MatMut::from_slice(&mut data);
            let (left, right) = view.split_cols(1);
            assert_eq!(left.shape(), (0, 1));
            assert_eq!(right.shape(), (0, 0));
        }

        /// `split_rows` on a zero-column view over an empty slice offsets a
        /// dangling pointer by `mid`.
        #[test]
        fn split_rows_of_zero_column_view() {
            let mut data: [f64; 0] = [];
            let view = MatMut::from_strided(&mut data, 3, 0, 3).expect("empty view");
            let (top, bottom) = view.split_rows(2);
            assert_eq!(top.shape(), (2, 0));
            assert_eq!(bottom.shape(), (1, 0));
        }

        /// A zero-row view accepts any stride, so `col_as_slice_mut` must not
        /// form `j * row_stride` for it.
        #[test]
        fn col_as_slice_mut_of_zero_row_view_with_huge_stride() {
            let mut data = [0.0f64; 4];
            let mut view = MatMut::from_strided(&mut data, 0, 5, 999).expect("empty view");
            assert_eq!(view.col_as_slice_mut(3).map(|s| s.len()), Some(0));
            assert_eq!(view.col_as_slice_mut(4).map(|s| s.len()), Some(0));
            assert!(view.col_as_slice_mut(5).is_none());

            let mut huge = MatMut::from_strided(&mut data, 0, 3, usize::MAX).expect("empty view");
            assert_eq!(huge.col_as_slice_mut(2).map(|s| s.len()), Some(0));
        }

        /// `ptr_at` / `ptr_at_mut` are safe and must be sound for any indices;
        /// only dereferencing needs in-range indices.
        #[test]
        fn ptr_at_out_of_range_is_sound() {
            let mut m: Mat<f64> = Mat::zeros(2, 2);
            let mut view = m.as_mut();
            assert!(!view.ptr_at_mut(100, 100).is_null());
            assert!(!view.ptr_at(100, 100).is_null());
            // Huge indices must not overflow-panic either.
            let _ = view.ptr_at_mut(usize::MAX, usize::MAX);
            let _ = view.ptr_at(usize::MAX, 1);
            // In-range results are unchanged.
            let base = view.as_ptr();
            let stride = view.row_stride();
            assert_eq!(view.ptr_at(1, 1), base.wrapping_add(1 + stride));
        }

        /// `row_start + nrows` used to wrap around in release builds, so the
        /// bounds check accepted an out-of-bounds submatrix.
        #[test]
        #[should_panic(expected = "Submatrix out of bounds")]
        fn submatrix_wraparound_rejected() {
            let mut m: Mat<f64> = Mat::zeros(4, 4);
            let _ = m.as_mut().submatrix(usize::MAX - 3, 0, 4, 1);
        }

        /// Empty submatrices at the right or bottom edge are legal.
        #[test]
        fn submatrix_empty_at_edges() {
            let mut m: Mat<f64> = Mat::zeros(4, 4);
            assert_eq!(m.as_mut().submatrix(1, 4, 3, 0).shape(), (3, 0));
            assert_eq!(m.as_mut().submatrix(4, 1, 0, 3).shape(), (0, 3));
            assert_eq!(m.as_mut().submatrix(4, 4, 0, 0).shape(), (0, 0));
            let mut sub = m.as_mut().submatrix(1, 1, 3, 3);
            sub[(2, 2)] = 5.0;
            assert_eq!(m[(3, 3)], 5.0);
        }

        /// `MatMut::new` allows `row_stride < nrows`, so distinct indices can
        /// share an element. Splitting such a view would hand out two mutable
        /// views of one element; that must panic instead.
        #[test]
        #[should_panic(expected = "overlapping")]
        fn split_cols_rejects_overlapping_halves() {
            let mut data = [0.0f64; 4];
            // SAFETY: offsets `i + j` for `i < 3`, `j < 2` are at most 3, inside
            // `data`, which is borrowed exclusively for the view's lifetime.
            let view = unsafe { MatMut::new(data.as_mut_ptr(), 3, 2, 1) };
            let _ = view.split_cols(1);
        }

        /// Same for `split_rows`.
        #[test]
        #[should_panic(expected = "overlapping")]
        fn split_rows_rejects_overlapping_halves() {
            let mut data = [0.0f64; 3];
            // SAFETY: offsets `i + j` for `i < 2`, `j < 2` are at most 2, inside
            // `data`, which is borrowed exclusively for the view's lifetime.
            let view = unsafe { MatMut::new(data.as_mut_ptr(), 2, 2, 1) };
            let _ = view.split_rows(1);
        }

        /// Splits that cannot overlap still work on such views: one half is
        /// empty, or the view has a single column.
        #[test]
        fn non_overlapping_splits_of_short_stride_views() {
            let mut data = [0.0f64; 4];
            // SAFETY: offsets `i + j` for `i < 3`, `j < 2` are at most 3, inside
            // `data`, which is borrowed exclusively for the view's lifetime.
            let view = unsafe { MatMut::new(data.as_mut_ptr(), 3, 2, 1) };
            let (left, right) = view.split_cols(2);
            assert_eq!((left.shape(), right.shape()), ((3, 2), (3, 0)));

            let mut col = [0.0f64; 4];
            let (mut top, mut bottom) = MatMut::from_slice(&mut col).split_rows(1);
            top[(0, 0)] = 1.0;
            bottom[(2, 0)] = 2.0;
            assert_eq!(col, [1.0, 0.0, 0.0, 2.0]);
        }
    }
}
