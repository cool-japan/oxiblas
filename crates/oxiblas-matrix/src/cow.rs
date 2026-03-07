//! Copy-on-Write matrix type.
//!
//! `CowMat<T>` provides efficient matrix sharing with copy-on-write semantics.
//! Multiple matrices can share the same underlying data until a mutation is needed,
//! at which point the data is cloned.
//!
//! # Use Cases
//!
//! - Passing matrices through multiple computation stages where most don't mutate
//! - Caching intermediate results that may or may not be modified
//! - Reducing memory allocations in algorithms that conditionally modify matrices
//!
//! # Example
//!
//! ```
//! use oxiblas_matrix::CowMat;
//!
//! // Create a matrix
//! let a: CowMat<f64> = CowMat::zeros(3, 3);
//!
//! // Clone is cheap - shares the underlying data
//! let b = a.clone();
//! assert!(a.is_shared()); // Now shared with b
//!
//! // Reading doesn't trigger a copy
//! let _val = a[(0, 0)];
//!
//! // Writing to b triggers a copy of the data
//! let mut b = b;
//! b.make_mut()[(0, 0)] = 1.0;
//! assert!(!b.is_shared()); // b now has its own copy
//! ```

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::mat::Mat;
use crate::mat_mut::MatMut;
use crate::mat_ref::MatRef;
#[cfg(not(feature = "std"))]
use alloc::sync::Arc;
use core::ops::{Index, IndexMut};
use oxiblas_core::memory::{AlignedVec, DEFAULT_ALIGN};
use oxiblas_core::scalar::Scalar;
#[cfg(feature = "std")]
use std::sync::Arc;

/// Shared matrix data with reference counting.
struct SharedMatData<T: Scalar> {
    /// Underlying data storage.
    data: AlignedVec<T, DEFAULT_ALIGN>,
    /// Number of rows.
    nrows: usize,
    /// Number of columns.
    ncols: usize,
    /// Stride between consecutive rows.
    row_stride: usize,
}

/// A Copy-on-Write matrix.
///
/// This type wraps matrix data in an `Arc`, allowing multiple `CowMat` instances
/// to share the same underlying data. When a mutation is needed, the data is
/// cloned if it's shared with other instances.
///
/// # Performance Characteristics
///
/// - **Clone**: O(1) - just increments reference count
/// - **Read access**: O(1) - direct pointer access
/// - **Write access**: O(1) if unique, O(n*m) if shared (triggers copy)
/// - **Memory**: Shared until mutation
pub struct CowMat<T: Scalar> {
    /// Shared data wrapped in Arc.
    inner: Arc<SharedMatData<T>>,
}

impl<T: Scalar> CowMat<T> {
    /// Creates a new COW matrix filled with zeros.
    pub fn zeros(nrows: usize, ncols: usize) -> Self
    where
        T: bytemuck::Zeroable,
    {
        let mat = Mat::zeros(nrows, ncols);
        Self::from_mat(mat)
    }

    /// Creates a new COW matrix filled with a specific value.
    pub fn filled(nrows: usize, ncols: usize, value: T) -> Self
    where
        T: bytemuck::Zeroable,
    {
        let mat = Mat::filled(nrows, ncols, value);
        Self::from_mat(mat)
    }

    /// Creates a new COW matrix from a flat slice in column-major order.
    pub fn from_slice(nrows: usize, ncols: usize, data: &[T]) -> Self
    where
        T: bytemuck::Zeroable,
    {
        let mat = Mat::from_slice(nrows, ncols, data);
        Self::from_mat(mat)
    }

    /// Creates a new COW matrix from row data.
    pub fn from_rows(rows: &[&[T]]) -> Self
    where
        T: bytemuck::Zeroable,
    {
        let mat = Mat::from_rows(rows);
        Self::from_mat(mat)
    }

    /// Creates an identity matrix.
    pub fn eye(n: usize) -> Self
    where
        T: bytemuck::Zeroable,
    {
        let mat = Mat::eye(n);
        Self::from_mat(mat)
    }

    /// Creates a diagonal matrix from a slice.
    pub fn diag(diagonal: &[T]) -> Self
    where
        T: bytemuck::Zeroable,
    {
        let mat = Mat::diag(diagonal);
        Self::from_mat(mat)
    }

    /// Creates a COW matrix from an owned Mat.
    pub fn from_mat(mat: Mat<T>) -> Self
    where
        T: bytemuck::Zeroable,
    {
        let nrows = mat.nrows();
        let ncols = mat.ncols();
        let row_stride = mat.row_stride();
        let total = row_stride * ncols;

        let mut data = AlignedVec::zeros(total);

        // Copy data from mat
        let mat_ref = mat.as_ref();
        for j in 0..ncols {
            for i in 0..nrows {
                data[i + j * row_stride] = mat_ref[(i, j)];
            }
            // Fill padding with zeros (already done by zeros())
        }

        CowMat {
            inner: Arc::new(SharedMatData {
                data,
                nrows,
                ncols,
                row_stride,
            }),
        }
    }

    /// Returns the number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.inner.nrows
    }

    /// Returns the number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.inner.ncols
    }

    /// Returns the shape as (nrows, ncols).
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.inner.nrows, self.inner.ncols)
    }

    /// Returns the row stride.
    #[inline]
    pub fn row_stride(&self) -> usize {
        self.inner.row_stride
    }

    /// Returns true if the data is shared with other CowMat instances.
    #[inline]
    pub fn is_shared(&self) -> bool {
        Arc::strong_count(&self.inner) > 1
    }

    /// Returns true if this is the only reference to the data.
    #[inline]
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.inner) == 1
    }

    /// Returns the reference count.
    #[inline]
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    /// Returns an immutable view of the matrix.
    #[inline]
    pub fn as_ref(&self) -> MatRef<'_, T> {
        MatRef::new(
            self.inner.data.as_ptr(),
            self.inner.nrows,
            self.inner.ncols,
            self.inner.row_stride,
        )
    }

    /// Returns a pointer to the first element.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.inner.data.as_ptr()
    }

    /// Returns the element at (row, col).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.inner.nrows && col < self.inner.ncols {
            Some(&self.inner.data[row + col * self.inner.row_stride])
        } else {
            None
        }
    }

    /// Returns a submatrix view.
    #[inline]
    pub fn submatrix(
        &self,
        row_start: usize,
        col_start: usize,
        nrows: usize,
        ncols: usize,
    ) -> MatRef<'_, T> {
        self.as_ref().submatrix(row_start, col_start, nrows, ncols)
    }

    /// Returns a column view.
    #[inline]
    pub fn col(&self, j: usize) -> MatRef<'_, T> {
        assert!(j < self.inner.ncols, "Column index out of bounds");
        self.submatrix(0, j, self.inner.nrows, 1)
    }

    /// Returns a row view.
    #[inline]
    pub fn row(&self, i: usize) -> MatRef<'_, T> {
        assert!(i < self.inner.nrows, "Row index out of bounds");
        self.submatrix(i, 0, 1, self.inner.ncols)
    }

    /// Returns the diagonal as a vector.
    pub fn diagonal(&self) -> Vec<T> {
        let n = self.inner.nrows.min(self.inner.ncols);
        (0..n)
            .map(|i| self.inner.data[i + i * self.inner.row_stride])
            .collect()
    }

    /// Converts to an owned Mat, cloning if shared.
    pub fn into_owned(self) -> Mat<T>
    where
        T: bytemuck::Zeroable,
    {
        self.to_mat()
    }

    /// Creates a new owned Mat from this COW matrix.
    pub fn to_mat(&self) -> Mat<T>
    where
        T: bytemuck::Zeroable,
    {
        let mut mat = Mat::zeros(self.inner.nrows, self.inner.ncols);
        let mut mat_mut = mat.as_mut();
        let self_ref = self.as_ref();

        for j in 0..self.inner.ncols {
            for i in 0..self.inner.nrows {
                mat_mut[(i, j)] = self_ref[(i, j)];
            }
        }

        mat
    }

    /// Ensures unique ownership, cloning if necessary, and returns a mutable view.
    ///
    /// This is the key COW operation. If the data is shared, it will be cloned
    /// before returning a mutable reference.
    pub fn make_mut(&mut self) -> MatMut<'_, T> {
        // If we're not unique, we need to clone the data
        if !self.is_unique() {
            let cloned = SharedMatData {
                data: self.inner.data.clone(),
                nrows: self.inner.nrows,
                ncols: self.inner.ncols,
                row_stride: self.inner.row_stride,
            };
            self.inner = Arc::new(cloned);
        }

        // Now we're guaranteed to be unique, get mutable access
        // Safety: We just ensured we're the only owner
        let inner = Arc::get_mut(&mut self.inner).expect("Should be unique after clone");

        MatMut::new(
            inner.data.as_mut_ptr(),
            inner.nrows,
            inner.ncols,
            inner.row_stride,
        )
    }

    /// Sets an element, cloning if the data is shared.
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        assert!(
            row < self.inner.nrows && col < self.inner.ncols,
            "Index out of bounds"
        );
        self.make_mut()[(row, col)] = value;
    }

    /// Returns a mutable pointer, cloning if shared.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.make_mut().as_mut_ptr()
    }
}

impl<T: Scalar> Clone for CowMat<T> {
    /// Clone is O(1) - just increments the reference count.
    #[inline]
    fn clone(&self) -> Self {
        CowMat {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T: Scalar> Index<(usize, usize)> for CowMat<T> {
    type Output = T;

    #[inline]
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        assert!(
            row < self.inner.nrows && col < self.inner.ncols,
            "Index out of bounds"
        );
        &self.inner.data[row + col * self.inner.row_stride]
    }
}

impl<T: Scalar> IndexMut<(usize, usize)> for CowMat<T> {
    /// Mutable indexing triggers copy-on-write if the data is shared.
    #[inline]
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        assert!(
            row < self.inner.nrows && col < self.inner.ncols,
            "Index out of bounds"
        );

        // Ensure we have unique ownership
        if !self.is_unique() {
            let cloned = SharedMatData {
                data: self.inner.data.clone(),
                nrows: self.inner.nrows,
                ncols: self.inner.ncols,
                row_stride: self.inner.row_stride,
            };
            self.inner = Arc::new(cloned);
        }

        let inner = Arc::get_mut(&mut self.inner).expect("Should be unique");
        let idx = row + col * inner.row_stride;
        &mut inner.data[idx]
    }
}

impl<T: Scalar + bytemuck::Zeroable> From<Mat<T>> for CowMat<T> {
    fn from(mat: Mat<T>) -> Self {
        CowMat::from_mat(mat)
    }
}

impl<T: Scalar + core::fmt::Debug> core::fmt::Debug for CowMat<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CowMat")
            .field("nrows", &self.inner.nrows)
            .field("ncols", &self.inner.ncols)
            .field("ref_count", &self.ref_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cow_basic() {
        let a: CowMat<f64> = CowMat::zeros(3, 3);
        assert_eq!(a.nrows(), 3);
        assert_eq!(a.ncols(), 3);
        assert!(!a.is_shared());
        assert_eq!(a.ref_count(), 1);
    }

    #[test]
    fn test_cow_clone_shares_data() {
        let a: CowMat<f64> = CowMat::zeros(3, 3);
        let b = a.clone();

        assert!(a.is_shared());
        assert!(b.is_shared());
        assert_eq!(a.ref_count(), 2);
        assert_eq!(b.ref_count(), 2);

        // Data pointers should be the same
        assert_eq!(a.as_ptr(), b.as_ptr());
    }

    #[test]
    fn test_cow_read_no_copy() {
        let a: CowMat<f64> = CowMat::filled(3, 3, 1.0);
        let b = a.clone();

        // Reading shouldn't trigger a copy
        let _val = a[(0, 0)];
        let _val = b[(1, 1)];

        assert!(a.is_shared());
        assert_eq!(a.as_ptr(), b.as_ptr());
    }

    #[test]
    fn test_cow_write_triggers_copy() {
        let a: CowMat<f64> = CowMat::filled(3, 3, 1.0);
        let mut b = a.clone();

        let a_ptr = a.as_ptr();
        let b_ptr_before = b.as_ptr();
        assert_eq!(a_ptr, b_ptr_before);

        // Writing to b should trigger a copy
        b[(0, 0)] = 2.0;

        let b_ptr_after = b.as_ptr();
        assert_ne!(a_ptr, b_ptr_after); // b now has its own data

        // a should still have original value
        assert_eq!(a[(0, 0)], 1.0);
        // b should have new value
        assert_eq!(b[(0, 0)], 2.0);

        // a is no longer shared (b has its own copy)
        assert!(!a.is_shared());
        assert!(!b.is_shared());
    }

    #[test]
    fn test_cow_make_mut() {
        let a: CowMat<f64> = CowMat::zeros(3, 3);
        let mut b = a.clone();

        assert!(b.is_shared());

        {
            let mut mut_view = b.make_mut();
            mut_view[(0, 0)] = 5.0;
        }

        assert!(!b.is_shared());
        assert_eq!(b[(0, 0)], 5.0);
        assert_eq!(a[(0, 0)], 0.0);
    }

    #[test]
    fn test_cow_from_mat() {
        let mat: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let cow: CowMat<f64> = CowMat::from_mat(mat);

        assert_eq!(cow.nrows(), 2);
        assert_eq!(cow.ncols(), 3);
        assert_eq!(cow[(0, 0)], 1.0);
        assert_eq!(cow[(1, 2)], 6.0);
    }

    #[test]
    fn test_cow_to_mat() {
        let cow: CowMat<f64> = CowMat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);

        let mat = cow.to_mat();

        assert_eq!(mat.nrows(), 2);
        assert_eq!(mat.ncols(), 2);
        assert_eq!(mat[(0, 0)], 1.0);
        assert_eq!(mat[(1, 1)], 4.0);
    }

    #[test]
    fn test_cow_diagonal() {
        let cow: CowMat<f64> =
            CowMat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        let diag = cow.diagonal();
        assert_eq!(diag, vec![1.0, 5.0, 9.0]);
    }

    #[test]
    fn test_cow_submatrix() {
        let cow: CowMat<f64> =
            CowMat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        let sub = cow.submatrix(1, 1, 2, 2);
        assert_eq!(sub[(0, 0)], 5.0);
        assert_eq!(sub[(1, 1)], 9.0);
    }

    #[test]
    fn test_cow_eye() {
        let cow: CowMat<f64> = CowMat::eye(3);

        assert_eq!(cow[(0, 0)], 1.0);
        assert_eq!(cow[(1, 1)], 1.0);
        assert_eq!(cow[(2, 2)], 1.0);
        assert_eq!(cow[(0, 1)], 0.0);
    }

    #[test]
    fn test_cow_unique_no_copy_on_write() {
        let mut a: CowMat<f64> = CowMat::filled(3, 3, 1.0);

        let ptr_before = a.as_ptr();

        // Since we're unique, writing shouldn't allocate new memory
        a[(0, 0)] = 2.0;

        let ptr_after = a.as_ptr();
        assert_eq!(ptr_before, ptr_after); // Same memory
        assert_eq!(a[(0, 0)], 2.0);
    }

    #[test]
    fn test_cow_multiple_clones() {
        let a: CowMat<f64> = CowMat::zeros(2, 2);
        let b = a.clone();
        let c = a.clone();

        assert_eq!(a.ref_count(), 3);
        assert_eq!(b.ref_count(), 3);
        assert_eq!(c.ref_count(), 3);

        drop(b);

        assert_eq!(a.ref_count(), 2);
        assert_eq!(c.ref_count(), 2);
    }

    #[test]
    fn test_cow_set() {
        let a: CowMat<f64> = CowMat::zeros(2, 2);
        let mut b = a.clone();

        b.set(0, 0, 10.0);

        assert_eq!(a[(0, 0)], 0.0);
        assert_eq!(b[(0, 0)], 10.0);
    }
}
