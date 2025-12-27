//! Packed matrix storage for triangular and symmetric matrices.
//!
//! Packed storage stores only the upper or lower triangular portion of a matrix
//! in a one-dimensional array, reducing memory usage by nearly half.
//!
//! # Storage Layout
//!
//! For an `n × n` matrix, packed storage uses `n*(n+1)/2` elements.
//!
//! **Upper triangular (column-major)**: Elements are stored column by column,
//! starting from the diagonal:
//! ```text
//! [a00, a01, a11, a02, a12, a22, a03, a13, a23, a33, ...]
//! ```
//!
//! **Lower triangular (column-major)**: Elements are stored column by column:
//! ```text
//! [a00, a10, a20, a30, a11, a21, a31, a22, a32, a33, ...]
//! ```

use oxiblas_core::memory::AlignedVec;
use oxiblas_core::scalar::Scalar;

/// Specifies whether to use upper or lower triangular storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriangularKind {
    /// Upper triangular: stores elements where row <= col.
    Upper,
    /// Lower triangular: stores elements where row >= col.
    Lower,
}

/// A packed matrix storing only the triangular portion.
///
/// This is useful for symmetric, Hermitian, and triangular matrices
/// where only half the elements need to be stored.
///
/// # Example
///
/// ```
/// use oxiblas_matrix::packed::{PackedMat, TriangularKind};
///
/// // Create a 3x3 upper triangular packed matrix
/// let mut p: PackedMat<f64> = PackedMat::zeros(3, TriangularKind::Upper);
///
/// // Set diagonal and upper triangle
/// p.set(0, 0, 1.0);
/// p.set(0, 1, 2.0);
/// p.set(0, 2, 3.0);
/// p.set(1, 1, 4.0);
/// p.set(1, 2, 5.0);
/// p.set(2, 2, 6.0);
///
/// // Access elements
/// assert_eq!(p.get(0, 1), Some(&2.0));
/// assert_eq!(p.get(1, 0), None); // Below diagonal in upper triangular
/// ```
#[derive(Clone)]
pub struct PackedMat<T: Scalar> {
    /// Packed data storage.
    data: AlignedVec<T>,
    /// Matrix dimension (n × n).
    n: usize,
    /// Upper or lower triangular.
    kind: TriangularKind,
}

impl<T: Scalar> PackedMat<T> {
    /// Creates a new packed matrix filled with zeros.
    pub fn zeros(n: usize, kind: TriangularKind) -> Self
    where
        T: bytemuck::Zeroable,
    {
        let len = Self::packed_len(n);
        PackedMat {
            data: AlignedVec::zeros(len),
            n,
            kind,
        }
    }

    /// Creates a new packed matrix filled with a specific value.
    pub fn filled(n: usize, kind: TriangularKind, value: T) -> Self {
        let len = Self::packed_len(n);
        PackedMat {
            data: AlignedVec::filled(len, value),
            n,
            kind,
        }
    }

    /// Creates a packed matrix from a slice.
    ///
    /// # Panics
    /// Panics if the slice length doesn't match `n*(n+1)/2`.
    pub fn from_slice(n: usize, kind: TriangularKind, data: &[T]) -> Self {
        let len = Self::packed_len(n);
        assert_eq!(
            data.len(),
            len,
            "Slice length must equal n*(n+1)/2 = {}",
            len
        );

        PackedMat {
            data: AlignedVec::from_slice(data),
            n,
            kind,
        }
    }

    /// Computes the packed storage length for dimension n.
    #[inline]
    pub const fn packed_len(n: usize) -> usize {
        n * (n + 1) / 2
    }

    /// Returns the matrix dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Returns the storage kind (upper or lower).
    #[inline]
    pub fn kind(&self) -> TriangularKind {
        self.kind
    }

    /// Returns the packed data length.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the matrix is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Computes the packed index for element (row, col).
    ///
    /// Returns `None` if the element is in the non-stored triangle.
    #[inline]
    pub fn packed_index(&self, row: usize, col: usize) -> Option<usize> {
        if row >= self.n || col >= self.n {
            return None;
        }

        match self.kind {
            TriangularKind::Upper => {
                if row <= col {
                    // Column-major upper: index = col*(col+1)/2 + row
                    Some(col * (col + 1) / 2 + row)
                } else {
                    None
                }
            }
            TriangularKind::Lower => {
                if row >= col {
                    // Column-major lower:
                    // Column j starts at index: n*j - j*(j-1)/2
                    // Element (row, col) is at: start + (row - col)
                    let offset = self.n * col - col * (col.saturating_sub(1)) / 2;
                    Some(offset + (row - col))
                } else {
                    None
                }
            }
        }
    }

    /// Returns a reference to the element at (row, col).
    ///
    /// Returns `None` if the element is outside the stored triangle.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.packed_index(row, col).map(|idx| &self.data[idx])
    }

    /// Returns a mutable reference to the element at (row, col).
    ///
    /// Returns `None` if the element is outside the stored triangle.
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        self.packed_index(row, col).map(|idx| &mut self.data[idx])
    }

    /// Sets the element at (row, col).
    ///
    /// # Panics
    /// Panics if the element is outside the stored triangle.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        let idx = self
            .packed_index(row, col)
            .expect("Element outside stored triangle");
        self.data[idx] = value;
    }

    /// Returns a pointer to the packed data.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Returns a mutable pointer to the packed data.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    /// Returns the packed data as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }

    /// Returns the packed data as a mutable slice.
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }

    /// Converts to a full dense matrix.
    pub fn to_dense(&self) -> crate::Mat<T>
    where
        T: bytemuck::Zeroable,
    {
        let mut mat = crate::Mat::zeros(self.n, self.n);

        for j in 0..self.n {
            for i in 0..self.n {
                if let Some(idx) = self.packed_index(i, j) {
                    mat[(i, j)] = self.data[idx];
                }
            }
        }

        mat
    }

    /// Creates a packed matrix from a dense matrix.
    ///
    /// Only copies elements from the specified triangle.
    pub fn from_dense(mat: &crate::MatRef<'_, T>, kind: TriangularKind) -> Self
    where
        T: bytemuck::Zeroable,
    {
        assert_eq!(mat.nrows(), mat.ncols(), "Matrix must be square");
        let n = mat.nrows();
        let mut packed = Self::zeros(n, kind);

        for j in 0..n {
            for i in 0..n {
                if packed.packed_index(i, j).is_some() {
                    packed.set(i, j, mat[(i, j)]);
                }
            }
        }

        packed
    }

    /// Returns the diagonal elements as a vector.
    pub fn diagonal(&self) -> Vec<T> {
        (0..self.n).map(|i| *self.get(i, i).unwrap()).collect()
    }

    /// Sets the diagonal elements from a slice.
    pub fn set_diagonal(&mut self, diag: &[T]) {
        assert_eq!(
            diag.len(),
            self.n,
            "Diagonal length must match matrix dimension"
        );
        for (i, &val) in diag.iter().enumerate() {
            self.set(i, i, val);
        }
    }

    /// Fills the stored triangle with a value.
    pub fn fill(&mut self, value: T) {
        for elem in self.data.as_mut_slice() {
            *elem = value;
        }
    }

    /// Scales all stored elements by a scalar.
    pub fn scale(&mut self, alpha: T) {
        for elem in self.data.as_mut_slice() {
            *elem *= alpha;
        }
    }

    /// Converts between upper and lower triangular representation.
    ///
    /// For symmetric matrices, this effectively transposes the packed data.
    pub fn transpose(&self) -> Self
    where
        T: bytemuck::Zeroable,
    {
        let new_kind = match self.kind {
            TriangularKind::Upper => TriangularKind::Lower,
            TriangularKind::Lower => TriangularKind::Upper,
        };

        let mut result = Self::zeros(self.n, new_kind);

        for j in 0..self.n {
            for i in 0..self.n {
                if let Some(src_idx) = self.packed_index(i, j) {
                    // In transposed storage, (i,j) becomes (j,i)
                    if result.packed_index(j, i).is_some() {
                        result.set(j, i, self.data[src_idx]);
                    }
                }
            }
        }

        result
    }
}

impl<T: Scalar + core::fmt::Debug> core::fmt::Debug for PackedMat<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "PackedMat {}x{} {:?} {{", self.n, self.n, self.kind)?;

        for i in 0..self.n.min(8) {
            write!(f, "  [")?;
            for j in 0..self.n.min(8) {
                if j > 0 {
                    write!(f, ", ")?;
                }
                match self.get(i, j) {
                    Some(v) => write!(f, "{:8.4?}", v)?,
                    None => write!(f, "      * ")?,
                }
            }
            if self.n > 8 {
                write!(f, ", ...")?;
            }
            writeln!(f, "]")?;
        }
        if self.n > 8 {
            writeln!(f, "  ...")?;
        }
        write!(f, "}}")
    }
}

/// A view into packed matrix data.
#[derive(Clone, Copy)]
pub struct PackedRef<'a, T: Scalar> {
    /// Pointer to packed data.
    ptr: *const T,
    /// Matrix dimension.
    n: usize,
    /// Storage kind.
    kind: TriangularKind,
    /// Lifetime marker.
    _marker: core::marker::PhantomData<&'a T>,
}

impl<'a, T: Scalar> PackedRef<'a, T> {
    /// Creates a new packed reference from raw components.
    #[inline]
    pub fn new(ptr: *const T, n: usize, kind: TriangularKind) -> Self {
        PackedRef {
            ptr,
            n,
            kind,
            _marker: core::marker::PhantomData,
        }
    }

    /// Creates a packed reference from a slice.
    #[inline]
    pub fn from_slice(data: &'a [T], n: usize, kind: TriangularKind) -> Self {
        let expected_len = PackedMat::<T>::packed_len(n);
        assert_eq!(
            data.len(),
            expected_len,
            "Slice length must equal n*(n+1)/2"
        );
        PackedRef::new(data.as_ptr(), n, kind)
    }

    /// Returns the matrix dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Returns the storage kind.
    #[inline]
    pub fn kind(&self) -> TriangularKind {
        self.kind
    }

    /// Computes the packed index for element (row, col).
    #[inline]
    pub fn packed_index(&self, row: usize, col: usize) -> Option<usize> {
        if row >= self.n || col >= self.n {
            return None;
        }

        match self.kind {
            TriangularKind::Upper => {
                if row <= col {
                    Some(col * (col + 1) / 2 + row)
                } else {
                    None
                }
            }
            TriangularKind::Lower => {
                if row >= col {
                    // Column-major lower:
                    // Column j starts at index: n*j - j*(j-1)/2
                    // Element (row, col) is at: start + (row - col)
                    let offset = self.n * col - col * (col.saturating_sub(1)) / 2;
                    Some(offset + (row - col))
                } else {
                    None
                }
            }
        }
    }

    /// Returns a reference to the element at (row, col).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.packed_index(row, col)
            .map(|idx| unsafe { &*self.ptr.add(idx) })
    }

    /// Returns a pointer to the packed data.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }
}

unsafe impl<'a, T: Scalar + Send> Send for PackedRef<'a, T> {}
unsafe impl<'a, T: Scalar + Sync> Sync for PackedRef<'a, T> {}

/// A mutable view into packed matrix data.
pub struct PackedMut<'a, T: Scalar> {
    /// Pointer to packed data.
    ptr: *mut T,
    /// Matrix dimension.
    n: usize,
    /// Storage kind.
    kind: TriangularKind,
    /// Lifetime marker.
    _marker: core::marker::PhantomData<&'a mut T>,
}

impl<'a, T: Scalar> PackedMut<'a, T> {
    /// Creates a new mutable packed reference from raw components.
    #[inline]
    pub fn new(ptr: *mut T, n: usize, kind: TriangularKind) -> Self {
        PackedMut {
            ptr,
            n,
            kind,
            _marker: core::marker::PhantomData,
        }
    }

    /// Creates a mutable packed reference from a mutable slice.
    #[inline]
    pub fn from_slice(data: &'a mut [T], n: usize, kind: TriangularKind) -> Self {
        let expected_len = PackedMat::<T>::packed_len(n);
        assert_eq!(
            data.len(),
            expected_len,
            "Slice length must equal n*(n+1)/2"
        );
        PackedMut::new(data.as_mut_ptr(), n, kind)
    }

    /// Returns the matrix dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Returns the storage kind.
    #[inline]
    pub fn kind(&self) -> TriangularKind {
        self.kind
    }

    /// Computes the packed index for element (row, col).
    #[inline]
    pub fn packed_index(&self, row: usize, col: usize) -> Option<usize> {
        if row >= self.n || col >= self.n {
            return None;
        }

        match self.kind {
            TriangularKind::Upper => {
                if row <= col {
                    Some(col * (col + 1) / 2 + row)
                } else {
                    None
                }
            }
            TriangularKind::Lower => {
                if row >= col {
                    // Column-major lower:
                    // Column j starts at index: n*j - j*(j-1)/2
                    // Element (row, col) is at: start + (row - col)
                    let offset = self.n * col - col * (col.saturating_sub(1)) / 2;
                    Some(offset + (row - col))
                } else {
                    None
                }
            }
        }
    }

    /// Returns a reference to the element at (row, col).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        self.packed_index(row, col)
            .map(|idx| unsafe { &*self.ptr.add(idx) })
    }

    /// Returns a mutable reference to the element at (row, col).
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        self.packed_index(row, col)
            .map(|idx| unsafe { &mut *self.ptr.add(idx) })
    }

    /// Sets the element at (row, col).
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        let idx = self
            .packed_index(row, col)
            .expect("Element outside stored triangle");
        unsafe {
            *self.ptr.add(idx) = value;
        }
    }

    /// Returns a pointer to the packed data.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns a mutable pointer to the packed data.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Creates an immutable reborrow.
    #[inline]
    pub fn rb(&self) -> PackedRef<'_, T> {
        PackedRef::new(self.ptr, self.n, self.kind)
    }

    /// Creates a mutable reborrow.
    #[inline]
    pub fn rb_mut(&mut self) -> PackedMut<'_, T> {
        PackedMut::new(self.ptr, self.n, self.kind)
    }
}

unsafe impl<'a, T: Scalar + Send> Send for PackedMut<'a, T> {}
unsafe impl<'a, T: Scalar + Sync> Sync for PackedMut<'a, T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packed_upper_indexing() {
        // For a 3x3 upper triangular matrix:
        // [0  1  3]
        // [*  2  4]
        // [*  *  5]
        // Packed: [a00, a01, a11, a02, a12, a22]
        let mut p: PackedMat<f64> = PackedMat::zeros(3, TriangularKind::Upper);

        // Check indices
        assert_eq!(p.packed_index(0, 0), Some(0));
        assert_eq!(p.packed_index(0, 1), Some(1));
        assert_eq!(p.packed_index(1, 1), Some(2));
        assert_eq!(p.packed_index(0, 2), Some(3));
        assert_eq!(p.packed_index(1, 2), Some(4));
        assert_eq!(p.packed_index(2, 2), Some(5));

        // Below diagonal should return None
        assert_eq!(p.packed_index(1, 0), None);
        assert_eq!(p.packed_index(2, 0), None);
        assert_eq!(p.packed_index(2, 1), None);

        // Set and get values
        p.set(0, 0, 1.0);
        p.set(0, 1, 2.0);
        p.set(1, 1, 3.0);
        p.set(0, 2, 4.0);
        p.set(1, 2, 5.0);
        p.set(2, 2, 6.0);

        assert_eq!(p.get(0, 0), Some(&1.0));
        assert_eq!(p.get(0, 1), Some(&2.0));
        assert_eq!(p.get(1, 1), Some(&3.0));
        assert_eq!(p.get(0, 2), Some(&4.0));
        assert_eq!(p.get(1, 2), Some(&5.0));
        assert_eq!(p.get(2, 2), Some(&6.0));
    }

    #[test]
    fn test_packed_lower_indexing() {
        // For a 3x3 lower triangular matrix:
        // [0  *  *]
        // [1  3  *]
        // [2  4  5]
        // Packed: [a00, a10, a20, a11, a21, a22]
        let mut p: PackedMat<f64> = PackedMat::zeros(3, TriangularKind::Lower);

        // Check indices
        assert_eq!(p.packed_index(0, 0), Some(0));
        assert_eq!(p.packed_index(1, 0), Some(1));
        assert_eq!(p.packed_index(2, 0), Some(2));
        assert_eq!(p.packed_index(1, 1), Some(3));
        assert_eq!(p.packed_index(2, 1), Some(4));
        assert_eq!(p.packed_index(2, 2), Some(5));

        // Above diagonal should return None
        assert_eq!(p.packed_index(0, 1), None);
        assert_eq!(p.packed_index(0, 2), None);
        assert_eq!(p.packed_index(1, 2), None);

        // Set and get values
        p.set(0, 0, 1.0);
        p.set(1, 0, 2.0);
        p.set(2, 0, 3.0);
        p.set(1, 1, 4.0);
        p.set(2, 1, 5.0);
        p.set(2, 2, 6.0);

        assert_eq!(p.get(0, 0), Some(&1.0));
        assert_eq!(p.get(1, 0), Some(&2.0));
        assert_eq!(p.get(2, 0), Some(&3.0));
        assert_eq!(p.get(1, 1), Some(&4.0));
        assert_eq!(p.get(2, 1), Some(&5.0));
        assert_eq!(p.get(2, 2), Some(&6.0));
    }

    #[test]
    fn test_packed_len() {
        assert_eq!(PackedMat::<f64>::packed_len(0), 0);
        assert_eq!(PackedMat::<f64>::packed_len(1), 1);
        assert_eq!(PackedMat::<f64>::packed_len(2), 3);
        assert_eq!(PackedMat::<f64>::packed_len(3), 6);
        assert_eq!(PackedMat::<f64>::packed_len(4), 10);
        assert_eq!(PackedMat::<f64>::packed_len(10), 55);
    }

    #[test]
    fn test_packed_to_dense() {
        let mut p: PackedMat<f64> = PackedMat::zeros(3, TriangularKind::Upper);
        p.set(0, 0, 1.0);
        p.set(0, 1, 2.0);
        p.set(1, 1, 3.0);
        p.set(0, 2, 4.0);
        p.set(1, 2, 5.0);
        p.set(2, 2, 6.0);

        let dense = p.to_dense();
        assert_eq!(dense[(0, 0)], 1.0);
        assert_eq!(dense[(0, 1)], 2.0);
        assert_eq!(dense[(1, 1)], 3.0);
        assert_eq!(dense[(0, 2)], 4.0);
        assert_eq!(dense[(1, 2)], 5.0);
        assert_eq!(dense[(2, 2)], 6.0);

        // Below diagonal should be zero
        assert_eq!(dense[(1, 0)], 0.0);
        assert_eq!(dense[(2, 0)], 0.0);
        assert_eq!(dense[(2, 1)], 0.0);
    }

    #[test]
    fn test_packed_from_dense() {
        use crate::Mat;

        let dense = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        let upper = PackedMat::from_dense(&dense.as_ref(), TriangularKind::Upper);
        assert_eq!(upper.get(0, 0), Some(&1.0));
        assert_eq!(upper.get(0, 1), Some(&2.0));
        assert_eq!(upper.get(0, 2), Some(&3.0));
        assert_eq!(upper.get(1, 1), Some(&5.0));
        assert_eq!(upper.get(1, 2), Some(&6.0));
        assert_eq!(upper.get(2, 2), Some(&9.0));

        let lower = PackedMat::from_dense(&dense.as_ref(), TriangularKind::Lower);
        assert_eq!(lower.get(0, 0), Some(&1.0));
        assert_eq!(lower.get(1, 0), Some(&4.0));
        assert_eq!(lower.get(2, 0), Some(&7.0));
        assert_eq!(lower.get(1, 1), Some(&5.0));
        assert_eq!(lower.get(2, 1), Some(&8.0));
        assert_eq!(lower.get(2, 2), Some(&9.0));
    }

    #[test]
    fn test_packed_diagonal() {
        let mut p: PackedMat<f64> = PackedMat::zeros(3, TriangularKind::Upper);
        p.set(0, 0, 1.0);
        p.set(0, 1, 10.0);
        p.set(1, 1, 2.0);
        p.set(0, 2, 20.0);
        p.set(1, 2, 30.0);
        p.set(2, 2, 3.0);

        let diag = p.diagonal();
        assert_eq!(diag, vec![1.0, 2.0, 3.0]);

        // Set diagonal
        p.set_diagonal(&[10.0, 20.0, 30.0]);
        let diag2 = p.diagonal();
        assert_eq!(diag2, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_packed_transpose() {
        let mut upper: PackedMat<f64> = PackedMat::zeros(3, TriangularKind::Upper);
        upper.set(0, 0, 1.0);
        upper.set(0, 1, 2.0);
        upper.set(1, 1, 3.0);
        upper.set(0, 2, 4.0);
        upper.set(1, 2, 5.0);
        upper.set(2, 2, 6.0);

        let lower = upper.transpose();
        assert_eq!(lower.kind(), TriangularKind::Lower);

        // Transposed elements should be swapped
        assert_eq!(lower.get(0, 0), Some(&1.0));
        assert_eq!(lower.get(1, 0), Some(&2.0)); // Was (0, 1)
        assert_eq!(lower.get(1, 1), Some(&3.0));
        assert_eq!(lower.get(2, 0), Some(&4.0)); // Was (0, 2)
        assert_eq!(lower.get(2, 1), Some(&5.0)); // Was (1, 2)
        assert_eq!(lower.get(2, 2), Some(&6.0));
    }

    #[test]
    fn test_packed_ref() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let pref = PackedRef::from_slice(&data, 3, TriangularKind::Upper);

        assert_eq!(pref.dim(), 3);
        assert_eq!(pref.get(0, 0), Some(&1.0));
        assert_eq!(pref.get(0, 1), Some(&2.0));
        assert_eq!(pref.get(1, 1), Some(&3.0));
        assert_eq!(pref.get(0, 2), Some(&4.0));
        assert_eq!(pref.get(1, 2), Some(&5.0));
        assert_eq!(pref.get(2, 2), Some(&6.0));
    }

    #[test]
    fn test_packed_mut() {
        let mut data = [0.0f64; 6];
        let mut pmut = PackedMut::from_slice(&mut data, 3, TriangularKind::Lower);

        pmut.set(0, 0, 1.0);
        pmut.set(1, 0, 2.0);
        pmut.set(2, 0, 3.0);
        pmut.set(1, 1, 4.0);
        pmut.set(2, 1, 5.0);
        pmut.set(2, 2, 6.0);

        assert_eq!(data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_packed_scale() {
        let mut p: PackedMat<f64> = PackedMat::zeros(2, TriangularKind::Upper);
        p.set(0, 0, 1.0);
        p.set(0, 1, 2.0);
        p.set(1, 1, 3.0);

        p.scale(2.0);

        assert_eq!(p.get(0, 0), Some(&2.0));
        assert_eq!(p.get(0, 1), Some(&4.0));
        assert_eq!(p.get(1, 1), Some(&6.0));
    }
}
