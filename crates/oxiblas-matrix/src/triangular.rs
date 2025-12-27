//! Triangular matrix types.
//!
//! Triangular matrices are square matrices where all elements above (lower)
//! or below (upper) the main diagonal are zero. They arise frequently in
//! matrix decompositions (LU, Cholesky, QR) and are important in BLAS.
//!
//! # Storage
//!
//! This module provides two storage options:
//! - Full storage: Uses a standard dense matrix, treating the non-stored
//!   triangle as zeros implicitly.
//! - Packed storage: Uses `PackedMat` to store only the triangular portion.
//!
//! # Unit Triangular
//!
//! A unit triangular matrix has ones on the diagonal. This is common in LU
//! decomposition where L is unit lower triangular.

use crate::packed::{PackedMat, PackedMut, PackedRef, TriangularKind};
use crate::{Mat, MatMut, MatRef};
use oxiblas_core::scalar::Scalar;

/// Diagonal type for triangular matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagonalKind {
    /// Non-unit diagonal (general values on diagonal).
    NonUnit,
    /// Unit diagonal (ones on diagonal, not stored).
    Unit,
}

/// A triangular matrix view over dense storage.
///
/// This is a logical view that treats the non-stored triangle as zeros.
/// The underlying storage is a full dense matrix, but operations only
/// access the stored triangle.
///
/// # Example
///
/// ```
/// use oxiblas_matrix::{Mat, triangular::{TriangularView, DiagonalKind}};
/// use oxiblas_matrix::packed::TriangularKind;
///
/// let mut m: Mat<f64> = Mat::eye(3);
/// m[(0, 1)] = 2.0;
/// m[(0, 2)] = 3.0;
/// m[(1, 2)] = 4.0;
///
/// // Create upper triangular view
/// let tri = TriangularView::new(m.as_ref(), TriangularKind::Upper, DiagonalKind::NonUnit);
///
/// // Access upper triangle
/// assert_eq!(tri.get(0, 1), Some(&2.0));
/// // Lower triangle returns zero
/// assert_eq!(tri.get(1, 0), None);
/// ```
#[derive(Clone, Copy)]
pub struct TriangularView<'a, T: Scalar> {
    /// Underlying matrix view.
    inner: MatRef<'a, T>,
    /// Upper or lower triangular.
    uplo: TriangularKind,
    /// Unit or non-unit diagonal.
    diag: DiagonalKind,
}

impl<'a, T: Scalar> TriangularView<'a, T> {
    /// Creates a new triangular view over a matrix.
    ///
    /// # Panics
    /// Panics if the matrix is not square.
    #[inline]
    pub fn new(mat: MatRef<'a, T>, uplo: TriangularKind, diag: DiagonalKind) -> Self {
        assert!(mat.is_square(), "Triangular matrix must be square");
        TriangularView {
            inner: mat,
            uplo,
            diag,
        }
    }

    /// Returns the matrix dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.inner.nrows()
    }

    /// Returns the shape (n, n).
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    /// Returns the triangular kind (upper/lower).
    #[inline]
    pub fn uplo(&self) -> TriangularKind {
        self.uplo
    }

    /// Returns the diagonal kind (unit/non-unit).
    #[inline]
    pub fn diag(&self) -> DiagonalKind {
        self.diag
    }

    /// Returns true if element (row, col) is in the stored triangle.
    #[inline]
    pub fn in_triangle(&self, row: usize, col: usize) -> bool {
        match self.uplo {
            TriangularKind::Upper => row <= col,
            TriangularKind::Lower => row >= col,
        }
    }

    /// Returns a reference to the element at (row, col).
    ///
    /// Returns `None` if the element is in the non-stored triangle.
    /// For unit triangular matrices, diagonal elements return `None`
    /// (they are implicitly one).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if !self.in_triangle(row, col) {
            return None;
        }

        if self.diag == DiagonalKind::Unit && row == col {
            return None; // Diagonal is implicitly one
        }

        self.inner.get(row, col)
    }

    /// Returns the underlying matrix reference.
    #[inline]
    pub fn as_inner(&self) -> MatRef<'a, T> {
        self.inner
    }

    /// Returns a pointer to the matrix data.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.inner.as_ptr()
    }

    /// Returns the row stride.
    #[inline]
    pub fn row_stride(&self) -> usize {
        self.inner.row_stride()
    }

    /// Converts to a full dense matrix.
    pub fn to_dense(&self) -> Mat<T>
    where
        T: bytemuck::Zeroable,
    {
        let n = self.dim();
        let mut mat = Mat::zeros(n, n);

        for j in 0..n {
            for i in 0..n {
                if self.in_triangle(i, j) {
                    if self.diag == DiagonalKind::Unit && i == j {
                        mat[(i, j)] = T::one();
                    } else {
                        mat[(i, j)] = self.inner[(i, j)];
                    }
                }
            }
        }

        mat
    }

    /// Converts to packed storage.
    pub fn to_packed(&self) -> PackedMat<T>
    where
        T: bytemuck::Zeroable,
    {
        PackedMat::from_dense(&self.inner, self.uplo)
    }
}

/// A mutable triangular matrix view over dense storage.
pub struct TriangularViewMut<'a, T: Scalar> {
    /// Underlying mutable matrix view.
    inner: MatMut<'a, T>,
    /// Upper or lower triangular.
    uplo: TriangularKind,
    /// Unit or non-unit diagonal.
    diag: DiagonalKind,
}

impl<'a, T: Scalar> TriangularViewMut<'a, T> {
    /// Creates a new mutable triangular view over a matrix.
    #[inline]
    pub fn new(mat: MatMut<'a, T>, uplo: TriangularKind, diag: DiagonalKind) -> Self {
        assert!(mat.is_square(), "Triangular matrix must be square");
        TriangularViewMut {
            inner: mat,
            uplo,
            diag,
        }
    }

    /// Returns the matrix dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.inner.nrows()
    }

    /// Returns the shape (n, n).
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    /// Returns the triangular kind.
    #[inline]
    pub fn uplo(&self) -> TriangularKind {
        self.uplo
    }

    /// Returns the diagonal kind.
    #[inline]
    pub fn diag(&self) -> DiagonalKind {
        self.diag
    }

    /// Returns true if element is in stored triangle.
    #[inline]
    pub fn in_triangle(&self, row: usize, col: usize) -> bool {
        match self.uplo {
            TriangularKind::Upper => row <= col,
            TriangularKind::Lower => row >= col,
        }
    }

    /// Returns a reference to element.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if !self.in_triangle(row, col) {
            return None;
        }
        if self.diag == DiagonalKind::Unit && row == col {
            return None;
        }
        self.inner.get(row, col)
    }

    /// Returns a mutable reference to element.
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if !self.in_triangle(row, col) {
            return None;
        }
        if self.diag == DiagonalKind::Unit && row == col {
            return None;
        }
        self.inner.get_mut(row, col)
    }

    /// Sets an element in the stored triangle.
    ///
    /// # Panics
    /// Panics if the element is outside the stored triangle or on the
    /// diagonal for unit triangular matrices.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        assert!(
            self.in_triangle(row, col),
            "Element outside stored triangle"
        );
        assert!(
            !(self.diag == DiagonalKind::Unit && row == col),
            "Cannot set diagonal of unit triangular matrix"
        );
        self.inner.set(row, col, value);
    }

    /// Creates an immutable reborrow.
    #[inline]
    pub fn rb(&self) -> TriangularView<'_, T> {
        TriangularView {
            inner: self.inner.rb(),
            uplo: self.uplo,
            diag: self.diag,
        }
    }

    /// Creates a mutable reborrow.
    #[inline]
    pub fn rb_mut(&mut self) -> TriangularViewMut<'_, T> {
        TriangularViewMut {
            inner: self.inner.rb_mut(),
            uplo: self.uplo,
            diag: self.diag,
        }
    }

    /// Returns a mutable pointer to the data.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.inner.as_mut_ptr()
    }

    /// Fills the stored triangle with a value.
    ///
    /// Does not modify the diagonal for unit triangular matrices.
    pub fn fill(&mut self, value: T) {
        let n = self.dim();
        for j in 0..n {
            for i in 0..n {
                if self.in_triangle(i, j) {
                    if self.diag == DiagonalKind::Unit && i == j {
                        continue;
                    }
                    self.inner.set(i, j, value);
                }
            }
        }
    }

    /// Scales the stored triangle by a scalar.
    pub fn scale(&mut self, alpha: T) {
        let n = self.dim();
        for j in 0..n {
            for i in 0..n {
                if self.in_triangle(i, j) {
                    if self.diag == DiagonalKind::Unit && i == j {
                        continue;
                    }
                    if let Some(val) = self.inner.get(i, j) {
                        self.inner.set(i, j, *val * alpha);
                    }
                }
            }
        }
    }

    /// Clears the non-stored triangle to zero.
    pub fn zero_non_triangle(&mut self)
    where
        T: num_traits::Zero,
    {
        let n = self.dim();
        for j in 0..n {
            for i in 0..n {
                if !self.in_triangle(i, j) {
                    self.inner.set(i, j, T::zero());
                }
            }
        }
    }
}

/// Triangular matrix using packed storage.
///
/// This is a wrapper around `PackedMat` that provides triangular matrix
/// semantics with efficient packed storage.
#[derive(Clone)]
pub struct TriangularMat<T: Scalar> {
    /// Packed storage.
    packed: PackedMat<T>,
    /// Unit or non-unit diagonal.
    diag: DiagonalKind,
}

impl<T: Scalar> TriangularMat<T> {
    /// Creates a new triangular matrix filled with zeros.
    pub fn zeros(n: usize, uplo: TriangularKind, diag: DiagonalKind) -> Self
    where
        T: bytemuck::Zeroable,
    {
        TriangularMat {
            packed: PackedMat::zeros(n, uplo),
            diag,
        }
    }

    /// Creates a unit triangular matrix (identity-like).
    ///
    /// The diagonal is implicitly one and not stored.
    pub fn unit_zeros(n: usize, uplo: TriangularKind) -> Self
    where
        T: bytemuck::Zeroable,
    {
        Self::zeros(n, uplo, DiagonalKind::Unit)
    }

    /// Creates from a packed matrix.
    #[inline]
    pub fn from_packed(packed: PackedMat<T>, diag: DiagonalKind) -> Self {
        TriangularMat { packed, diag }
    }

    /// Creates from a dense matrix view.
    pub fn from_dense(mat: &MatRef<'_, T>, uplo: TriangularKind, diag: DiagonalKind) -> Self
    where
        T: bytemuck::Zeroable,
    {
        TriangularMat {
            packed: PackedMat::from_dense(mat, uplo),
            diag,
        }
    }

    /// Returns the matrix dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.packed.dim()
    }

    /// Returns the shape (n, n).
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        let n = self.dim();
        (n, n)
    }

    /// Returns the triangular kind.
    #[inline]
    pub fn uplo(&self) -> TriangularKind {
        self.packed.kind()
    }

    /// Returns the diagonal kind.
    #[inline]
    pub fn diag(&self) -> DiagonalKind {
        self.diag
    }

    /// Returns the packed storage length.
    #[inline]
    pub fn len(&self) -> usize {
        self.packed.len()
    }

    /// Returns true if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.packed.is_empty()
    }

    /// Returns true if element is in the stored triangle.
    #[inline]
    pub fn in_triangle(&self, row: usize, col: usize) -> bool {
        self.packed.packed_index(row, col).is_some()
    }

    /// Returns a reference to element.
    ///
    /// For unit triangular matrices, returns `None` for diagonal elements.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if self.diag == DiagonalKind::Unit && row == col {
            return None;
        }
        self.packed.get(row, col)
    }

    /// Returns a mutable reference to element.
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if self.diag == DiagonalKind::Unit && row == col {
            return None;
        }
        self.packed.get_mut(row, col)
    }

    /// Sets an element.
    ///
    /// # Panics
    /// Panics if setting diagonal on unit triangular matrix.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        assert!(
            !(self.diag == DiagonalKind::Unit && row == col),
            "Cannot set diagonal of unit triangular matrix"
        );
        self.packed.set(row, col, value);
    }

    /// Returns a pointer to the packed data.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.packed.as_ptr()
    }

    /// Returns a mutable pointer to the packed data.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.packed.as_mut_ptr()
    }

    /// Returns the packed data as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        self.packed.as_slice()
    }

    /// Returns the packed data as a mutable slice.
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        self.packed.as_slice_mut()
    }

    /// Returns a reference to the underlying packed matrix.
    #[inline]
    pub fn as_packed(&self) -> &PackedMat<T> {
        &self.packed
    }

    /// Returns a mutable reference to the underlying packed matrix.
    #[inline]
    pub fn as_packed_mut(&mut self) -> &mut PackedMat<T> {
        &mut self.packed
    }

    /// Converts to a full dense matrix.
    pub fn to_dense(&self) -> Mat<T>
    where
        T: bytemuck::Zeroable,
    {
        let n = self.dim();
        let mut mat = Mat::zeros(n, n);

        for j in 0..n {
            for i in 0..n {
                if let Some(&val) = self.packed.get(i, j) {
                    if self.diag == DiagonalKind::Unit && i == j {
                        mat[(i, j)] = T::one();
                    } else {
                        mat[(i, j)] = val;
                    }
                } else if self.diag == DiagonalKind::Unit && i == j {
                    mat[(i, j)] = T::one();
                }
            }
        }

        mat
    }

    /// Returns the diagonal elements.
    ///
    /// For unit triangular matrices, returns a vector of ones.
    pub fn diagonal(&self) -> Vec<T> {
        let n = self.dim();
        if self.diag == DiagonalKind::Unit {
            vec![T::one(); n]
        } else {
            self.packed.diagonal()
        }
    }

    /// Sets the diagonal elements.
    ///
    /// # Panics
    /// Panics for unit triangular matrices.
    pub fn set_diagonal(&mut self, diag: &[T]) {
        assert!(
            self.diag != DiagonalKind::Unit,
            "Cannot set diagonal of unit triangular matrix"
        );
        self.packed.set_diagonal(diag);
    }

    /// Fills the stored triangle with a value.
    pub fn fill(&mut self, value: T) {
        if self.diag == DiagonalKind::Unit {
            // Fill off-diagonal only
            let n = self.dim();
            for j in 0..n {
                for i in 0..n {
                    if self.in_triangle(i, j) && i != j {
                        self.packed.set(i, j, value);
                    }
                }
            }
        } else {
            self.packed.fill(value);
        }
    }

    /// Scales the stored triangle by a scalar.
    pub fn scale(&mut self, alpha: T) {
        if self.diag == DiagonalKind::Unit {
            // Scale off-diagonal only
            let n = self.dim();
            for j in 0..n {
                for i in 0..n {
                    if self.in_triangle(i, j) && i != j {
                        if let Some(val) = self.packed.get_mut(i, j) {
                            *val *= alpha;
                        }
                    }
                }
            }
        } else {
            self.packed.scale(alpha);
        }
    }

    /// Returns the transpose (flips upper/lower).
    pub fn transpose(&self) -> Self
    where
        T: bytemuck::Zeroable,
    {
        TriangularMat {
            packed: self.packed.transpose(),
            diag: self.diag,
        }
    }
}

impl<T: Scalar + core::fmt::Debug> core::fmt::Debug for TriangularMat<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let n = self.dim();
        writeln!(
            f,
            "TriangularMat {}×{} ({:?}, {:?}) {{",
            n,
            n,
            self.uplo(),
            self.diag
        )?;

        let max_dim = 8.min(n);
        for i in 0..max_dim {
            write!(f, "  [")?;
            for j in 0..max_dim {
                if j > 0 {
                    write!(f, ", ")?;
                }
                if self.in_triangle(i, j) {
                    if self.diag == DiagonalKind::Unit && i == j {
                        write!(f, "{:8.4?}", T::one())?;
                    } else if let Some(v) = self.packed.get(i, j) {
                        write!(f, "{:8.4?}", v)?;
                    } else {
                        write!(f, "      * ")?;
                    }
                } else {
                    write!(f, "      0 ")?;
                }
            }
            if n > max_dim {
                write!(f, ", ...")?;
            }
            writeln!(f, "]")?;
        }
        if n > max_dim {
            writeln!(f, "  ...")?;
        }
        write!(f, "}}")
    }
}

/// A reference to triangular packed data.
#[derive(Clone, Copy)]
pub struct TriangularRef<'a, T: Scalar> {
    /// Packed reference.
    packed: PackedRef<'a, T>,
    /// Diagonal kind.
    diag: DiagonalKind,
}

impl<'a, T: Scalar> TriangularRef<'a, T> {
    /// Creates a new triangular reference.
    #[inline]
    pub fn new(packed: PackedRef<'a, T>, diag: DiagonalKind) -> Self {
        TriangularRef { packed, diag }
    }

    /// Creates from a slice.
    #[inline]
    pub fn from_slice(data: &'a [T], n: usize, uplo: TriangularKind, diag: DiagonalKind) -> Self {
        TriangularRef {
            packed: PackedRef::from_slice(data, n, uplo),
            diag,
        }
    }

    /// Returns the dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.packed.dim()
    }

    /// Returns the triangular kind.
    #[inline]
    pub fn uplo(&self) -> TriangularKind {
        self.packed.kind()
    }

    /// Returns the diagonal kind.
    #[inline]
    pub fn diag(&self) -> DiagonalKind {
        self.diag
    }

    /// Returns element at (row, col).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if self.diag == DiagonalKind::Unit && row == col {
            return None;
        }
        self.packed.get(row, col)
    }

    /// Returns the underlying packed reference.
    #[inline]
    pub fn as_packed(&self) -> PackedRef<'a, T> {
        self.packed
    }
}

/// A mutable reference to triangular packed data.
pub struct TriangularMut<'a, T: Scalar> {
    /// Packed mutable reference.
    packed: PackedMut<'a, T>,
    /// Diagonal kind.
    diag: DiagonalKind,
}

impl<'a, T: Scalar> TriangularMut<'a, T> {
    /// Creates a new mutable triangular reference.
    #[inline]
    pub fn new(packed: PackedMut<'a, T>, diag: DiagonalKind) -> Self {
        TriangularMut { packed, diag }
    }

    /// Creates from a mutable slice.
    #[inline]
    pub fn from_slice(
        data: &'a mut [T],
        n: usize,
        uplo: TriangularKind,
        diag: DiagonalKind,
    ) -> Self {
        TriangularMut {
            packed: PackedMut::from_slice(data, n, uplo),
            diag,
        }
    }

    /// Returns the dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.packed.dim()
    }

    /// Returns the triangular kind.
    #[inline]
    pub fn uplo(&self) -> TriangularKind {
        self.packed.kind()
    }

    /// Returns the diagonal kind.
    #[inline]
    pub fn diag(&self) -> DiagonalKind {
        self.diag
    }

    /// Returns element at (row, col).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if self.diag == DiagonalKind::Unit && row == col {
            return None;
        }
        self.packed.get(row, col)
    }

    /// Returns mutable element at (row, col).
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if self.diag == DiagonalKind::Unit && row == col {
            return None;
        }
        self.packed.get_mut(row, col)
    }

    /// Sets element at (row, col).
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        assert!(
            !(self.diag == DiagonalKind::Unit && row == col),
            "Cannot set diagonal of unit triangular matrix"
        );
        self.packed.set(row, col, value);
    }

    /// Creates an immutable reborrow.
    #[inline]
    pub fn rb(&self) -> TriangularRef<'_, T> {
        TriangularRef {
            packed: self.packed.rb(),
            diag: self.diag,
        }
    }

    /// Creates a mutable reborrow.
    #[inline]
    pub fn rb_mut(&mut self) -> TriangularMut<'_, T> {
        TriangularMut {
            packed: self.packed.rb_mut(),
            diag: self.diag,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangular_view_upper() {
        let mut m: Mat<f64> = Mat::zeros(3, 3);
        m[(0, 0)] = 1.0;
        m[(0, 1)] = 2.0;
        m[(0, 2)] = 3.0;
        m[(1, 1)] = 4.0;
        m[(1, 2)] = 5.0;
        m[(2, 2)] = 6.0;
        // Set some values below diagonal (should be ignored)
        m[(1, 0)] = 99.0;
        m[(2, 0)] = 99.0;
        m[(2, 1)] = 99.0;

        let tri = TriangularView::new(m.as_ref(), TriangularKind::Upper, DiagonalKind::NonUnit);

        // Upper triangle
        assert_eq!(tri.get(0, 0), Some(&1.0));
        assert_eq!(tri.get(0, 1), Some(&2.0));
        assert_eq!(tri.get(0, 2), Some(&3.0));
        assert_eq!(tri.get(1, 1), Some(&4.0));
        assert_eq!(tri.get(1, 2), Some(&5.0));
        assert_eq!(tri.get(2, 2), Some(&6.0));

        // Lower triangle returns None
        assert_eq!(tri.get(1, 0), None);
        assert_eq!(tri.get(2, 0), None);
        assert_eq!(tri.get(2, 1), None);

        // Test to_dense
        let dense = tri.to_dense();
        assert_eq!(dense[(0, 0)], 1.0);
        assert_eq!(dense[(0, 1)], 2.0);
        assert_eq!(dense[(1, 0)], 0.0); // Zero, not 99
        assert_eq!(dense[(2, 1)], 0.0);
    }

    #[test]
    fn test_triangular_view_unit() {
        let mut m: Mat<f64> = Mat::zeros(3, 3);
        m[(0, 0)] = 99.0; // Should be treated as 1
        m[(0, 1)] = 2.0;
        m[(0, 2)] = 3.0;
        m[(1, 1)] = 99.0; // Should be treated as 1
        m[(1, 2)] = 4.0;
        m[(2, 2)] = 99.0; // Should be treated as 1

        let tri = TriangularView::new(m.as_ref(), TriangularKind::Upper, DiagonalKind::Unit);

        // Diagonal returns None (implicitly one)
        assert_eq!(tri.get(0, 0), None);
        assert_eq!(tri.get(1, 1), None);
        assert_eq!(tri.get(2, 2), None);

        // Off-diagonal upper elements
        assert_eq!(tri.get(0, 1), Some(&2.0));
        assert_eq!(tri.get(0, 2), Some(&3.0));
        assert_eq!(tri.get(1, 2), Some(&4.0));

        // to_dense should have ones on diagonal
        let dense = tri.to_dense();
        assert_eq!(dense[(0, 0)], 1.0);
        assert_eq!(dense[(1, 1)], 1.0);
        assert_eq!(dense[(2, 2)], 1.0);
        assert_eq!(dense[(0, 1)], 2.0);
    }

    #[test]
    fn test_triangular_view_lower() {
        let mut m: Mat<f64> = Mat::zeros(3, 3);
        m[(0, 0)] = 1.0;
        m[(1, 0)] = 2.0;
        m[(1, 1)] = 3.0;
        m[(2, 0)] = 4.0;
        m[(2, 1)] = 5.0;
        m[(2, 2)] = 6.0;

        let tri = TriangularView::new(m.as_ref(), TriangularKind::Lower, DiagonalKind::NonUnit);

        // Lower triangle
        assert_eq!(tri.get(0, 0), Some(&1.0));
        assert_eq!(tri.get(1, 0), Some(&2.0));
        assert_eq!(tri.get(1, 1), Some(&3.0));
        assert_eq!(tri.get(2, 0), Some(&4.0));
        assert_eq!(tri.get(2, 1), Some(&5.0));
        assert_eq!(tri.get(2, 2), Some(&6.0));

        // Upper triangle returns None
        assert_eq!(tri.get(0, 1), None);
        assert_eq!(tri.get(0, 2), None);
        assert_eq!(tri.get(1, 2), None);
    }

    #[test]
    fn test_triangular_mat_packed() {
        let mut tri: TriangularMat<f64> =
            TriangularMat::zeros(3, TriangularKind::Upper, DiagonalKind::NonUnit);

        tri.set(0, 0, 1.0);
        tri.set(0, 1, 2.0);
        tri.set(0, 2, 3.0);
        tri.set(1, 1, 4.0);
        tri.set(1, 2, 5.0);
        tri.set(2, 2, 6.0);

        assert_eq!(tri.get(0, 0), Some(&1.0));
        assert_eq!(tri.get(0, 1), Some(&2.0));
        assert_eq!(tri.get(1, 2), Some(&5.0));

        // Cannot access lower triangle
        assert_eq!(tri.get(1, 0), None);

        let diag = tri.diagonal();
        assert_eq!(diag, vec![1.0, 4.0, 6.0]);
    }

    #[test]
    fn test_triangular_mat_unit() {
        let mut tri: TriangularMat<f64> = TriangularMat::unit_zeros(3, TriangularKind::Lower);

        // Off-diagonal elements
        tri.set(1, 0, 2.0);
        tri.set(2, 0, 3.0);
        tri.set(2, 1, 4.0);

        // Diagonal returns None (implicit one)
        assert_eq!(tri.get(0, 0), None);
        assert_eq!(tri.get(1, 1), None);
        assert_eq!(tri.get(2, 2), None);

        // Off-diagonal elements
        assert_eq!(tri.get(1, 0), Some(&2.0));
        assert_eq!(tri.get(2, 0), Some(&3.0));
        assert_eq!(tri.get(2, 1), Some(&4.0));

        // Diagonal should be ones
        let diag = tri.diagonal();
        assert_eq!(diag, vec![1.0, 1.0, 1.0]);

        // to_dense
        let dense = tri.to_dense();
        assert_eq!(dense[(0, 0)], 1.0);
        assert_eq!(dense[(1, 1)], 1.0);
        assert_eq!(dense[(2, 2)], 1.0);
        assert_eq!(dense[(1, 0)], 2.0);
        assert_eq!(dense[(0, 1)], 0.0);
    }

    #[test]
    fn test_triangular_mat_transpose() {
        let mut tri: TriangularMat<f64> =
            TriangularMat::zeros(3, TriangularKind::Upper, DiagonalKind::NonUnit);

        tri.set(0, 0, 1.0);
        tri.set(0, 1, 2.0);
        tri.set(0, 2, 3.0);
        tri.set(1, 1, 4.0);
        tri.set(1, 2, 5.0);
        tri.set(2, 2, 6.0);

        let tri_t = tri.transpose();
        assert_eq!(tri_t.uplo(), TriangularKind::Lower);

        // Elements should be transposed
        assert_eq!(tri_t.get(0, 0), Some(&1.0));
        assert_eq!(tri_t.get(1, 0), Some(&2.0)); // Was (0, 1)
        assert_eq!(tri_t.get(2, 0), Some(&3.0)); // Was (0, 2)
        assert_eq!(tri_t.get(1, 1), Some(&4.0));
        assert_eq!(tri_t.get(2, 1), Some(&5.0)); // Was (1, 2)
        assert_eq!(tri_t.get(2, 2), Some(&6.0));
    }

    #[test]
    fn test_triangular_view_mut() {
        let mut m: Mat<f64> = Mat::zeros(3, 3);
        let mut view =
            TriangularViewMut::new(m.as_mut(), TriangularKind::Upper, DiagonalKind::NonUnit);

        view.set(0, 0, 1.0);
        view.set(0, 1, 2.0);
        view.set(1, 1, 3.0);

        assert_eq!(view.get(0, 0), Some(&1.0));
        assert_eq!(view.get(0, 1), Some(&2.0));
        assert_eq!(view.get(1, 1), Some(&3.0));

        // zero_non_triangle
        view.zero_non_triangle();
        // Check that lower triangle is zeroed (it was already zero)

        // Fill
        view.fill(5.0);
        assert_eq!(view.get(0, 0), Some(&5.0));
        assert_eq!(view.get(0, 1), Some(&5.0));
    }

    #[test]
    fn test_triangular_from_dense() {
        let m = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        let tri =
            TriangularMat::from_dense(&m.as_ref(), TriangularKind::Upper, DiagonalKind::NonUnit);

        assert_eq!(tri.get(0, 0), Some(&1.0));
        assert_eq!(tri.get(0, 1), Some(&2.0));
        assert_eq!(tri.get(0, 2), Some(&3.0));
        assert_eq!(tri.get(1, 1), Some(&5.0));
        assert_eq!(tri.get(1, 2), Some(&6.0));
        assert_eq!(tri.get(2, 2), Some(&9.0));

        // Lower part is not stored
        assert_eq!(tri.get(1, 0), None);
        assert_eq!(tri.get(2, 0), None);
        assert_eq!(tri.get(2, 1), None);
    }

    #[test]
    fn test_triangular_ref_mut() {
        let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut tmut =
            TriangularMut::from_slice(&mut data, 3, TriangularKind::Upper, DiagonalKind::NonUnit);

        assert_eq!(tmut.get(0, 0), Some(&1.0));
        assert_eq!(tmut.get(0, 1), Some(&2.0));

        tmut.set(0, 0, 10.0);
        assert_eq!(tmut.get(0, 0), Some(&10.0));
        assert_eq!(data[0], 10.0);
    }
}
