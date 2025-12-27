//! Symmetric and Hermitian matrix types.
//!
//! Symmetric matrices satisfy A = A^T, and Hermitian matrices satisfy A = A^H
//! (conjugate transpose). These matrices only need to store one triangle,
//! reducing storage by nearly half.
//!
//! # Storage
//!
//! Both symmetric and Hermitian matrices can use:
//! - Full storage: A dense matrix storing both triangles (with enforcement)
//! - Packed storage: Only the upper or lower triangle is stored
//!
//! # BLAS Compatibility
//!
//! The packed storage format is compatible with BLAS routines like DSPMV,
//! DSPR, DSPR2 (symmetric) and ZHPMV, ZHPR, ZHPR2 (Hermitian).

use crate::packed::{PackedMat, PackedMut, PackedRef, TriangularKind};
use crate::{Mat, MatMut, MatRef};
use num_complex::Complex;
use oxiblas_core::scalar::Scalar;

/// A symmetric matrix using packed storage.
///
/// For a symmetric matrix A, `A[i,j] = A[j,i]`. This implementation stores
/// only one triangle (upper or lower) and provides symmetric access.
///
/// # Example
///
/// ```
/// use oxiblas_matrix::symmetric::SymmetricMat;
/// use oxiblas_matrix::packed::TriangularKind;
///
/// let mut s: SymmetricMat<f64> = SymmetricMat::zeros(3, TriangularKind::Upper);
///
/// // Set elements (automatically symmetric)
/// s.set(0, 0, 1.0);
/// s.set(0, 1, 2.0);  // Also sets (1, 0)
/// s.set(1, 1, 3.0);
///
/// // Access is symmetric
/// assert_eq!(s.get(0, 1), Some(&2.0));
/// assert_eq!(s.get(1, 0), Some(&2.0));
/// ```
#[derive(Clone)]
pub struct SymmetricMat<T: Scalar> {
    /// Packed storage for one triangle.
    packed: PackedMat<T>,
}

impl<T: Scalar> SymmetricMat<T> {
    /// Creates a new symmetric matrix filled with zeros.
    pub fn zeros(n: usize, uplo: TriangularKind) -> Self
    where
        T: bytemuck::Zeroable,
    {
        SymmetricMat {
            packed: PackedMat::zeros(n, uplo),
        }
    }

    /// Creates a new symmetric matrix filled with a specific value.
    pub fn filled(n: usize, uplo: TriangularKind, value: T) -> Self {
        SymmetricMat {
            packed: PackedMat::filled(n, uplo, value),
        }
    }

    /// Creates from a packed matrix.
    #[inline]
    pub fn from_packed(packed: PackedMat<T>) -> Self {
        SymmetricMat { packed }
    }

    /// Creates from a dense matrix view.
    ///
    /// Only copies elements from the specified triangle. The caller is
    /// responsible for ensuring the matrix is actually symmetric.
    pub fn from_dense(mat: &MatRef<'_, T>, uplo: TriangularKind) -> Self
    where
        T: bytemuck::Zeroable,
    {
        assert!(mat.is_square(), "Matrix must be square");
        SymmetricMat {
            packed: PackedMat::from_dense(mat, uplo),
        }
    }

    /// Creates a symmetric matrix from a dense matrix, verifying symmetry.
    ///
    /// Returns `None` if the matrix is not symmetric within the given tolerance.
    pub fn from_dense_checked(
        mat: &MatRef<'_, T>,
        uplo: TriangularKind,
        tol: T::Real,
    ) -> Option<Self>
    where
        T: bytemuck::Zeroable,
    {
        assert!(mat.is_square(), "Matrix must be square");
        let n = mat.nrows();

        // Check symmetry
        for j in 0..n {
            for i in 0..j {
                let diff = mat[(i, j)] - mat[(j, i)];
                if diff.abs() > tol {
                    return None;
                }
            }
        }

        Some(Self::from_dense(mat, uplo))
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

    /// Returns the storage kind (upper or lower).
    #[inline]
    pub fn uplo(&self) -> TriangularKind {
        self.packed.kind()
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

    /// Returns a reference to element at (row, col).
    ///
    /// For symmetric access, (i, j) and (j, i) return the same element.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        // Map to the stored triangle
        match self.uplo() {
            TriangularKind::Upper => {
                if row <= col {
                    self.packed.get(row, col)
                } else {
                    self.packed.get(col, row)
                }
            }
            TriangularKind::Lower => {
                if row >= col {
                    self.packed.get(row, col)
                } else {
                    self.packed.get(col, row)
                }
            }
        }
    }

    /// Returns a mutable reference to element at (row, col).
    ///
    /// Note: Modifying this element affects both (row, col) and (col, row).
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        match self.uplo() {
            TriangularKind::Upper => {
                if row <= col {
                    self.packed.get_mut(row, col)
                } else {
                    self.packed.get_mut(col, row)
                }
            }
            TriangularKind::Lower => {
                if row >= col {
                    self.packed.get_mut(row, col)
                } else {
                    self.packed.get_mut(col, row)
                }
            }
        }
    }

    /// Sets the element at (row, col).
    ///
    /// This automatically sets (col, row) to the same value.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        match self.uplo() {
            TriangularKind::Upper => {
                if row <= col {
                    self.packed.set(row, col, value);
                } else {
                    self.packed.set(col, row, value);
                }
            }
            TriangularKind::Lower => {
                if row >= col {
                    self.packed.set(row, col, value);
                } else {
                    self.packed.set(col, row, value);
                }
            }
        }
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
                if let Some(&val) = self.get(i, j) {
                    mat[(i, j)] = val;
                }
            }
        }

        mat
    }

    /// Returns the diagonal elements.
    #[inline]
    pub fn diagonal(&self) -> Vec<T> {
        self.packed.diagonal()
    }

    /// Sets the diagonal elements.
    #[inline]
    pub fn set_diagonal(&mut self, diag: &[T]) {
        self.packed.set_diagonal(diag);
    }

    /// Fills all elements with a value.
    #[inline]
    pub fn fill(&mut self, value: T) {
        self.packed.fill(value);
    }

    /// Scales all elements by a scalar.
    #[inline]
    pub fn scale(&mut self, alpha: T) {
        self.packed.scale(alpha);
    }

    /// Adds a scalar to the diagonal (A = A + alpha * I).
    pub fn add_diagonal(&mut self, alpha: T) {
        let n = self.dim();
        for i in 0..n {
            if let Some(val) = self.get_mut(i, i) {
                *val += alpha;
            }
        }
    }
}

impl SymmetricMat<f32> {
    /// Returns the Frobenius norm squared.
    pub fn frobenius_norm_squared(&self) -> f32 {
        let n = self.dim();
        let mut sum = 0.0f32;

        for j in 0..n {
            for i in 0..n {
                if let Some(&val) = self.get(i, j) {
                    // For off-diagonal elements, count twice (since we only store once)
                    if i == j {
                        sum += val * val;
                    } else if (self.uplo() == TriangularKind::Upper && i < j)
                        || (self.uplo() == TriangularKind::Lower && i > j)
                    {
                        // Only count stored elements once, then double
                        sum += 2.0 * val * val;
                    }
                }
            }
        }

        sum
    }
}

impl SymmetricMat<f64> {
    /// Returns the Frobenius norm squared.
    pub fn frobenius_norm_squared(&self) -> f64 {
        let n = self.dim();
        let mut sum = 0.0f64;

        for j in 0..n {
            for i in 0..n {
                if let Some(&val) = self.get(i, j) {
                    // For off-diagonal elements, count twice (since we only store once)
                    if i == j {
                        sum += val * val;
                    } else if (self.uplo() == TriangularKind::Upper && i < j)
                        || (self.uplo() == TriangularKind::Lower && i > j)
                    {
                        // Only count stored elements once, then double
                        sum += 2.0 * val * val;
                    }
                }
            }
        }

        sum
    }
}

impl<T: Scalar + core::fmt::Debug> core::fmt::Debug for SymmetricMat<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let n = self.dim();
        writeln!(f, "SymmetricMat {}×{} ({:?}) {{", n, n, self.uplo())?;

        let max_dim = 8.min(n);
        for i in 0..max_dim {
            write!(f, "  [")?;
            for j in 0..max_dim {
                if j > 0 {
                    write!(f, ", ")?;
                }
                if let Some(v) = self.get(i, j) {
                    write!(f, "{:8.4?}", v)?;
                } else {
                    write!(f, "      * ")?;
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

/// A symmetric matrix view over dense storage.
///
/// This provides a symmetric view into a dense matrix, treating (i, j) and
/// (j, i) as the same element.
#[derive(Clone, Copy)]
pub struct SymmetricView<'a, T: Scalar> {
    /// Underlying matrix view.
    inner: MatRef<'a, T>,
    /// Which triangle is stored.
    uplo: TriangularKind,
}

impl<'a, T: Scalar> SymmetricView<'a, T> {
    /// Creates a new symmetric view.
    #[inline]
    pub fn new(mat: MatRef<'a, T>, uplo: TriangularKind) -> Self {
        assert!(mat.is_square(), "Symmetric matrix must be square");
        SymmetricView { inner: mat, uplo }
    }

    /// Returns the dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.inner.nrows()
    }

    /// Returns the stored triangle.
    #[inline]
    pub fn uplo(&self) -> TriangularKind {
        self.uplo
    }

    /// Returns element at (row, col) with symmetric access.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        let n = self.dim();
        if row >= n || col >= n {
            return None;
        }

        match self.uplo {
            TriangularKind::Upper => {
                if row <= col {
                    self.inner.get(row, col)
                } else {
                    self.inner.get(col, row)
                }
            }
            TriangularKind::Lower => {
                if row >= col {
                    self.inner.get(row, col)
                } else {
                    self.inner.get(col, row)
                }
            }
        }
    }

    /// Converts to packed storage.
    pub fn to_packed(&self) -> SymmetricMat<T>
    where
        T: bytemuck::Zeroable,
    {
        SymmetricMat::from_dense(&self.inner, self.uplo)
    }
}

/// A mutable symmetric matrix view over dense storage.
pub struct SymmetricViewMut<'a, T: Scalar> {
    /// Underlying mutable matrix view.
    inner: MatMut<'a, T>,
    /// Which triangle is stored.
    uplo: TriangularKind,
}

impl<'a, T: Scalar> SymmetricViewMut<'a, T> {
    /// Creates a new mutable symmetric view.
    #[inline]
    pub fn new(mat: MatMut<'a, T>, uplo: TriangularKind) -> Self {
        assert!(mat.is_square(), "Symmetric matrix must be square");
        SymmetricViewMut { inner: mat, uplo }
    }

    /// Returns the dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.inner.nrows()
    }

    /// Returns the stored triangle.
    #[inline]
    pub fn uplo(&self) -> TriangularKind {
        self.uplo
    }

    /// Returns element at (row, col).
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        let n = self.dim();
        if row >= n || col >= n {
            return None;
        }

        match self.uplo {
            TriangularKind::Upper => {
                if row <= col {
                    self.inner.get(row, col)
                } else {
                    self.inner.get(col, row)
                }
            }
            TriangularKind::Lower => {
                if row >= col {
                    self.inner.get(row, col)
                } else {
                    self.inner.get(col, row)
                }
            }
        }
    }

    /// Sets element at (row, col), maintaining symmetry.
    ///
    /// Sets both (row, col) and (col, row) to the same value.
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        let n = self.dim();
        assert!(row < n && col < n, "Index out of bounds");

        // Set both symmetric positions
        self.inner.set(row, col, value);
        if row != col {
            self.inner.set(col, row, value);
        }
    }

    /// Creates an immutable reborrow.
    #[inline]
    pub fn rb(&self) -> SymmetricView<'_, T> {
        SymmetricView {
            inner: self.inner.rb(),
            uplo: self.uplo,
        }
    }
}

/// A Hermitian matrix using packed storage.
///
/// For a Hermitian matrix A, `A[i,j] = conj(A[j,i])`. The diagonal elements
/// are real. This is the complex analogue of a symmetric matrix.
///
/// # Example
///
/// ```
/// use oxiblas_matrix::symmetric::HermitianMat;
/// use oxiblas_matrix::packed::TriangularKind;
/// use num_complex::Complex64;
///
/// let mut h: HermitianMat<Complex64> = HermitianMat::zeros(2, TriangularKind::Upper);
///
/// // Diagonal must be real
/// h.set(0, 0, Complex64::new(1.0, 0.0));
/// h.set(1, 1, Complex64::new(2.0, 0.0));
///
/// // Off-diagonal: (0,1) and (1,0) are conjugates
/// h.set(0, 1, Complex64::new(3.0, 4.0));
///
/// assert_eq!(h.get(0, 1), Some(&Complex64::new(3.0, 4.0)));
/// // (1, 0) returns the conjugate
/// // Note: Our implementation stores only one triangle, so get(1,0) returns
/// // the stored value which is conceptually the conjugate when used in operations
/// ```
#[derive(Clone)]
pub struct HermitianMat<T: Scalar> {
    /// Packed storage.
    packed: PackedMat<T>,
}

impl<T: Scalar> HermitianMat<T> {
    /// Creates a new Hermitian matrix filled with zeros.
    pub fn zeros(n: usize, uplo: TriangularKind) -> Self
    where
        T: bytemuck::Zeroable,
    {
        HermitianMat {
            packed: PackedMat::zeros(n, uplo),
        }
    }

    /// Creates from a packed matrix.
    #[inline]
    pub fn from_packed(packed: PackedMat<T>) -> Self {
        HermitianMat { packed }
    }

    /// Returns the dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.packed.dim()
    }

    /// Returns the shape.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        let n = self.dim();
        (n, n)
    }

    /// Returns the storage kind.
    #[inline]
    pub fn uplo(&self) -> TriangularKind {
        self.packed.kind()
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

    /// Returns element at (row, col) from the stored triangle.
    ///
    /// Note: For Hermitian matrices, `A[j,i] = conj(A[i,j])`. This method
    /// returns the stored value; the conjugate relationship must be
    /// handled by the caller for off-diagonal access to the non-stored triangle.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        match self.uplo() {
            TriangularKind::Upper => {
                if row <= col {
                    self.packed.get(row, col)
                } else {
                    self.packed.get(col, row)
                }
            }
            TriangularKind::Lower => {
                if row >= col {
                    self.packed.get(row, col)
                } else {
                    self.packed.get(col, row)
                }
            }
        }
    }

    /// Returns mutable element at (row, col).
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        match self.uplo() {
            TriangularKind::Upper => {
                if row <= col {
                    self.packed.get_mut(row, col)
                } else {
                    self.packed.get_mut(col, row)
                }
            }
            TriangularKind::Lower => {
                if row >= col {
                    self.packed.get_mut(row, col)
                } else {
                    self.packed.get_mut(col, row)
                }
            }
        }
    }

    /// Sets element at (row, col).
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        match self.uplo() {
            TriangularKind::Upper => {
                if row <= col {
                    self.packed.set(row, col, value);
                } else {
                    self.packed.set(col, row, value);
                }
            }
            TriangularKind::Lower => {
                if row >= col {
                    self.packed.set(row, col, value);
                } else {
                    self.packed.set(col, row, value);
                }
            }
        }
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

    /// Returns the underlying packed matrix.
    #[inline]
    pub fn as_packed(&self) -> &PackedMat<T> {
        &self.packed
    }

    /// Returns a mutable reference to the underlying packed matrix.
    #[inline]
    pub fn as_packed_mut(&mut self) -> &mut PackedMat<T> {
        &mut self.packed
    }

    /// Returns the diagonal elements.
    #[inline]
    pub fn diagonal(&self) -> Vec<T> {
        self.packed.diagonal()
    }

    /// Sets the diagonal elements.
    ///
    /// For Hermitian matrices, diagonal elements should be real.
    #[inline]
    pub fn set_diagonal(&mut self, diag: &[T]) {
        self.packed.set_diagonal(diag);
    }

    /// Fills all stored elements with a value.
    #[inline]
    pub fn fill(&mut self, value: T) {
        self.packed.fill(value);
    }

    /// Scales all elements by a real scalar.
    ///
    /// For Hermitian matrices, scaling by a real preserves Hermitianness.
    #[inline]
    pub fn scale(&mut self, alpha: T) {
        self.packed.scale(alpha);
    }
}

// Specialized implementation for Complex types with f32
impl HermitianMat<Complex<f32>> {
    /// Converts to a full dense Hermitian matrix.
    ///
    /// The non-stored triangle is filled with conjugate values.
    pub fn to_dense_f32(&self) -> Mat<Complex<f32>> {
        let n = self.dim();
        let mut mat = Mat::filled(n, n, Complex::new(0.0f32, 0.0f32));

        for j in 0..n {
            for i in 0..n {
                if let Some(&val) = self.get(i, j) {
                    if i == j {
                        // Diagonal: should be real
                        mat[(i, j)] = val;
                    } else {
                        // Off-diagonal
                        mat[(i, j)] = val;
                        mat[(j, i)] = val.conj();
                    }
                }
            }
        }

        mat
    }

    /// Creates from a dense matrix, verifying Hermitian property.
    ///
    /// Returns `None` if the matrix is not Hermitian within tolerance.
    pub fn from_dense_checked_f32(
        mat: &MatRef<'_, Complex<f32>>,
        uplo: TriangularKind,
        tol: f32,
    ) -> Option<Self> {
        assert!(mat.is_square(), "Matrix must be square");
        let n = mat.nrows();

        // Check Hermitian property: A[i,j] = conj(A[j,i])
        for j in 0..n {
            for i in 0..j {
                let a_ij = mat[(i, j)];
                let a_ji = mat[(j, i)];
                let diff = a_ij - a_ji.conj();
                if diff.norm() > tol {
                    return None;
                }
            }
            // Diagonal must be real
            let diag = mat[(j, j)];
            if diag.im.abs() > tol {
                return None;
            }
        }

        let packed = PackedMat::from_dense(mat, uplo);
        Some(HermitianMat { packed })
    }
}

// Specialized implementation for Complex types with f64
impl HermitianMat<Complex<f64>> {
    /// Converts to a full dense Hermitian matrix.
    ///
    /// The non-stored triangle is filled with conjugate values.
    pub fn to_dense(&self) -> Mat<Complex<f64>> {
        let n = self.dim();
        let mut mat = Mat::filled(n, n, Complex::new(0.0f64, 0.0f64));

        for j in 0..n {
            for i in 0..n {
                if let Some(&val) = self.get(i, j) {
                    if i == j {
                        // Diagonal: should be real
                        mat[(i, j)] = val;
                    } else {
                        // Off-diagonal
                        mat[(i, j)] = val;
                        mat[(j, i)] = val.conj();
                    }
                }
            }
        }

        mat
    }

    /// Creates from a dense matrix, verifying Hermitian property.
    ///
    /// Returns `None` if the matrix is not Hermitian within tolerance.
    pub fn from_dense_checked(
        mat: &MatRef<'_, Complex<f64>>,
        uplo: TriangularKind,
        tol: f64,
    ) -> Option<Self> {
        assert!(mat.is_square(), "Matrix must be square");
        let n = mat.nrows();

        // Check Hermitian property: A[i,j] = conj(A[j,i])
        for j in 0..n {
            for i in 0..j {
                let a_ij = mat[(i, j)];
                let a_ji = mat[(j, i)];
                let diff = a_ij - a_ji.conj();
                if diff.norm() > tol {
                    return None;
                }
            }
            // Diagonal must be real
            let diag = mat[(j, j)];
            if diag.im.abs() > tol {
                return None;
            }
        }

        let packed = PackedMat::from_dense(mat, uplo);
        Some(HermitianMat { packed })
    }
}

impl<T: Scalar + core::fmt::Debug> core::fmt::Debug for HermitianMat<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let n = self.dim();
        writeln!(f, "HermitianMat {}×{} ({:?}) {{", n, n, self.uplo())?;

        let max_dim = 6.min(n);
        for i in 0..max_dim {
            write!(f, "  [")?;
            for j in 0..max_dim {
                if j > 0 {
                    write!(f, ", ")?;
                }
                if let Some(v) = self.get(i, j) {
                    write!(f, "{:?}", v)?;
                } else {
                    write!(f, "*")?;
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

/// A symmetric reference view into packed data.
#[derive(Clone, Copy)]
pub struct SymmetricRef<'a, T: Scalar> {
    /// Packed reference.
    packed: PackedRef<'a, T>,
}

impl<'a, T: Scalar> SymmetricRef<'a, T> {
    /// Creates a new symmetric reference.
    #[inline]
    pub fn new(packed: PackedRef<'a, T>) -> Self {
        SymmetricRef { packed }
    }

    /// Creates from a slice.
    #[inline]
    pub fn from_slice(data: &'a [T], n: usize, uplo: TriangularKind) -> Self {
        SymmetricRef {
            packed: PackedRef::from_slice(data, n, uplo),
        }
    }

    /// Returns the dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.packed.dim()
    }

    /// Returns the storage kind.
    #[inline]
    pub fn uplo(&self) -> TriangularKind {
        self.packed.kind()
    }

    /// Returns element with symmetric access.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        match self.uplo() {
            TriangularKind::Upper => {
                if row <= col {
                    self.packed.get(row, col)
                } else {
                    self.packed.get(col, row)
                }
            }
            TriangularKind::Lower => {
                if row >= col {
                    self.packed.get(row, col)
                } else {
                    self.packed.get(col, row)
                }
            }
        }
    }

    /// Returns the underlying packed reference.
    #[inline]
    pub fn as_packed(&self) -> PackedRef<'a, T> {
        self.packed
    }
}

/// A mutable symmetric reference into packed data.
pub struct SymmetricMut<'a, T: Scalar> {
    /// Packed mutable reference.
    packed: PackedMut<'a, T>,
}

impl<'a, T: Scalar> SymmetricMut<'a, T> {
    /// Creates a new mutable symmetric reference.
    #[inline]
    pub fn new(packed: PackedMut<'a, T>) -> Self {
        SymmetricMut { packed }
    }

    /// Creates from a mutable slice.
    #[inline]
    pub fn from_slice(data: &'a mut [T], n: usize, uplo: TriangularKind) -> Self {
        SymmetricMut {
            packed: PackedMut::from_slice(data, n, uplo),
        }
    }

    /// Returns the dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.packed.dim()
    }

    /// Returns the storage kind.
    #[inline]
    pub fn uplo(&self) -> TriangularKind {
        self.packed.kind()
    }

    /// Returns element with symmetric access.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        match self.uplo() {
            TriangularKind::Upper => {
                if row <= col {
                    self.packed.get(row, col)
                } else {
                    self.packed.get(col, row)
                }
            }
            TriangularKind::Lower => {
                if row >= col {
                    self.packed.get(row, col)
                } else {
                    self.packed.get(col, row)
                }
            }
        }
    }

    /// Returns mutable element.
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        match self.uplo() {
            TriangularKind::Upper => {
                if row <= col {
                    self.packed.get_mut(row, col)
                } else {
                    self.packed.get_mut(col, row)
                }
            }
            TriangularKind::Lower => {
                if row >= col {
                    self.packed.get_mut(row, col)
                } else {
                    self.packed.get_mut(col, row)
                }
            }
        }
    }

    /// Sets element (symmetric: sets one storage location).
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        match self.uplo() {
            TriangularKind::Upper => {
                if row <= col {
                    self.packed.set(row, col, value);
                } else {
                    self.packed.set(col, row, value);
                }
            }
            TriangularKind::Lower => {
                if row >= col {
                    self.packed.set(row, col, value);
                } else {
                    self.packed.set(col, row, value);
                }
            }
        }
    }

    /// Creates an immutable reborrow.
    #[inline]
    pub fn rb(&self) -> SymmetricRef<'_, T> {
        SymmetricRef {
            packed: self.packed.rb(),
        }
    }

    /// Creates a mutable reborrow.
    #[inline]
    pub fn rb_mut(&mut self) -> SymmetricMut<'_, T> {
        SymmetricMut {
            packed: self.packed.rb_mut(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn test_symmetric_basic() {
        let mut s: SymmetricMat<f64> = SymmetricMat::zeros(3, TriangularKind::Upper);

        // Set elements
        s.set(0, 0, 1.0);
        s.set(0, 1, 2.0);
        s.set(0, 2, 3.0);
        s.set(1, 1, 4.0);
        s.set(1, 2, 5.0);
        s.set(2, 2, 6.0);

        // Symmetric access
        assert_eq!(s.get(0, 0), Some(&1.0));
        assert_eq!(s.get(0, 1), Some(&2.0));
        assert_eq!(s.get(1, 0), Some(&2.0)); // Symmetric!
        assert_eq!(s.get(2, 0), Some(&3.0)); // Symmetric!
        assert_eq!(s.get(1, 2), Some(&5.0));
        assert_eq!(s.get(2, 1), Some(&5.0)); // Symmetric!

        // Diagonal
        let diag = s.diagonal();
        assert_eq!(diag, vec![1.0, 4.0, 6.0]);
    }

    #[test]
    fn test_symmetric_lower() {
        let mut s: SymmetricMat<f64> = SymmetricMat::zeros(3, TriangularKind::Lower);

        s.set(0, 0, 1.0);
        s.set(1, 0, 2.0);
        s.set(1, 1, 3.0);
        s.set(2, 0, 4.0);
        s.set(2, 1, 5.0);
        s.set(2, 2, 6.0);

        // Symmetric access
        assert_eq!(s.get(0, 0), Some(&1.0));
        assert_eq!(s.get(1, 0), Some(&2.0));
        assert_eq!(s.get(0, 1), Some(&2.0)); // Symmetric!
        assert_eq!(s.get(0, 2), Some(&4.0)); // Symmetric!
    }

    #[test]
    fn test_symmetric_to_dense() {
        let mut s: SymmetricMat<f64> = SymmetricMat::zeros(3, TriangularKind::Upper);

        s.set(0, 0, 1.0);
        s.set(0, 1, 2.0);
        s.set(0, 2, 3.0);
        s.set(1, 1, 4.0);
        s.set(1, 2, 5.0);
        s.set(2, 2, 6.0);

        let dense = s.to_dense();

        // Upper triangle
        assert_eq!(dense[(0, 0)], 1.0);
        assert_eq!(dense[(0, 1)], 2.0);
        assert_eq!(dense[(0, 2)], 3.0);
        assert_eq!(dense[(1, 1)], 4.0);
        assert_eq!(dense[(1, 2)], 5.0);
        assert_eq!(dense[(2, 2)], 6.0);

        // Lower triangle (symmetric)
        assert_eq!(dense[(1, 0)], 2.0);
        assert_eq!(dense[(2, 0)], 3.0);
        assert_eq!(dense[(2, 1)], 5.0);
    }

    #[test]
    fn test_symmetric_from_dense() {
        let dense = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[2.0, 4.0, 5.0], &[3.0, 5.0, 6.0]]);

        let s = SymmetricMat::from_dense(&dense.as_ref(), TriangularKind::Upper);

        assert_eq!(s.get(0, 0), Some(&1.0));
        assert_eq!(s.get(0, 1), Some(&2.0));
        assert_eq!(s.get(1, 0), Some(&2.0));
        assert_eq!(s.get(2, 2), Some(&6.0));
    }

    #[test]
    fn test_symmetric_from_dense_checked() {
        // Symmetric matrix
        let symmetric = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[2.0, 4.0, 5.0], &[3.0, 5.0, 6.0]]);

        let result =
            SymmetricMat::from_dense_checked(&symmetric.as_ref(), TriangularKind::Upper, 1e-10);
        assert!(result.is_some());

        // Non-symmetric matrix
        let non_symmetric = Mat::from_rows(&[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0], // 4.0 != 2.0
            &[3.0, 5.0, 6.0],
        ]);

        let result =
            SymmetricMat::from_dense_checked(&non_symmetric.as_ref(), TriangularKind::Upper, 1e-10);
        assert!(result.is_none());
    }

    #[test]
    fn test_symmetric_operations() {
        let mut s: SymmetricMat<f64> = SymmetricMat::filled(3, TriangularKind::Upper, 1.0);

        // Scale
        s.scale(2.0);
        assert_eq!(s.get(0, 0), Some(&2.0));
        assert_eq!(s.get(0, 1), Some(&2.0));

        // Add to diagonal
        s.add_diagonal(1.0);
        assert_eq!(s.get(0, 0), Some(&3.0));
        assert_eq!(s.get(1, 1), Some(&3.0));
        assert_eq!(s.get(0, 1), Some(&2.0)); // Off-diagonal unchanged
    }

    #[test]
    fn test_symmetric_view() {
        let dense = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[2.0, 4.0, 5.0], &[3.0, 5.0, 6.0]]);

        let view = SymmetricView::new(dense.as_ref(), TriangularKind::Upper);

        assert_eq!(view.get(0, 1), Some(&2.0));
        assert_eq!(view.get(1, 0), Some(&2.0)); // Uses upper triangle
    }

    #[test]
    fn test_symmetric_view_mut() {
        let mut dense: Mat<f64> = Mat::zeros(3, 3);
        {
            let mut view = SymmetricViewMut::new(dense.as_mut(), TriangularKind::Upper);

            view.set(0, 1, 5.0);
        }

        // Both positions should be set
        assert_eq!(dense[(0, 1)], 5.0);
        assert_eq!(dense[(1, 0)], 5.0);
    }

    #[test]
    fn test_hermitian_basic() {
        let mut h: HermitianMat<Complex64> = HermitianMat::zeros(2, TriangularKind::Upper);

        // Diagonal (real)
        h.set(0, 0, Complex64::new(1.0, 0.0));
        h.set(1, 1, Complex64::new(2.0, 0.0));

        // Off-diagonal
        h.set(0, 1, Complex64::new(3.0, 4.0));

        assert_eq!(h.get(0, 0), Some(&Complex64::new(1.0, 0.0)));
        assert_eq!(h.get(1, 1), Some(&Complex64::new(2.0, 0.0)));
        assert_eq!(h.get(0, 1), Some(&Complex64::new(3.0, 4.0)));
        assert_eq!(h.get(1, 0), Some(&Complex64::new(3.0, 4.0))); // Stored value
    }

    #[test]
    fn test_hermitian_to_dense() {
        let mut h: HermitianMat<Complex64> = HermitianMat::zeros(2, TriangularKind::Upper);

        h.set(0, 0, Complex64::new(1.0, 0.0));
        h.set(1, 1, Complex64::new(2.0, 0.0));
        h.set(0, 1, Complex64::new(3.0, 4.0));

        let dense = h.to_dense();

        assert_eq!(dense[(0, 0)], Complex64::new(1.0, 0.0));
        assert_eq!(dense[(1, 1)], Complex64::new(2.0, 0.0));
        assert_eq!(dense[(0, 1)], Complex64::new(3.0, 4.0));
        assert_eq!(dense[(1, 0)], Complex64::new(3.0, -4.0)); // Conjugate!
    }

    #[test]
    fn test_hermitian_from_dense_checked() {
        // Valid Hermitian matrix
        let hermitian = Mat::from_rows(&[
            &[Complex64::new(1.0, 0.0), Complex64::new(3.0, 4.0)],
            &[Complex64::new(3.0, -4.0), Complex64::new(2.0, 0.0)],
        ]);

        let result =
            HermitianMat::from_dense_checked(&hermitian.as_ref(), TriangularKind::Upper, 1e-10);
        assert!(result.is_some());

        // Non-Hermitian (imaginary diagonal)
        let non_hermitian = Mat::from_rows(&[
            &[Complex64::new(1.0, 0.5), Complex64::new(3.0, 4.0)],
            &[Complex64::new(3.0, -4.0), Complex64::new(2.0, 0.0)],
        ]);

        let result =
            HermitianMat::from_dense_checked(&non_hermitian.as_ref(), TriangularKind::Upper, 1e-10);
        assert!(result.is_none());
    }

    #[test]
    fn test_symmetric_ref_mut() {
        let mut data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut smut = SymmetricMut::from_slice(&mut data, 3, TriangularKind::Upper);

        assert_eq!(smut.get(0, 0), Some(&1.0));
        assert_eq!(smut.get(0, 1), Some(&2.0));
        assert_eq!(smut.get(1, 0), Some(&2.0)); // Symmetric access

        smut.set(0, 1, 10.0);
        assert_eq!(smut.get(0, 1), Some(&10.0));
        assert_eq!(smut.get(1, 0), Some(&10.0)); // Both directions
    }
}
