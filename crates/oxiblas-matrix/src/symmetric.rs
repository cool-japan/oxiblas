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

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::packed::{PackedMat, PackedMut, PackedRef, TriangularKind};
use crate::{Mat, MatMut, MatRef};
use num_complex::Complex;
use num_traits::{One, Zero};
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
    ///
    /// # Panics
    /// Panics if `row` or `col` is out of bounds.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        // `get_mut` folds `(row, col)` onto the stored triangle (swapping to
        // `(col, row)` for the other half), so it is `None` exactly when an
        // index is out of bounds.
        let Some(slot) = self.get_mut(row, col) else {
            panic!("Index out of bounds");
        };
        *slot = value;
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

impl<T: Scalar> SymmetricMat<T> {
    /// Returns the Frobenius norm squared: the sum of `|a_ij|^2` over all `i, j`.
    ///
    /// Off-diagonal entries are physically stored once but occur twice in the
    /// full matrix (`a_ij` and `a_ji`), so their squared magnitude is counted
    /// twice. Using `abs_sq` (`|z|^2`) instead of `val * val` makes the result
    /// correct for complex scalars — where `val * val` is generally complex and
    /// not the squared modulus — as well as for real scalars, for which the two
    /// coincide.
    pub fn frobenius_norm_squared(&self) -> T::Real {
        let n = self.dim();
        // `T::Real: Scalar` supplies `One`, so `1 + 1` yields the real `2`
        // without needing a fallible primitive conversion.
        let two = T::Real::one() + T::Real::one();
        let mut sum = T::Real::zero();

        for j in 0..n {
            for i in 0..n {
                // Visit each entry exactly once by reading only from the
                // authoritative stored triangle: `get()` returns the raw stored
                // value for BOTH orientations of an off-diagonal pair, so
                // iterating the whole grid would double-visit every pair.
                let in_stored_triangle = match self.uplo() {
                    TriangularKind::Upper => i <= j,
                    TriangularKind::Lower => i >= j,
                };
                if !in_stored_triangle {
                    continue;
                }
                if let Some(&val) = self.get(i, j) {
                    if i == j {
                        sum += val.abs_sq();
                    } else {
                        sum += two * val.abs_sq();
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
/// Only one triangle (`uplo`) is stored. [`set`](Self::set) and
/// [`element`](Self::element) work with the *logical* matrix in either
/// triangle; [`get`](Self::get) and [`get_mut`](Self::get_mut) give raw
/// access to the single stored slot of a pair, which holds `A[i,j]` for the
/// stored orientation and `conj(A[i,j])` for the other one.
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
/// // Diagonal is real (only the real part of a diagonal write is kept)
/// h.set(0, 0, Complex64::new(1.0, 0.0));
/// h.set(1, 1, Complex64::new(2.0, 0.0));
///
/// // Off-diagonal: (0,1) and (1,0) are conjugates
/// h.set(0, 1, Complex64::new(3.0, 4.0));
///
/// assert_eq!(h.element(0, 1), Some(Complex64::new(3.0, 4.0)));
/// assert_eq!(h.element(1, 0), Some(Complex64::new(3.0, -4.0)));
///
/// // Writing through the non-stored triangle is equivalent.
/// h.set(1, 0, Complex64::new(5.0, -6.0));
/// assert_eq!(h.element(0, 1), Some(Complex64::new(5.0, 6.0)));
///
/// // `get` returns the raw stored slot (here the upper triangle) for both
/// // orientations of a pair.
/// assert_eq!(h.get(1, 0), Some(&Complex64::new(5.0, 6.0)));
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

    /// Returns `true` when `(row, col)` lies in the stored triangle.
    #[inline]
    fn is_stored(&self, row: usize, col: usize) -> bool {
        match self.uplo() {
            TriangularKind::Upper => row <= col,
            TriangularKind::Lower => row >= col,
        }
    }

    /// Returns element at (row, col) from the stored triangle.
    ///
    /// Note: For Hermitian matrices, `A[j,i] = conj(A[i,j])`. This method
    /// returns the stored value; the conjugate relationship must be
    /// handled by the caller for off-diagonal access to the non-stored triangle.
    /// Use [`element`](Self::element) to read the logical `A[row, col]` in
    /// either triangle.
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

    /// Returns a mutable reference to the stored slot of (row, col).
    ///
    /// Like [`get`](Self::get) this is raw storage access: for (row, col) in
    /// the non-stored triangle the slot is that of (col, row) and holds
    /// `conj(A[row, col])`, so writing `z` through it makes
    /// `A[row, col] = conj(z)`. Writing a non-real value to a diagonal slot
    /// breaks the Hermitian invariant. Use [`set`](Self::set) to assign a
    /// logical element.
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

    /// Returns the logical element `A[row, col]` by value, in either triangle:
    /// the stored value for the stored triangle and its conjugate for the
    /// other one. Returns `None` if an index is out of bounds.
    #[inline]
    pub fn element(&self, row: usize, col: usize) -> Option<T> {
        let stored = *self.get(row, col)?;
        Some(if self.is_stored(row, col) {
            stored
        } else {
            stored.conj()
        })
    }

    /// Sets element at (row, col).
    ///
    /// After the call the logical matrix satisfies `A[row, col] = value` and,
    /// by the Hermitian symmetry, `A[col, row] = conj(value)`, whichever
    /// triangle is stored (a write to the non-stored triangle stores
    /// `conj(value)` in the mirrored slot).
    ///
    /// The diagonal of a Hermitian matrix is real (`A[i, i] = conj(A[i, i])`),
    /// so for `row == col` only the real part of `value` is stored and its
    /// imaginary part is discarded.
    ///
    /// # Panics
    /// Panics if `row` or `col` is out of bounds.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        let stored = if row == col {
            T::from_real(value.real())
        } else if self.is_stored(row, col) {
            value
        } else {
            value.conj()
        };
        // `get_mut` folds `(row, col)` onto the stored triangle (swapping to
        // `(col, row)` for the other half), so it is `None` exactly when an
        // index is out of bounds.
        let Some(slot) = self.get_mut(row, col) else {
            panic!("Index out of bounds");
        };
        *slot = stored;
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
    /// The diagonal of a Hermitian matrix is real, so only the real part of
    /// each value is stored (the same rule as [`set`](Self::set)).
    #[inline]
    pub fn set_diagonal(&mut self, diag: &[T]) {
        let real_diag: Vec<T> = diag.iter().map(|d| T::from_real(d.real())).collect();
        self.packed.set_diagonal(&real_diag);
    }

    /// Fills all stored elements with a value.
    ///
    /// This writes `value` verbatim to every stored slot, the diagonal
    /// included; with a non-real `value` the diagonal is therefore not real.
    #[inline]
    pub fn fill(&mut self, value: T) {
        self.packed.fill(value);
    }

    /// Scales all elements by a real scalar.
    ///
    /// The factor is the real component type `T::Real`, not a full complex
    /// scalar, because scaling a Hermitian matrix by a non-real number would
    /// violate the invariant `A = A^H` (each off-diagonal pair `a_ij`,
    /// `conj(a_ij)` would pick up mismatched phases and the diagonal would
    /// gain an imaginary part). Enforcing realness in the type makes a complex
    /// factor a compile error rather than a silent invariant break. For real
    /// `T`, `T::Real` is simply `T`.
    #[inline]
    pub fn scale(&mut self, alpha: T::Real) {
        self.packed.scale(T::from_real(alpha));
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

        // Read only from the authoritative stored triangle. `get()` returns the
        // *raw* stored value for BOTH orientations of an off-diagonal pair (it
        // never conjugates), so writing on every visited (i, j) would let the
        // non-stored orientation overwrite the correct entry with its conjugate
        // — producing conj(A) whenever the smaller `j` carries the wrong write,
        // i.e. for Lower storage. Writing each pair exactly once from its single
        // stored value keeps both triangles correct regardless of `uplo`.
        for j in 0..n {
            for i in 0..n {
                let in_stored_triangle = match self.uplo() {
                    TriangularKind::Upper => i <= j,
                    TriangularKind::Lower => i >= j,
                };
                if !in_stored_triangle {
                    continue;
                }
                if let Some(&val) = self.get(i, j) {
                    if i == j {
                        // Diagonal of a Hermitian matrix is real; store as-is.
                        mat[(i, j)] = val;
                    } else {
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

        // Read only from the authoritative stored triangle. `get()` returns the
        // *raw* stored value for BOTH orientations of an off-diagonal pair (it
        // never conjugates), so writing on every visited (i, j) would let the
        // non-stored orientation overwrite the correct entry with its conjugate
        // — producing conj(A) whenever the smaller `j` carries the wrong write,
        // i.e. for Lower storage. Writing each pair exactly once from its single
        // stored value keeps both triangles correct regardless of `uplo`.
        for j in 0..n {
            for i in 0..n {
                let in_stored_triangle = match self.uplo() {
                    TriangularKind::Upper => i <= j,
                    TriangularKind::Lower => i >= j,
                };
                if !in_stored_triangle {
                    continue;
                }
                if let Some(&val) = self.get(i, j) {
                    if i == j {
                        // Diagonal of a Hermitian matrix is real; store as-is.
                        mat[(i, j)] = val;
                    } else {
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
                // The logical element (conjugated in the non-stored triangle).
                if let Some(v) = self.element(i, j) {
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
    ///
    /// # Panics
    /// Panics if `row` or `col` is out of bounds.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, value: T) {
        // `get_mut` folds `(row, col)` onto the stored triangle (swapping to
        // `(col, row)` for the other half), so it is `None` exactly when an
        // index is out of bounds.
        let Some(slot) = self.get_mut(row, col) else {
            panic!("Index out of bounds");
        };
        *slot = value;
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

    /// Regression for the Lower-storage `to_dense` conjugate-transpose bug.
    ///
    /// Previously `to_dense` visited every (i, j) and wrote `mat[(i,j)] = val;
    /// mat[(j,i)] = conj(val)` from the *raw* stored value returned by `get()`
    /// for both orientations. Because the outer loop is over `j` ascending, for
    /// Lower storage the correct write (from the stored triangle, at the smaller
    /// `j`) was overwritten by the later non-stored-orientation write, yielding
    /// `conj(A)` instead of `A`. This builds a Hermitian matrix from a Lower
    /// triangle and checks the round trip reproduces the ORIGINAL dense matrix
    /// element-for-element (not its conjugate mirror).
    #[test]
    fn test_hermitian_to_dense_lower_roundtrip() {
        // A genuine 3x3 Hermitian matrix: A[i,j] == conj(A[j,i]), real diagonal.
        let original = Mat::from_rows(&[
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(3.0, 4.0),
                Complex64::new(5.0, -2.0),
            ],
            &[
                Complex64::new(3.0, -4.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(6.0, 1.0),
            ],
            &[
                Complex64::new(5.0, 2.0),
                Complex64::new(6.0, -1.0),
                Complex64::new(7.0, 0.0),
            ],
        ]);

        let herm =
            HermitianMat::from_dense_checked(&original.as_ref(), TriangularKind::Lower, 1e-12)
                .expect("matrix is Hermitian by construction");
        let dense = herm.to_dense();

        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(
                    dense[(i, j)],
                    original[(i, j)],
                    "mismatch at ({i}, {j}): Lower-storage to_dense must reproduce A, not conj(A)"
                );
            }
        }

        // Explicitly pin the pair the finding flagged: A[1,0] = 3-4i must stay
        // 3-4i (not become 3+4i), and its partner A[0,1] must be its conjugate.
        assert_eq!(dense[(1, 0)], Complex64::new(3.0, -4.0));
        assert_eq!(dense[(0, 1)], Complex64::new(3.0, 4.0));
    }

    /// Same Lower-storage round-trip regression for the f32 `to_dense_f32` path.
    #[test]
    fn test_hermitian_to_dense_f32_lower_roundtrip() {
        let original = Mat::from_rows(&[
            &[Complex::new(1.0f32, 0.0), Complex::new(3.0, 4.0)],
            &[Complex::new(3.0, -4.0), Complex::new(2.0, 0.0)],
        ]);

        let herm =
            HermitianMat::from_dense_checked_f32(&original.as_ref(), TriangularKind::Lower, 1e-6)
                .expect("matrix is Hermitian by construction");
        let dense = herm.to_dense_f32();

        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(dense[(i, j)], original[(i, j)], "mismatch at ({i}, {j})");
            }
        }
        assert_eq!(dense[(1, 0)], Complex::new(3.0f32, -4.0));
        assert_eq!(dense[(0, 1)], Complex::new(3.0f32, 4.0));
    }

    /// The Upper-storage path (the only one the original test covered) must keep
    /// working after the fix.
    #[test]
    fn test_hermitian_to_dense_upper_roundtrip() {
        let original = Mat::from_rows(&[
            &[Complex64::new(1.0, 0.0), Complex64::new(3.0, 4.0)],
            &[Complex64::new(3.0, -4.0), Complex64::new(2.0, 0.0)],
        ]);

        let herm =
            HermitianMat::from_dense_checked(&original.as_ref(), TriangularKind::Upper, 1e-12)
                .expect("matrix is Hermitian by construction");
        let dense = herm.to_dense();

        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(dense[(i, j)], original[(i, j)], "mismatch at ({i}, {j})");
            }
        }
    }

    /// `frobenius_norm_squared` is now generic over `Scalar`; verify it works for
    /// complex scalars using `|z|^2` (not `z * z`) and still matches the real
    /// case. For a symmetric matrix (A = A^T) the norm is the sum of `|a_ij|^2`
    /// over the full matrix, with off-diagonals counted twice.
    #[test]
    fn test_symmetric_frobenius_norm_squared_generic() {
        // Real case: full matrix is
        //   [1 2 3; 2 4 5; 3 5 6]
        // sum of squares = 1+4+9 + 4+16+25 + 9+25+36 = 129.
        let mut s: SymmetricMat<f64> = SymmetricMat::zeros(3, TriangularKind::Upper);
        s.set(0, 0, 1.0);
        s.set(0, 1, 2.0);
        s.set(0, 2, 3.0);
        s.set(1, 1, 4.0);
        s.set(1, 2, 5.0);
        s.set(2, 2, 6.0);
        assert_eq!(s.frobenius_norm_squared(), 129.0);

        // Complex case: symmetric (A = A^T), 2x2
        //   [1+1i  2+0i]
        //   [2+0i  3-1i]
        // sum |a_ij|^2 = |1+1i|^2 + 2*|2|^2 + |3-1i|^2 = 2 + 8 + 10 = 20.
        let mut c: SymmetricMat<Complex64> = SymmetricMat::zeros(2, TriangularKind::Upper);
        c.set(0, 0, Complex64::new(1.0, 1.0));
        c.set(0, 1, Complex64::new(2.0, 0.0));
        c.set(1, 1, Complex64::new(3.0, -1.0));
        let norm_sq: f64 = c.frobenius_norm_squared();
        assert!((norm_sq - 20.0).abs() < 1e-12, "got {norm_sq}");

        // Lower storage must give the same result as Upper storage.
        let mut c_lower: SymmetricMat<Complex64> = SymmetricMat::zeros(2, TriangularKind::Lower);
        c_lower.set(0, 0, Complex64::new(1.0, 1.0));
        c_lower.set(1, 0, Complex64::new(2.0, 0.0));
        c_lower.set(1, 1, Complex64::new(3.0, -1.0));
        let norm_sq_lower: f64 = c_lower.frobenius_norm_squared();
        assert!((norm_sq_lower - 20.0).abs() < 1e-12, "got {norm_sq_lower}");
    }

    /// `HermitianMat::scale` now takes a real factor (`T::Real`); scaling by a
    /// real value must preserve the Hermitian invariant and simply scale every
    /// entry. (A complex factor is now rejected at compile time.)
    #[test]
    fn test_hermitian_scale_real_preserves_hermitian() {
        let mut h: HermitianMat<Complex64> = HermitianMat::zeros(2, TriangularKind::Upper);
        h.set(0, 0, Complex64::new(1.0, 0.0));
        h.set(1, 1, Complex64::new(2.0, 0.0));
        h.set(0, 1, Complex64::new(3.0, 4.0));

        // `alpha` is f64 (== Complex64::Real), enforced by the signature.
        h.scale(2.0);

        let dense = h.to_dense();
        assert_eq!(dense[(0, 0)], Complex64::new(2.0, 0.0));
        assert_eq!(dense[(1, 1)], Complex64::new(4.0, 0.0));
        assert_eq!(dense[(0, 1)], Complex64::new(6.0, 8.0));
        // Hermitian invariant: A[1,0] == conj(A[0,1]).
        assert_eq!(dense[(1, 0)], dense[(0, 1)].conj());
        assert_eq!(dense[(1, 0)], Complex64::new(6.0, -8.0));
    }

    // --- Regression: `set` without `expect` -------------------------------
    //
    // `set` used to fold `(row, col)` onto the stored triangle by hand and
    // `.expect` the packed write. It now writes through `get_mut`, which does
    // the same folding; these tests pin that every in-bounds index (stored or
    // mirrored half, both storage kinds) still lands in the single packed slot
    // it always did, and that out-of-bounds indices still panic.

    /// Packed slot of `(row, col)` after folding onto the stored triangle.
    fn folded_slot(n: usize, uplo: TriangularKind, row: usize, col: usize) -> usize {
        let (i, j) = match uplo {
            TriangularKind::Upper => (row.min(col), row.max(col)),
            TriangularKind::Lower => (row.max(col), row.min(col)),
        };
        let probe: PackedMat<f64> = PackedMat::zeros(n, uplo);
        probe
            .packed_index(i, j)
            .unwrap_or_else(|| panic!("({i}, {j}) must be stored"))
    }

    #[test]
    fn test_symmetric_set_folds_every_index_onto_stored_slot() {
        let n = 4;
        for uplo in [TriangularKind::Upper, TriangularKind::Lower] {
            for row in 0..n {
                for col in 0..n {
                    let value = (10 * row + col) as f64 + 1.0;
                    let expected = folded_slot(n, uplo, row, col);

                    let mut s: SymmetricMat<f64> = SymmetricMat::zeros(n, uplo);
                    s.set(row, col, value);
                    assert_eq!(s.as_slice()[expected], value, "{uplo:?} ({row}, {col})");
                    assert_eq!(s.get(row, col), Some(&value));
                    assert_eq!(s.get(col, row), Some(&value));

                    let mut data = vec![0.0f64; PackedMat::<f64>::packed_len(n)];
                    let mut sm = SymmetricMut::from_slice(&mut data, n, uplo);
                    sm.set(row, col, value);
                    assert_eq!(sm.get(col, row), Some(&value));
                    assert_eq!(
                        data[expected], value,
                        "SymmetricMut {uplo:?} ({row}, {col})"
                    );

                    let mut h: HermitianMat<Complex64> = HermitianMat::zeros(n, uplo);
                    let z = Complex64::new(value, -value);
                    h.set(row, col, z);
                    // One stored slot serves both (row, col) and (col, row). It
                    // holds A[row, col] = z when (row, col) is the stored
                    // orientation, conj(z) = A[col, row] when it is the mirrored
                    // one, and only the real part on the diagonal.
                    let stored_orientation = match uplo {
                        TriangularKind::Upper => row <= col,
                        TriangularKind::Lower => row >= col,
                    };
                    let slot_value = if row == col {
                        Complex64::new(value, 0.0)
                    } else if stored_orientation {
                        z
                    } else {
                        z.conj()
                    };
                    assert_eq!(
                        h.as_slice()[expected],
                        slot_value,
                        "Hermitian {uplo:?} ({row}, {col})"
                    );
                    assert_eq!(h.get(col, row), Some(&slot_value));
                    assert_eq!(h.get(row, col), Some(&slot_value));
                }
            }
        }
    }

    /// `set(row, col, z)` must make the *logical* matrix satisfy
    /// `A[row, col] = z` and `A[col, row] = conj(z)` for every index, whichever
    /// triangle is stored, and a diagonal write keeps only the real part.
    /// Previously a write to the non-stored triangle stored `z` unconjugated in
    /// the mirrored slot, so the matrix read back `conj(z)` at `(row, col)`.
    #[test]
    fn test_hermitian_set_writes_logical_element_in_both_triangles() {
        let n = 4;
        for uplo in [TriangularKind::Upper, TriangularKind::Lower] {
            for row in 0..n {
                for col in 0..n {
                    let z = Complex64::new(1.0 + row as f64, 0.5 + 2.0 * col as f64);
                    let expected = if row == col {
                        Complex64::new(z.re, 0.0)
                    } else {
                        z
                    };
                    let mut h: HermitianMat<Complex64> = HermitianMat::zeros(n, uplo);
                    h.set(row, col, z);

                    // Logical access in both triangles.
                    assert_eq!(
                        h.element(row, col),
                        Some(expected),
                        "{uplo:?} element({row},{col})"
                    );
                    assert_eq!(
                        h.element(col, row),
                        Some(expected.conj()),
                        "{uplo:?} element({col},{row}) after set({row},{col})"
                    );

                    let dense = h.to_dense();
                    assert_eq!(dense[(row, col)], expected, "{uplo:?} A[{row},{col}]");
                    assert_eq!(
                        dense[(col, row)],
                        expected.conj(),
                        "{uplo:?} A[{col},{row}] after set({row},{col})"
                    );
                    // Every other entry stays zero.
                    for i in 0..n {
                        for j in 0..n {
                            if (i, j) != (row, col) && (i, j) != (col, row) {
                                assert_eq!(dense[(i, j)], Complex64::new(0.0, 0.0));
                            }
                        }
                    }
                    // `get` reads the raw stored slot: the logical value for the
                    // stored orientation of the pair.
                    let stored_is_row_col = match uplo {
                        TriangularKind::Upper => row <= col,
                        TriangularKind::Lower => row >= col,
                    };
                    let (si, sj, logical) = if stored_is_row_col {
                        (row, col, expected)
                    } else {
                        (col, row, expected.conj())
                    };
                    assert_eq!(
                        h.get(si, sj),
                        Some(&logical),
                        "{uplo:?} raw slot ({si},{sj})"
                    );
                }
            }
        }
    }

    /// Filling a whole Hermitian matrix through `set` — in an order that writes
    /// both triangles — reproduces the dense matrix for both storage kinds and
    /// both precisions.
    #[test]
    fn test_hermitian_set_roundtrip_through_both_triangles() {
        let original = Mat::from_rows(&[
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(3.0, 4.0),
                Complex64::new(5.0, -2.0),
            ],
            &[
                Complex64::new(3.0, -4.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(6.0, 1.0),
            ],
            &[
                Complex64::new(5.0, 2.0),
                Complex64::new(6.0, -1.0),
                Complex64::new(7.0, 0.0),
            ],
        ]);
        for uplo in [TriangularKind::Upper, TriangularKind::Lower] {
            // Upper-triangle entries via (i, j), lower ones via (j, i): half of
            // the writes land in the non-stored triangle for either `uplo`.
            let mut h: HermitianMat<Complex64> = HermitianMat::zeros(3, uplo);
            for (i, j) in [(0, 0), (1, 0), (0, 2), (1, 1), (2, 1), (2, 2)] {
                h.set(i, j, original[(i, j)]);
            }
            let dense = h.to_dense();
            for i in 0..3 {
                for j in 0..3 {
                    assert_eq!(dense[(i, j)], original[(i, j)], "{uplo:?} ({i}, {j})");
                    assert_eq!(h.element(i, j), Some(original[(i, j)]), "{uplo:?} element");
                }
            }
            // The Debug rendering shows the logical matrix, so the mirrored
            // triangle prints the conjugates.
            let shown = format!("{h:?}");
            assert!(
                shown.contains(&format!("{:?}", original[(1, 0)]))
                    && shown.contains(&format!("{:?}", original[(0, 1)])),
                "{uplo:?} Debug must show both triangles: {shown}"
            );

            let mut h32: HermitianMat<Complex<f32>> = HermitianMat::zeros(3, uplo);
            for (i, j) in [(0, 0), (1, 0), (0, 2), (1, 1), (2, 1), (2, 2)] {
                let z = original[(i, j)];
                h32.set(i, j, Complex::new(z.re as f32, z.im as f32));
            }
            let dense32 = h32.to_dense_f32();
            for i in 0..3 {
                for j in 0..3 {
                    let z = original[(i, j)];
                    assert_eq!(
                        dense32[(i, j)],
                        Complex::new(z.re as f32, z.im as f32),
                        "f32 {uplo:?} ({i}, {j})"
                    );
                }
            }
        }
    }

    /// Diagonal rule: a Hermitian diagonal is real, so `set(i, i, z)` and
    /// `set_diagonal` keep only the real part of each value.
    #[test]
    fn test_hermitian_diagonal_writes_keep_real_part() {
        for uplo in [TriangularKind::Upper, TriangularKind::Lower] {
            let mut h: HermitianMat<Complex64> = HermitianMat::zeros(3, uplo);
            h.set(1, 1, Complex64::new(2.5, -7.0));
            assert_eq!(h.element(1, 1), Some(Complex64::new(2.5, 0.0)));
            assert_eq!(h.get(1, 1), Some(&Complex64::new(2.5, 0.0)));
            assert_eq!(h.to_dense()[(1, 1)], Complex64::new(2.5, 0.0));

            h.set_diagonal(&[
                Complex64::new(1.0, 1.0),
                Complex64::new(-2.0, 0.5),
                Complex64::new(3.0, 0.0),
            ]);
            assert_eq!(
                h.diagonal(),
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(-2.0, 0.0),
                    Complex64::new(3.0, 0.0),
                ]
            );
        }
    }

    /// `get_mut` stays a raw stored-slot accessor: writing `z` through the
    /// mirrored orientation stores `z` in the shared slot, i.e. sets
    /// `A[row, col] = conj(z)`, as documented.
    #[test]
    fn test_hermitian_get_mut_is_raw_slot_access() {
        let mut h: HermitianMat<Complex64> = HermitianMat::zeros(2, TriangularKind::Upper);
        let z = Complex64::new(1.0, 2.0);
        if let Some(slot) = h.get_mut(1, 0) {
            *slot = z;
        }
        assert_eq!(h.get(0, 1), Some(&z));
        assert_eq!(h.element(0, 1), Some(z));
        assert_eq!(h.element(1, 0), Some(z.conj()));
    }

    /// For a real element type conjugation is the identity: `set` on either
    /// triangle behaves symmetrically, and `element` agrees with `get`.
    #[test]
    fn test_hermitian_real_element_type() {
        for uplo in [TriangularKind::Upper, TriangularKind::Lower] {
            let mut h: HermitianMat<f64> = HermitianMat::zeros(3, uplo);
            h.set(2, 0, 4.0);
            h.set(1, 1, -1.5);
            assert_eq!(h.element(2, 0), Some(4.0));
            assert_eq!(h.element(0, 2), Some(4.0));
            assert_eq!(h.element(1, 1), Some(-1.5));
            assert_eq!(h.get(0, 2), Some(&4.0));
        }
    }

    #[test]
    fn test_hermitian_element_out_of_bounds_is_none() {
        let h: HermitianMat<Complex64> = HermitianMat::zeros(3, TriangularKind::Lower);
        assert_eq!(h.element(3, 0), None);
        assert_eq!(h.element(0, 3), None);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_symmetric_set_out_of_bounds_panics() {
        let mut s: SymmetricMat<f64> = SymmetricMat::zeros(3, TriangularKind::Lower);
        s.set(1, 3, 1.0);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_hermitian_set_out_of_bounds_panics() {
        let mut h: HermitianMat<Complex64> = HermitianMat::zeros(3, TriangularKind::Upper);
        h.set(3, 0, Complex64::new(1.0, 0.0));
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_symmetric_mut_set_out_of_bounds_panics() {
        let mut data = vec![0.0f64; PackedMat::<f64>::packed_len(3)];
        let mut sm = SymmetricMut::from_slice(&mut data, 3, TriangularKind::Upper);
        sm.set(0, 7, 1.0);
    }
}
