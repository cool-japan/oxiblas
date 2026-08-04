//! nalgebra type conversions for OxiBLAS matrices.
//!
//! This module provides conversion between OxiBLAS matrix types ([`Mat`],
//! [`MatRef`], [`crate::MatMut`]) and nalgebra types ([`DMatrix`],
//! [`DMatrixView`], [`DMatrixViewMut`]).
//!
//! # Which conversions are zero-copy
//!
//! Both `Mat`/`MatRef`/`MatMut` and `DMatrix` use column-major layout with a
//! per-matrix "leading dimension" (the stride between the start of one
//! column and the next): OxiBLAS calls it `row_stride`, nalgebra's strided
//! views call it `CStride`. Because the layouts genuinely coincide, borrowed
//! **view** conversions are true zero-copy borrows of the same buffer:
//!
//! - [`mat_ref_to_dmatrix_view`] / [`MatNalgebraExt::to_dmatrix_view`] --
//!   `MatRef`/`Mat` -> `DMatrixView` (read-only, zero-copy).
//! - [`mat_mut_to_dmatrix_view_mut`] -- `MatMut` -> `DMatrixViewMut`
//!   (read-write, zero-copy).
//! - [`dmatrix_to_mat_ref`] / [`DMatrixOxiblasExt::to_mat_ref`] -- `&DMatrix`
//!   -> `MatRef` (read-only, zero-copy; `DMatrix`'s `VecStorage` is always
//!   fully packed, i.e. its leading dimension equals `nrows` exactly, so no
//!   layout translation is needed).
//! - [`dmatrix_to_mat_mut`] -- `&mut DMatrix` -> `MatMut` (read-write,
//!   zero-copy).
//!
//! **Owned-to-owned conversions still copy.** `dmatrix_to_mat`,
//! `mat_to_dmatrix`, `dmatrix_view_to_mat`, and the vector conversions all
//! produce a *new*, independently owned matrix, so they must copy the data
//! into that new allocation -- there is no way to hand over ownership of one
//! type's buffer as if it were the other's, because `Mat`'s buffer is a
//! `oxiblas_core` `AlignedVec` (SIMD-aligned, with cache-line-padded
//! `row_stride`) while `DMatrix`'s buffer is a plain unpadded `Vec`; freeing
//! one with the other's allocator/layout would be unsound. Use the view
//! functions above instead whenever a borrow suffices.
//!
//! # Example
//!
//! ```
//! # #[cfg(feature = "nalgebra")] {
//! use oxiblas_matrix::prelude::*;
//! use oxiblas_matrix::nalgebra_compat::*;
//! use nalgebra::DMatrix;
//!
//! // Convert from nalgebra to oxiblas (copies: different owned buffers)
//! let na_mat = DMatrix::from_fn(3, 3, |i, j| (i + j) as f64);
//! let oxi_mat: Mat<f64> = dmatrix_to_mat(&na_mat);
//! assert_eq!(oxi_mat[(1, 2)], 3.0);
//!
//! // Convert from oxiblas to nalgebra (copies: different owned buffers)
//! let mat: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
//! let na_mat: DMatrix<f64> = mat_to_dmatrix(&mat);
//!
//! // Borrow instead of copying: genuine zero-copy view
//! let view = mat_ref_to_dmatrix_view(mat.as_ref());
//! assert_eq!(view[(0, 0)], 1.0);
//! # }
//! ```

use crate::{Mat, MatMut, MatRef};
use nalgebra::{
    DMatrix, DMatrixView, DMatrixViewMut, Dyn, Matrix, U1, ViewStorage, ViewStorageMut,
};
use num_traits::Zero;
use oxiblas_core::scalar::Scalar;

/// Converts a nalgebra `DMatrix` to an owned `Mat`.
///
/// This always creates a copy since `DMatrix` and `Mat` have different
/// internal storage representations.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "nalgebra")] {
/// use oxiblas_matrix::nalgebra_compat::dmatrix_to_mat;
/// use nalgebra::DMatrix;
///
/// let na = DMatrix::from_fn(3, 4, |i, j| (i * 4 + j) as f64);
/// let mat = dmatrix_to_mat(&na);
/// assert_eq!(mat.nrows(), 3);
/// assert_eq!(mat.ncols(), 4);
/// # }
/// ```
pub fn dmatrix_to_mat<T: Scalar + Clone + Zero>(dm: &DMatrix<T>) -> Mat<T> {
    let nrows = dm.nrows();
    let ncols = dm.ncols();
    let mut mat = Mat::filled(nrows, ncols, T::zero());

    // Copy element by element (handles any storage layout)
    for j in 0..ncols {
        for i in 0..nrows {
            mat[(i, j)] = dm[(i, j)];
        }
    }

    mat
}

/// Converts an owned `Mat` to a nalgebra `DMatrix`.
///
/// This always creates a copy since the storage formats differ.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "nalgebra")] {
/// use oxiblas_matrix::{Mat, nalgebra_compat::mat_to_dmatrix};
///
/// let mat: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
/// let na = mat_to_dmatrix(&mat);
/// assert_eq!(na[(0, 0)], 1.0);
/// assert_eq!(na[(1, 1)], 4.0);
/// # }
/// ```
pub fn mat_to_dmatrix<T: Scalar + Clone + Zero + nalgebra::Scalar>(mat: &Mat<T>) -> DMatrix<T> {
    let nrows = mat.nrows();
    let ncols = mat.ncols();

    DMatrix::from_fn(nrows, ncols, |i, j| mat[(i, j)])
}

/// Converts a `MatRef` to a nalgebra `DMatrix`.
///
/// Creates a copy of the data.
///
/// # Example
///
/// ```
/// # #[cfg(feature = "nalgebra")] {
/// use oxiblas_matrix::{Mat, nalgebra_compat::mat_ref_to_dmatrix};
///
/// let mat: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
/// let na = mat_ref_to_dmatrix(mat.as_ref());
/// assert_eq!(na.nrows(), 2);
/// assert_eq!(na.ncols(), 2);
/// # }
/// ```
pub fn mat_ref_to_dmatrix<T: Scalar + Clone + Zero + nalgebra::Scalar>(
    mat: MatRef<'_, T>,
) -> DMatrix<T> {
    let nrows = mat.nrows();
    let ncols = mat.ncols();

    DMatrix::from_fn(nrows, ncols, |i, j| mat[(i, j)])
}

// =============================================================================
// Zero-copy view conversions
// =============================================================================
//
// Unlike the owned-to-owned conversions above, the functions in this section
// do not copy any data: they construct a view that borrows the source's
// existing buffer. See the "Which conversions are zero-copy" section of the
// module docs for why this is sound.

/// Creates a zero-copy, read-only `DMatrixView` over a `MatRef`.
///
/// No data is copied: the returned view borrows `mat`'s buffer directly.
/// This is possible because `MatRef` and `DMatrixView<'_, T>` (with its
/// default strides `RStride = U1`, `CStride = Dyn`) describe the exact same
/// memory layout -- elements within a column are contiguous (`RStride =
/// 1`), and `mat.row_stride()` is precisely nalgebra's dynamic leading
/// dimension (`CStride`).
///
/// # Example
///
/// ```
/// use oxiblas_matrix::{Mat, nalgebra_compat::mat_ref_to_dmatrix_view};
///
/// let mat: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
/// let view = mat_ref_to_dmatrix_view(mat.as_ref());
///
/// assert_eq!(view[(0, 0)], 1.0);
/// assert_eq!(view[(1, 1)], 4.0);
/// // Same buffer, no copy:
/// assert_eq!(view.as_ptr(), mat.as_ptr());
/// ```
pub fn mat_ref_to_dmatrix_view<'a, T: Scalar + nalgebra::Scalar>(
    mat: MatRef<'a, T>,
) -> DMatrixView<'a, T> {
    let shape = (Dyn(mat.nrows()), Dyn(mat.ncols()));
    let strides = (U1, Dyn(mat.row_stride()));

    // Safety: `MatRef` guarantees, by its own construction contract, that
    // `mat.as_ptr()` refers to valid, initialized column-major data with
    // `mat.nrows()` contiguous elements per column (row stride 1), spaced
    // `mat.row_stride()` elements apart across `mat.ncols()` columns --
    // exactly the layout `ViewStorage::from_raw_parts` requires. The
    // borrow's lifetime `'a` matches `mat`'s own lifetime, so the view
    // cannot outlive the data it points to.
    let storage: ViewStorage<'a, T, Dyn, Dyn, U1, Dyn> =
        unsafe { ViewStorage::from_raw_parts(mat.as_ptr(), shape, strides) };

    Matrix::from_data(storage)
}

/// Creates a zero-copy, mutable `DMatrixViewMut` over a `MatMut`.
///
/// No data is copied: the returned view borrows `mat`'s buffer directly.
/// See [`mat_ref_to_dmatrix_view`] for why the layouts coincide.
///
/// # Example
///
/// ```
/// use oxiblas_matrix::{Mat, nalgebra_compat::mat_mut_to_dmatrix_view_mut};
///
/// let mut mat: Mat<f64> = Mat::zeros(2, 2);
/// {
///     let mut view = mat_mut_to_dmatrix_view_mut(mat.as_mut());
///     view[(0, 1)] = 7.0;
/// }
/// assert_eq!(mat[(0, 1)], 7.0);
/// ```
pub fn mat_mut_to_dmatrix_view_mut<'a, T: Scalar + nalgebra::Scalar>(
    mut mat: MatMut<'a, T>,
) -> DMatrixViewMut<'a, T> {
    let shape = (Dyn(mat.nrows()), Dyn(mat.ncols()));
    let strides = (U1, Dyn(mat.row_stride()));
    let ptr = mat.as_mut_ptr();

    // Safety: same layout guarantee as `mat_ref_to_dmatrix_view`, and
    // `MatMut` guarantees exclusive (`&mut`-like) access to the pointed-to
    // data for its lifetime `'a`, matching `ViewStorageMut`'s aliasing
    // requirement.
    let storage: ViewStorageMut<'a, T, Dyn, Dyn, U1, Dyn> =
        unsafe { ViewStorageMut::from_raw_parts(ptr, shape, strides) };

    Matrix::from_data(storage)
}

/// Creates a zero-copy `MatRef` view over a nalgebra `DMatrix`.
///
/// No data is copied. `DMatrix`'s backing `VecStorage` is always fully
/// packed column-major data (no padding between columns), so its leading
/// dimension is exactly `dm.nrows()` -- which is exactly what `MatRef`
/// expects as its `row_stride`.
///
/// # Example
///
/// ```
/// use nalgebra::DMatrix;
/// use oxiblas_matrix::nalgebra_compat::dmatrix_to_mat_ref;
///
/// let dm = DMatrix::from_fn(2, 2, |i, j| (i + j) as f64);
/// let view = dmatrix_to_mat_ref(&dm);
///
/// assert_eq!(view[(1, 1)], 2.0);
/// assert_eq!(view.as_ptr(), dm.as_ptr());
/// ```
pub fn dmatrix_to_mat_ref<T: Scalar + nalgebra::Scalar>(dm: &DMatrix<T>) -> MatRef<'_, T> {
    // SAFETY: `dm` holds initialized, aligned elements that outlive the
    // returned view's borrow, and its packed column-major layout means
    // row_stride == nrows >= nrows trivially, so every in-bounds (i, j)
    // offset lies within the backing allocation.
    unsafe { MatRef::new(dm.as_ptr(), dm.nrows(), dm.ncols(), dm.nrows()) }
}

/// Creates a zero-copy `MatMut` view over a nalgebra `DMatrix`.
///
/// No data is copied. See [`dmatrix_to_mat_ref`] for why the layouts
/// coincide.
///
/// # Example
///
/// ```
/// use nalgebra::DMatrix;
/// use oxiblas_matrix::nalgebra_compat::dmatrix_to_mat_mut;
///
/// let mut dm = DMatrix::from_fn(2, 2, |i, j| (i + j) as f64);
/// {
///     let mut view = dmatrix_to_mat_mut(&mut dm);
///     view[(0, 0)] = 9.0;
/// }
/// assert_eq!(dm[(0, 0)], 9.0);
/// ```
pub fn dmatrix_to_mat_mut<T: Scalar + nalgebra::Scalar>(dm: &mut DMatrix<T>) -> MatMut<'_, T> {
    let nrows = dm.nrows();
    let ncols = dm.ncols();
    // SAFETY: a nalgebra `DMatrix` stores `nrows * ncols` initialized, aligned
    // elements contiguously in column-major order with leading dimension
    // `nrows`, so every in-bounds `(i, j)` offset is within the allocation and
    // no two indices alias; `&mut dm` keeps it alive and exclusive for `'_`.
    unsafe { MatMut::new(dm.as_mut_ptr(), nrows, ncols, nrows) }
}

/// Creates a `Mat` from a nalgebra `DMatrixView`.
///
/// Creates a copy of the viewed data.
pub fn dmatrix_view_to_mat<T: Scalar + Clone + Zero>(view: DMatrixView<'_, T>) -> Mat<T> {
    let nrows = view.nrows();
    let ncols = view.ncols();
    let mut mat = Mat::filled(nrows, ncols, T::zero());

    for j in 0..ncols {
        for i in 0..nrows {
            mat[(i, j)] = view[(i, j)];
        }
    }

    mat
}

/// Extension trait for `Mat` to provide nalgebra conversions.
pub trait MatNalgebraExt<T: Scalar> {
    /// Converts to a nalgebra `DMatrix`. Always copies.
    fn to_dmatrix(&self) -> DMatrix<T>
    where
        T: Clone + Zero + nalgebra::Scalar;

    /// Creates a zero-copy `DMatrixView` borrowing this matrix's data.
    fn to_dmatrix_view(&self) -> DMatrixView<'_, T>
    where
        T: nalgebra::Scalar;
}

impl<T: Scalar> MatNalgebraExt<T> for Mat<T> {
    fn to_dmatrix(&self) -> DMatrix<T>
    where
        T: Clone + Zero + nalgebra::Scalar,
    {
        mat_to_dmatrix(self)
    }

    fn to_dmatrix_view(&self) -> DMatrixView<'_, T>
    where
        T: nalgebra::Scalar,
    {
        mat_ref_to_dmatrix_view(self.as_ref())
    }
}

impl<T: Scalar> MatNalgebraExt<T> for MatRef<'_, T> {
    fn to_dmatrix(&self) -> DMatrix<T>
    where
        T: Clone + Zero + nalgebra::Scalar,
    {
        mat_ref_to_dmatrix(*self)
    }

    fn to_dmatrix_view(&self) -> DMatrixView<'_, T>
    where
        T: nalgebra::Scalar,
    {
        mat_ref_to_dmatrix_view(*self)
    }
}

/// Extension trait for nalgebra types to provide OxiBLAS conversions.
pub trait DMatrixOxiblasExt<T: Scalar> {
    /// Converts to an OxiBLAS `Mat`. Always copies.
    fn to_mat(&self) -> Mat<T>
    where
        T: Clone + Zero;

    /// Creates a zero-copy `MatRef` view borrowing this matrix's data.
    fn to_mat_ref(&self) -> MatRef<'_, T>
    where
        T: nalgebra::Scalar;
}

impl<T: Scalar + nalgebra::Scalar> DMatrixOxiblasExt<T> for DMatrix<T> {
    fn to_mat(&self) -> Mat<T>
    where
        T: Clone + Zero,
    {
        dmatrix_to_mat(self)
    }

    fn to_mat_ref(&self) -> MatRef<'_, T>
    where
        T: nalgebra::Scalar,
    {
        dmatrix_to_mat_ref(self)
    }
}

impl<T: Scalar + nalgebra::Scalar> DMatrixOxiblasExt<T> for DMatrixView<'_, T> {
    fn to_mat(&self) -> Mat<T>
    where
        T: Clone + Zero,
    {
        dmatrix_view_to_mat(*self)
    }

    fn to_mat_ref(&self) -> MatRef<'_, T>
    where
        T: nalgebra::Scalar,
    {
        // A `DMatrixView` may itself carry an arbitrary leading dimension
        // (e.g. a view over a sub-block of a larger `DMatrix`), so we
        // rebuild a `MatRef` directly from its own shape/pointer/stride
        // rather than routing through `dmatrix_to_mat_ref` (which assumes
        // the fully-packed layout that only owned `DMatrix` guarantees).
        // SAFETY: `self` (a `DMatrixView`) holds initialized, aligned elements
        // that outlive the returned view's borrow, and its own `strides().1`
        // is exactly the leading dimension nalgebra uses to address it, which
        // is always >= nrows for a valid view.
        unsafe { MatRef::new(self.as_ptr(), self.nrows(), self.ncols(), self.strides().1) }
    }
}

/// Implements `From<DMatrix<T>>` for `Mat<T>`.
impl<T: Scalar + Clone + Zero + nalgebra::Scalar> From<DMatrix<T>> for Mat<T> {
    fn from(dm: DMatrix<T>) -> Self {
        dmatrix_to_mat(&dm)
    }
}

/// Implements `From<&DMatrix<T>>` for `Mat<T>`.
impl<T: Scalar + Clone + Zero + nalgebra::Scalar> From<&DMatrix<T>> for Mat<T> {
    fn from(dm: &DMatrix<T>) -> Self {
        dmatrix_to_mat(dm)
    }
}

/// Implements `From<Mat<T>>` for `DMatrix<T>`.
impl<T: Scalar + Clone + Zero + nalgebra::Scalar> From<Mat<T>> for DMatrix<T> {
    fn from(mat: Mat<T>) -> Self {
        mat_to_dmatrix(&mat)
    }
}

/// Implements `From<&Mat<T>>` for `DMatrix<T>`.
impl<T: Scalar + Clone + Zero + nalgebra::Scalar> From<&Mat<T>> for DMatrix<T> {
    fn from(mat: &Mat<T>) -> Self {
        mat_to_dmatrix(mat)
    }
}

// Vector conversions

use nalgebra::DVector;

/// Converts a nalgebra `DVector` to a column vector `Mat`.
pub fn dvector_to_mat<T: Scalar + Clone + Zero>(dv: &DVector<T>) -> Mat<T> {
    let n = dv.len();
    let mut mat = Mat::filled(n, 1, T::zero());
    for i in 0..n {
        mat[(i, 0)] = dv[i];
    }
    mat
}

/// Converts a column vector `Mat` to a nalgebra `DVector`.
///
/// # Panics
///
/// Panics if the matrix has more than one column.
pub fn mat_to_dvector<T: Scalar + Clone + Zero + nalgebra::Scalar>(mat: &Mat<T>) -> DVector<T> {
    assert_eq!(mat.ncols(), 1, "Matrix must be a column vector");
    let n = mat.nrows();
    DVector::from_fn(n, |i, _| mat[(i, 0)])
}

/// Implements `From<DVector<T>>` for `Mat<T>`.
impl<T: Scalar + Clone + Zero + nalgebra::Scalar> From<DVector<T>> for Mat<T> {
    fn from(dv: DVector<T>) -> Self {
        dvector_to_mat(&dv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_dmatrix_to_mat_f64() {
        let dm = DMatrix::from_fn(3, 4, |i, j| (i * 4 + j) as f64);
        let mat = dmatrix_to_mat(&dm);

        assert_eq!(mat.nrows(), 3);
        assert_eq!(mat.ncols(), 4);

        for j in 0..4 {
            for i in 0..3 {
                assert_relative_eq!(mat[(i, j)], dm[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_mat_to_dmatrix_f64() {
        let mat: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let dm = mat_to_dmatrix(&mat);

        assert_eq!(dm.nrows(), 2);
        assert_eq!(dm.ncols(), 3);

        for j in 0..3 {
            for i in 0..2 {
                assert_relative_eq!(dm[(i, j)], mat[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_roundtrip_f64() {
        let original: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);
        let dm = mat_to_dmatrix(&original);
        let recovered = dmatrix_to_mat(&dm);

        assert_eq!(original.nrows(), recovered.nrows());
        assert_eq!(original.ncols(), recovered.ncols());

        for j in 0..original.ncols() {
            for i in 0..original.nrows() {
                assert_relative_eq!(original[(i, j)], recovered[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_from_trait_dmatrix_to_mat() {
        let dm = DMatrix::from_fn(2, 2, |i, j| (i + j) as f64);
        let mat: Mat<f64> = dm.clone().into();

        assert_eq!(mat[(0, 0)], 0.0);
        assert_eq!(mat[(0, 1)], 1.0);
        assert_eq!(mat[(1, 0)], 1.0);
        assert_eq!(mat[(1, 1)], 2.0);
    }

    #[test]
    fn test_from_trait_mat_to_dmatrix() {
        let mat: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let dm: DMatrix<f64> = mat.clone().into();

        assert_eq!(dm[(0, 0)], 1.0);
        assert_eq!(dm[(0, 1)], 2.0);
        assert_eq!(dm[(1, 0)], 3.0);
        assert_eq!(dm[(1, 1)], 4.0);
    }

    #[test]
    fn test_extension_traits() {
        let mat: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let dm = mat.to_dmatrix();
        let recovered = dm.to_mat();

        assert_eq!(mat[(0, 0)], recovered[(0, 0)]);
        assert_eq!(mat[(1, 1)], recovered[(1, 1)]);
    }

    #[test]
    fn test_vector_conversions() {
        let dv = DVector::from_vec(vec![1.0f64, 2.0, 3.0, 4.0]);
        let mat = dvector_to_mat(&dv);

        assert_eq!(mat.nrows(), 4);
        assert_eq!(mat.ncols(), 1);
        assert_eq!(mat[(0, 0)], 1.0);
        assert_eq!(mat[(3, 0)], 4.0);

        let dv2 = mat_to_dvector(&mat);
        assert_eq!(dv2[0], 1.0);
        assert_eq!(dv2[3], 4.0);
    }

    #[test]
    fn test_f32_conversions() {
        let dm = DMatrix::from_fn(2, 3, |i, j| (i + j) as f32);
        let mat = dmatrix_to_mat(&dm);
        let dm2 = mat_to_dmatrix(&mat);

        for j in 0..3 {
            for i in 0..2 {
                assert_relative_eq!(dm[(i, j)], dm2[(i, j)], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_complex_conversions() {
        use num_complex::Complex64;

        let dm = DMatrix::from_fn(2, 2, |i, j| Complex64::new((i + j) as f64, (i * j) as f64));
        let mat = dmatrix_to_mat(&dm);
        let dm2 = mat_to_dmatrix(&mat);

        for j in 0..2 {
            for i in 0..2 {
                assert_relative_eq!(dm[(i, j)].re, dm2[(i, j)].re, epsilon = 1e-10);
                assert_relative_eq!(dm[(i, j)].im, dm2[(i, j)].im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_empty_matrix() {
        let dm: DMatrix<f64> = DMatrix::zeros(0, 0);
        let mat = dmatrix_to_mat(&dm);
        assert_eq!(mat.nrows(), 0);
        assert_eq!(mat.ncols(), 0);
    }

    #[test]
    fn test_single_element() {
        let dm = DMatrix::from_element(1, 1, 42.0f64);
        let mat = dmatrix_to_mat(&dm);
        assert_eq!(mat.nrows(), 1);
        assert_eq!(mat.ncols(), 1);
        assert_eq!(mat[(0, 0)], 42.0);
    }

    #[test]
    fn test_large_matrix() {
        let dm = DMatrix::from_fn(100, 100, |i, j| (i * 100 + j) as f64);
        let mat = dmatrix_to_mat(&dm);
        let dm2 = mat_to_dmatrix(&mat);

        // Check corners and center
        assert_relative_eq!(dm[(0, 0)], dm2[(0, 0)], epsilon = 1e-10);
        assert_relative_eq!(dm[(99, 99)], dm2[(99, 99)], epsilon = 1e-10);
        assert_relative_eq!(dm[(50, 50)], dm2[(50, 50)], epsilon = 1e-10);
    }

    // Regression tests for finding #3: the module docs used to promise
    // "Zero-Copy Views" while every conversion actually copied. These tests
    // pin down that the new view functions really do borrow the same
    // buffer (pointer identity) rather than allocating a copy, and that
    // values read back correctly through the strides.

    #[test]
    fn test_mat_ref_to_dmatrix_view_zero_copy() {
        let mat: Mat<f64> = Mat::from_rows(&[
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            &[6.0, 7.0, 8.0, 9.0, 10.0],
            &[11.0, 12.0, 13.0, 14.0, 15.0],
        ]);
        // With a 3-row matrix, `row_stride` is padded up to a cache-line
        // multiple, so this genuinely exercises the strided (`CStride !=
        // nrows`) case, not just the trivially-contiguous case.
        assert!(mat.row_stride() > mat.nrows());

        let original_ptr = mat.as_ptr();
        let view = mat_ref_to_dmatrix_view(mat.as_ref());

        // Same buffer: this is a genuine zero-copy borrow, not a copy.
        assert_eq!(view.as_ptr(), original_ptr);
        assert_eq!(view.nrows(), 3);
        assert_eq!(view.ncols(), 5);

        for j in 0..5 {
            for i in 0..3 {
                assert_eq!(view[(i, j)], mat[(i, j)]);
            }
        }
    }

    #[test]
    fn test_mat_mut_to_dmatrix_view_mut_zero_copy() {
        let mut mat: Mat<f64> = Mat::zeros(3, 5);
        assert!(mat.row_stride() > mat.nrows());
        let original_ptr = mat.as_ptr();

        {
            let mut view = mat_mut_to_dmatrix_view_mut(mat.as_mut());
            assert_eq!(view.as_ptr(), original_ptr);
            for j in 0..5 {
                for i in 0..3 {
                    view[(i, j)] = (i * 10 + j) as f64;
                }
            }
        }

        // Mutations through the nalgebra view must be visible in `mat`
        // afterwards, proving they landed in the same buffer.
        for j in 0..5 {
            for i in 0..3 {
                assert_eq!(mat[(i, j)], (i * 10 + j) as f64);
            }
        }
    }

    #[test]
    fn test_dmatrix_to_mat_ref_zero_copy() {
        let dm = DMatrix::from_fn(4, 3, |i, j| (i * 3 + j) as f64);
        let original_ptr = dm.as_ptr();

        let view = dmatrix_to_mat_ref(&dm);

        assert_eq!(view.as_ptr(), original_ptr);
        assert_eq!(view.shape(), (4, 3));
        for j in 0..3 {
            for i in 0..4 {
                assert_eq!(view[(i, j)], dm[(i, j)]);
            }
        }
    }

    #[test]
    fn test_dmatrix_to_mat_mut_zero_copy() {
        let mut dm = DMatrix::from_fn(3, 3, |_, _| 0.0f64);
        let original_ptr = dm.as_ptr();

        {
            let mut view = dmatrix_to_mat_mut(&mut dm);
            assert_eq!(view.as_ptr(), original_ptr);
            view[(1, 2)] = 42.0;
        }

        // Mutation through the `MatMut` view must be visible in `dm`.
        assert_eq!(dm[(1, 2)], 42.0);
    }

    #[test]
    fn test_nalgebra_ext_zero_copy_methods() {
        let mat: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let view = mat.to_dmatrix_view();
        assert_eq!(view.as_ptr(), mat.as_ptr());
        assert_eq!(view[(1, 0)], 3.0);

        let dm = DMatrix::from_fn(2, 2, |i, j| (i + j) as f64);
        let mref = dm.to_mat_ref();
        assert_eq!(mref.as_ptr(), dm.as_ptr());
        assert_eq!(mref[(1, 1)], 2.0);
    }
}
