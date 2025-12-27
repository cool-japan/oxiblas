//! nalgebra type conversions for OxiBLAS matrices.
//!
//! This module provides seamless conversion between OxiBLAS matrix types
//! ([`Mat`], [`MatRef`], [`crate::MatMut`]) and nalgebra types ([`DMatrix`], [`DMatrixView`]).
//!
//! # Zero-Copy Views
//!
//! When the memory layout is compatible (column-major with unit column stride),
//! conversions create zero-copy views. Otherwise, data is copied.
//!
//! # Example
//!
//! ```ignore
//! use oxiblas_matrix::prelude::*;
//! use oxiblas_matrix::nalgebra_compat::*;
//! use nalgebra::DMatrix;
//!
//! // Convert from nalgebra to oxiblas
//! let na_mat = DMatrix::from_fn(3, 3, |i, j| (i + j) as f64);
//! let oxi_mat: Mat<f64> = dmatrix_to_mat(&na_mat);
//!
//! // Convert from oxiblas to nalgebra
//! let mat: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
//! let na_mat: DMatrix<f64> = mat_to_dmatrix(&mat);
//! ```

use crate::{Mat, MatRef};
use nalgebra::{DMatrix, DMatrixView};
use num_traits::Zero;
use oxiblas_core::scalar::Scalar;

/// Converts a nalgebra `DMatrix` to an owned `Mat`.
///
/// This always creates a copy since `DMatrix` and `Mat` have different
/// internal storage representations.
///
/// # Example
///
/// ```ignore
/// use oxiblas_matrix::nalgebra_compat::dmatrix_to_mat;
/// use nalgebra::DMatrix;
///
/// let na = DMatrix::from_fn(3, 4, |i, j| (i * 4 + j) as f64);
/// let mat = dmatrix_to_mat(&na);
/// assert_eq!(mat.nrows(), 3);
/// assert_eq!(mat.ncols(), 4);
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
/// ```ignore
/// use oxiblas_matrix::{Mat, nalgebra_compat::mat_to_dmatrix};
///
/// let mat: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
/// let na = mat_to_dmatrix(&mat);
/// assert_eq!(na[(0, 0)], 1.0);
/// assert_eq!(na[(1, 1)], 4.0);
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
/// ```ignore
/// use oxiblas_matrix::{Mat, nalgebra_compat::mat_ref_to_dmatrix};
///
/// let mat: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
/// let na = mat_ref_to_dmatrix(mat.as_ref());
/// assert_eq!(na.nrows(), 2);
/// assert_eq!(na.ncols(), 2);
/// ```
pub fn mat_ref_to_dmatrix<T: Scalar + Clone + Zero + nalgebra::Scalar>(
    mat: MatRef<'_, T>,
) -> DMatrix<T> {
    let nrows = mat.nrows();
    let ncols = mat.ncols();

    DMatrix::from_fn(nrows, ncols, |i, j| mat[(i, j)])
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
    /// Converts to a nalgebra `DMatrix`.
    fn to_dmatrix(&self) -> DMatrix<T>
    where
        T: Clone + Zero + nalgebra::Scalar;
}

impl<T: Scalar> MatNalgebraExt<T> for Mat<T> {
    fn to_dmatrix(&self) -> DMatrix<T>
    where
        T: Clone + Zero + nalgebra::Scalar,
    {
        mat_to_dmatrix(self)
    }
}

impl<T: Scalar> MatNalgebraExt<T> for MatRef<'_, T> {
    fn to_dmatrix(&self) -> DMatrix<T>
    where
        T: Clone + Zero + nalgebra::Scalar,
    {
        mat_ref_to_dmatrix(*self)
    }
}

/// Extension trait for nalgebra types to provide OxiBLAS conversions.
pub trait DMatrixOxiblasExt<T: Scalar> {
    /// Converts to an OxiBLAS `Mat`.
    fn to_mat(&self) -> Mat<T>
    where
        T: Clone + Zero;
}

impl<T: Scalar + nalgebra::Scalar> DMatrixOxiblasExt<T> for DMatrix<T> {
    fn to_mat(&self) -> Mat<T>
    where
        T: Clone + Zero,
    {
        dmatrix_to_mat(self)
    }
}

impl<T: Scalar + nalgebra::Scalar> DMatrixOxiblasExt<T> for DMatrixView<'_, T> {
    fn to_mat(&self) -> Mat<T>
    where
        T: Clone + Zero,
    {
        dmatrix_view_to_mat(*self)
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
}
