//! Conversion utilities between ndarray and OxiBLAS types.
//!
//! This module provides two distinct kinds of conversions between
//! ndarray's `Array2`/`ArrayView2`/`ArrayViewMut2` and OxiBLAS's
//! `Mat`/`MatRef`/`MatMut` types, and it is important not to confuse them:
//!
//! - **Genuinely zero-copy views**: [`array_view_to_mat_ref`],
//!   [`array_view_to_mat_ref_or_transposed`], [`array_view_mut_to_mat_mut`],
//!   [`array_viewd_to_mat_ref`], [`array_viewd_to_mat_ref_or_transposed`],
//!   and [`array_view_mutd_to_mat_mut`] wrap the source array's *existing*
//!   buffer in a borrowing `MatRef`/`MatMut` - no allocation, no element
//!   copy - whenever the array is contiguous along one axis with a
//!   non-negative stride. They return `None` for layouts that cannot be
//!   represented this way (non-contiguous, or a reversed/negative-stride
//!   axis, which `MatRef`/`MatMut` do not support).
//! - **Copying conversions**: [`array2_to_mat`], [`array2_into_mat`],
//!   [`arrayd_to_mat`], [`arrayd_into_mat`], [`mat_to_array2`],
//!   [`mat_to_array2_c`], [`mat_ref_to_array2`], [`mat_to_arrayd`], and
//!   [`mat_ref_to_arrayd`] always allocate a brand-new buffer and copy
//!   every element. `Mat`'s storage (`AlignedVec`) is always a distinct,
//!   cache-line-aligned allocation with a possibly-padded column stride
//!   (see `oxiblas_matrix::Mat`'s memory layout docs), so it can never
//!   adopt/reuse an `ndarray::Array2`'s `Vec`-backed buffer - regardless
//!   of whether the source array happens to already be column-major. If
//!   you need a real zero-copy path, work with `MatRef`/`MatMut` via the
//!   view conversions above instead of `Mat`.

use ndarray::{
    Array1, Array2, ArrayD, ArrayView1, ArrayView2, ArrayViewD, ArrayViewMut1, ArrayViewMut2,
    ArrayViewMutD, IxDyn, ShapeBuilder,
};
use oxiblas_core::scalar::Field;
use oxiblas_matrix::{Mat, MatMut, MatRef};

// =============================================================================
// Array2 <-> Mat Conversions
// =============================================================================

/// Converts an ndarray Array2 to an OxiBLAS Mat by copying every element.
///
/// # Notes
///
/// This is **always a copying conversion**, regardless of whether `arr` is
/// column-major, row-major, or otherwise strided: `Mat` allocates its own
/// cache-line-aligned, potentially row-padded column-major buffer (see
/// `oxiblas_matrix::Mat`), which is structurally incompatible with
/// `ndarray`'s `Vec`-backed storage, so there is no layout for which
/// `arr`'s buffer could be reused. If you need a real zero-copy view
/// instead, use [`array_view_to_mat_ref`] (or
/// [`array_view_to_mat_ref_or_transposed`]), which borrow `arr`'s existing
/// buffer via `MatRef` with no allocation and no element copy whenever the
/// layout allows it.
pub fn array2_to_mat<T: Field + Clone>(arr: &Array2<T>) -> Mat<T>
where
    T: bytemuck::Zeroable,
{
    let (nrows, ncols) = arr.dim();
    let mut mat = Mat::zeros(nrows, ncols);
    for i in 0..nrows {
        for j in 0..ncols {
            mat[(i, j)] = arr[[i, j]];
        }
    }
    mat
}

/// Converts an ndarray Array2 to an OxiBLAS Mat, consuming the array.
///
/// # Notes
///
/// Despite taking `arr` by value, this **still copies every element**: it
/// is not more efficient than [`array2_to_mat`], for any layout of `arr`.
/// `Mat`'s backing `AlignedVec` is always a fresh, cache-line-aligned
/// allocation (potentially with column padding for SIMD alignment) built
/// through a different allocator path than `ndarray`'s `Vec`, so `arr`'s
/// buffer can never be moved into the returned `Mat`. `arr` is consumed
/// (and dropped) purely so callers don't have to hold onto it after this
/// call; prefer [`array2_to_mat`] if you still need `arr` afterwards.
pub fn array2_into_mat<T: Field + Clone>(arr: Array2<T>) -> Mat<T>
where
    T: bytemuck::Zeroable,
{
    array2_to_mat(&arr)
}

/// Converts an OxiBLAS Mat to an ndarray Array2.
///
/// Creates a column-major (Fortran order) Array2.
pub fn mat_to_array2<T: Field + Clone>(mat: &Mat<T>) -> Array2<T> {
    let (nrows, ncols) = mat.shape();
    // Create in Fortran order for efficient conversion
    Array2::from_shape_fn((nrows, ncols).f(), |(i, j)| mat[(i, j)])
}

/// Converts an OxiBLAS MatRef to an ndarray Array2.
///
/// Creates a column-major (Fortran order) Array2.
pub fn mat_ref_to_array2<T: Field + Clone>(mat: MatRef<'_, T>) -> Array2<T> {
    let (nrows, ncols) = (mat.nrows(), mat.ncols());
    Array2::from_shape_fn((nrows, ncols).f(), |(i, j)| mat[(i, j)])
}

/// Converts an OxiBLAS Mat to a row-major ndarray Array2.
pub fn mat_to_array2_c<T: Field + Clone>(mat: &Mat<T>) -> Array2<T> {
    let (nrows, ncols) = mat.shape();
    Array2::from_shape_fn((nrows, ncols), |(i, j)| mat[(i, j)])
}

// =============================================================================
// ArrayD <-> Mat Conversions (Dynamic Dimension)
// =============================================================================

/// Converts an ndarray ArrayD (dynamic dimension) to an OxiBLAS Mat.
///
/// # Panics
/// Panics if the array is not 2-dimensional.
///
/// # Example
/// ```
/// use ndarray::{ArrayD, IxDyn};
/// use oxiblas_ndarray::conversions::arrayd_to_mat;
///
/// let arr = ArrayD::from_shape_fn(IxDyn(&[3, 4]), |idx| (idx[0] * 4 + idx[1]) as f64);
/// let mat = arrayd_to_mat(&arr);
/// assert_eq!(mat.shape(), (3, 4));
/// ```
pub fn arrayd_to_mat<T: Field + Clone>(arr: &ArrayD<T>) -> Mat<T>
where
    T: bytemuck::Zeroable,
{
    assert_eq!(
        arr.ndim(),
        2,
        "ArrayD must be 2-dimensional for matrix conversion"
    );
    let shape = arr.shape();
    let nrows = shape[0];
    let ncols = shape[1];

    let mut mat = Mat::zeros(nrows, ncols);
    for i in 0..nrows {
        for j in 0..ncols {
            mat[(i, j)] = arr[[i, j].as_ref()];
        }
    }
    mat
}

/// Converts an ndarray ArrayD to an OxiBLAS Mat, consuming the array.
///
/// # Panics
/// Panics if the array is not 2-dimensional.
pub fn arrayd_into_mat<T: Field + Clone>(arr: ArrayD<T>) -> Mat<T>
where
    T: bytemuck::Zeroable,
{
    arrayd_to_mat(&arr)
}

/// Converts an OxiBLAS Mat to an ndarray ArrayD.
///
/// Creates a column-major (Fortran order) ArrayD.
pub fn mat_to_arrayd<T: Field + Clone>(mat: &Mat<T>) -> ArrayD<T> {
    let (nrows, ncols) = mat.shape();
    let mut arr = ArrayD::from_elem(IxDyn(&[nrows, ncols]), T::zero());
    for i in 0..nrows {
        for j in 0..ncols {
            arr[[i, j].as_ref()] = mat[(i, j)];
        }
    }
    arr
}

/// Converts an OxiBLAS MatRef to an ndarray ArrayD.
pub fn mat_ref_to_arrayd<T: Field + Clone>(mat: MatRef<'_, T>) -> ArrayD<T> {
    let (nrows, ncols) = (mat.nrows(), mat.ncols());
    let mut arr = ArrayD::from_elem(IxDyn(&[nrows, ncols]), T::zero());
    for i in 0..nrows {
        for j in 0..ncols {
            arr[[i, j].as_ref()] = mat[(i, j)];
        }
    }
    arr
}

/// Converts an ArrayD to an Array2.
///
/// # Panics
/// Panics if the array is not 2-dimensional.
pub fn arrayd_to_array2<T: Clone>(arr: &ArrayD<T>) -> Array2<T> {
    assert_eq!(arr.ndim(), 2, "ArrayD must be 2-dimensional");
    let shape = arr.shape();
    Array2::from_shape_fn((shape[0], shape[1]), |(i, j)| arr[[i, j].as_ref()].clone())
}

/// Converts an Array2 to an ArrayD.
///
/// Works correctly for empty arrays (`nrows == 0` and/or `ncols == 0`):
/// this clones `arr` and reinterprets its dimension type via
/// [`ArrayBase::into_dyn`](ndarray::ArrayBase::into_dyn), which never
/// indexes into `arr`, so there is no "template element" that would be
/// out of bounds on an empty array.
pub fn array2_to_arrayd<T: Clone>(arr: &Array2<T>) -> ArrayD<T> {
    arr.clone().into_dyn()
}

/// Converts a (possibly negative) `ndarray` stride into the non-negative
/// stride representation that `MatRef`/`MatMut` require.
///
/// `oxiblas_matrix::MatRef`/`MatMut` store their stride as a plain `usize`
/// and have no concept of a reversed/negative-stride axis. `ndarray`
/// views produced by operations such as `.slice(s![..;-1, ..])` or
/// `.invert_axis(...)` report a negative stride for the reversed axis;
/// casting that directly with `as usize` would wrap around into an
/// enormous positive value that later corrupts pointer arithmetic
/// (`ptr.add(...)`) - safe code silently invoking undefined behavior.
///
/// Returns `None` for negative strides so callers can honestly report the
/// layout as unsupported (falling back to a copying conversion) instead
/// of ever performing that unchecked cast.
#[inline]
fn non_negative_stride(stride: isize) -> Option<usize> {
    usize::try_from(stride).ok()
}

// =============================================================================
// ArrayViewD -> MatRef Conversion
// =============================================================================

/// Creates a MatRef view from an ndarray ArrayViewD.
///
/// # Returns
/// - `Some(MatRef)` if the array is 2D and in column-major order
/// - `None` if the array is not 2D or layout is incompatible
pub fn array_viewd_to_mat_ref<'a, T: Field>(arr: &'a ArrayViewD<'a, T>) -> Option<MatRef<'a, T>> {
    if arr.ndim() != 2 {
        return None;
    }

    let shape = arr.shape();
    let nrows = shape[0];
    let ncols = shape[1];
    let strides = arr.strides();

    // Check for column-major order: row stride = 1
    if strides[0] == 1 {
        let col_stride = non_negative_stride(strides[1])?;
        let ptr = arr.as_ptr();
        // SAFETY: ptr comes from arr.as_ptr() and is valid for the lifetime 'a
        // borrowed from arr; nrows/ncols come from arr.shape() and col_stride is the
        // verified non-negative column stride from arr.strides(), so the resulting
        // MatRef addresses only elements within the source array's allocation.
        Some(unsafe { MatRef::new(ptr, nrows, ncols, col_stride) })
    } else {
        None
    }
}

/// Creates a MatRef view from an ndarray ArrayViewD, handling row-major layout.
///
/// # Returns
/// - `Some((MatRef, false))` if the array is 2D and column-major
/// - `Some((MatRef, true))` if the array is 2D and row-major (MatRef is transposed)
/// - `None` if the array is not 2D or layout is incompatible
pub fn array_viewd_to_mat_ref_or_transposed<'a, T: Field>(
    arr: &'a ArrayViewD<'a, T>,
) -> Option<(MatRef<'a, T>, bool)> {
    if arr.ndim() != 2 {
        return None;
    }

    let shape = arr.shape();
    let nrows = shape[0];
    let ncols = shape[1];
    let strides = arr.strides();

    if strides[0] == 1 {
        // Column-major
        let col_stride = non_negative_stride(strides[1])?;
        let ptr = arr.as_ptr();
        // SAFETY: ptr comes from arr.as_ptr() and is valid for the lifetime 'a
        // borrowed from arr; nrows/ncols come from arr.shape() and col_stride is the
        // verified non-negative column stride from arr.strides().
        Some((unsafe { MatRef::new(ptr, nrows, ncols, col_stride) }, false))
    } else if strides[1] == 1 {
        // Row-major: treat as transposed column-major
        let row_stride = non_negative_stride(strides[0])?;
        let ptr = arr.as_ptr();
        // SAFETY: ptr is valid for lifetime 'a borrowed from arr; ncols/nrows are
        // swapped to describe the transposed view and row_stride is the verified
        // non-negative stride for that transposed traversal.
        Some((unsafe { MatRef::new(ptr, ncols, nrows, row_stride) }, true))
    } else {
        None
    }
}

// =============================================================================
// ArrayViewMutD -> MatMut Conversion
// =============================================================================

/// Creates a MatMut view from an ndarray ArrayViewMutD.
///
/// # Returns
/// - `Some(MatMut)` if the array is 2D and in column-major order
/// - `None` if the array is not 2D or layout is incompatible
pub fn array_view_mutd_to_mat_mut<'a, T: Field>(
    arr: &'a mut ArrayViewMutD<'a, T>,
) -> Option<MatMut<'a, T>> {
    if arr.ndim() != 2 {
        return None;
    }

    let shape = arr.shape();
    let nrows = shape[0];
    let ncols = shape[1];
    let strides = arr.strides();

    if strides[0] == 1 {
        let col_stride = non_negative_stride(strides[1])?;
        let ptr = arr.as_mut_ptr();
        // SAFETY: `strides[0] == 1` and `col_stride` is the (non-negative)
        // column stride of a live `ArrayViewMut`, so ndarray guarantees every
        // `(i, j)` with `i < nrows`, `j < ncols` maps to a distinct,
        // initialized, aligned element inside the array's allocation; the
        // exclusive borrow keeps it alive and unaliased for `'a`.
        Some(unsafe { MatMut::new(ptr, nrows, ncols, col_stride) })
    } else {
        None
    }
}

// =============================================================================
// ArrayView2 -> MatRef Zero-Copy Conversion
// =============================================================================

/// Creates a MatRef view from an ndarray ArrayView2.
///
/// # Returns
/// - `Some(MatRef)` if the array is in column-major (Fortran) order
/// - `None` if the array layout is incompatible
///
/// # Safety
/// The returned MatRef borrows from the ArrayView2.
pub fn array_view_to_mat_ref<'a, T: Field>(arr: &'a ArrayView2<'a, T>) -> Option<MatRef<'a, T>> {
    let (nrows, ncols) = arr.dim();
    let strides = arr.strides();

    // Check for column-major order: row stride = 1
    if strides[0] == 1 {
        let col_stride = non_negative_stride(strides[1])?;
        let ptr = arr.as_ptr();
        // SAFETY: ptr comes from arr.as_ptr() and is valid for the lifetime 'a
        // borrowed from arr; nrows/ncols come from arr.dim() and col_stride is the
        // verified non-negative column stride from arr.strides().
        Some(unsafe { MatRef::new(ptr, nrows, ncols, col_stride) })
    } else {
        None
    }
}

/// Creates a MatRef view from an ndarray ArrayView2, handling row-major layout
/// by returning a transposed view if needed.
///
/// # Returns
/// - `(MatRef, false)` if the array is column-major
/// - `(MatRef, true)` if the array is row-major (MatRef is transposed)
/// - `None` if the array layout is incompatible (non-contiguous)
pub fn array_view_to_mat_ref_or_transposed<'a, T: Field>(
    arr: &'a ArrayView2<'a, T>,
) -> Option<(MatRef<'a, T>, bool)> {
    let (nrows, ncols) = arr.dim();
    let strides = arr.strides();

    if strides[0] == 1 {
        // Column-major
        let col_stride = non_negative_stride(strides[1])?;
        let ptr = arr.as_ptr();
        // SAFETY: ptr comes from arr.as_ptr() and is valid for the lifetime 'a
        // borrowed from arr; nrows/ncols come from arr.dim() and col_stride is the
        // verified non-negative column stride from arr.strides().
        Some((unsafe { MatRef::new(ptr, nrows, ncols, col_stride) }, false))
    } else if strides[1] == 1 {
        // Row-major: treat as transposed column-major
        let row_stride = non_negative_stride(strides[0])?;
        let ptr = arr.as_ptr();
        // Return transposed dimensions
        // SAFETY: ptr is valid for lifetime 'a borrowed from arr; ncols/nrows are
        // swapped to describe the transposed view and row_stride is the verified
        // non-negative stride for that transposed traversal.
        Some((unsafe { MatRef::new(ptr, ncols, nrows, row_stride) }, true))
    } else {
        // Non-contiguous
        None
    }
}

// =============================================================================
// ArrayViewMut2 -> MatMut Zero-Copy Conversion
// =============================================================================

/// Creates a MatMut view from an ndarray ArrayViewMut2.
///
/// # Returns
/// - `Some(MatMut)` if the array is in column-major (Fortran) order
/// - `None` if the array layout is incompatible
pub fn array_view_mut_to_mat_mut<'a, T: Field>(
    arr: &'a mut ArrayViewMut2<'a, T>,
) -> Option<MatMut<'a, T>> {
    let (nrows, ncols) = arr.dim();
    let strides = arr.strides();

    if strides[0] == 1 {
        let col_stride = non_negative_stride(strides[1])?;
        let ptr = arr.as_mut_ptr();
        // SAFETY: `strides[0] == 1` and `col_stride` is the (non-negative)
        // column stride of a live `ArrayViewMut`, so ndarray guarantees every
        // `(i, j)` with `i < nrows`, `j < ncols` maps to a distinct,
        // initialized, aligned element inside the array's allocation; the
        // exclusive borrow keeps it alive and unaliased for `'a`.
        Some(unsafe { MatMut::new(ptr, nrows, ncols, col_stride) })
    } else {
        None
    }
}

// =============================================================================
// 1D Array Conversions (for vectors)
// =============================================================================

/// Converts an ndarray Array1 to a Vec.
pub fn array1_to_vec<T: Clone>(arr: &Array1<T>) -> Vec<T> {
    arr.iter().cloned().collect()
}

/// Converts a slice to an ndarray Array1.
pub fn slice_to_array1<T: Clone>(slice: &[T]) -> Array1<T> {
    Array1::from_vec(slice.to_vec())
}

/// Gets a slice from an ArrayView1 if contiguous.
pub fn array_view1_as_slice<'a, T>(arr: &'a ArrayView1<'a, T>) -> Option<&'a [T]> {
    arr.as_slice()
}

/// Gets a mutable slice from an ArrayViewMut1 if contiguous.
pub fn array_view1_as_slice_mut<'a, T>(arr: &'a mut ArrayViewMut1<'a, T>) -> Option<&'a mut [T]> {
    arr.as_slice_mut()
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Creates a column-major Array2 (Fortran order).
///
/// This is the preferred layout for OxiBLAS operations as it allows
/// zero-copy conversions.
pub fn zeros_f<T: Clone + Default>(nrows: usize, ncols: usize) -> Array2<T> {
    Array2::from_shape_fn((nrows, ncols).f(), |_| T::default())
}

/// Creates a column-major Array2 filled with a value.
pub fn filled_f<T: Clone>(nrows: usize, ncols: usize, value: T) -> Array2<T> {
    Array2::from_shape_fn((nrows, ncols).f(), |_| value.clone())
}

/// Checks if an Array2 is in column-major (Fortran) order.
pub fn is_column_major<T>(arr: &Array2<T>) -> bool {
    let strides = arr.strides();
    let (nrows, _) = arr.dim();
    strides[0] == 1 && strides[1] == nrows as isize
}

/// Checks if an Array2 is in row-major (C) order.
pub fn is_row_major<T>(arr: &Array2<T>) -> bool {
    let strides = arr.strides();
    let (_, ncols) = arr.dim();
    strides[0] == ncols as isize && strides[1] == 1
}

/// Converts a row-major Array2 to column-major.
pub fn to_column_major<T: Clone + Default>(arr: &Array2<T>) -> Array2<T> {
    let (nrows, ncols) = arr.dim();
    let mut result = zeros_f(nrows, ncols);
    for i in 0..nrows {
        for j in 0..ncols {
            result[[i, j]] = arr[[i, j]].clone();
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_array2_to_mat_rowmajor() {
        let arr = Array2::from_shape_fn((3, 4), |(i, j)| (i * 4 + j) as f64);
        let mat = array2_to_mat(&arr);

        assert_eq!(mat.shape(), (3, 4));
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(mat[(i, j)], arr[[i, j]]);
            }
        }
    }

    #[test]
    fn test_array2_to_mat_colmajor() {
        let arr: Array2<f64> = Array2::from_shape_fn((3, 4).f(), |(i, j)| (i * 4 + j) as f64);
        assert!(is_column_major(&arr));

        let mat = array2_to_mat(&arr);
        assert_eq!(mat.shape(), (3, 4));
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(mat[(i, j)], arr[[i, j]]);
            }
        }
    }

    #[test]
    fn test_mat_to_array2() {
        let mat: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let arr = mat_to_array2(&mat);

        assert_eq!(arr.dim(), (2, 3));
        assert_eq!(arr[[0, 0]], 1.0);
        assert_eq!(arr[[1, 2]], 6.0);
    }

    #[test]
    fn test_roundtrip() {
        let original = Array2::from_shape_fn((5, 7), |(i, j)| (i * 7 + j) as f64);
        let mat = array2_to_mat(&original);
        let recovered = mat_to_array2(&mat);

        for i in 0..5 {
            for j in 0..7 {
                assert!((original[[i, j]] - recovered[[i, j]]).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn test_is_column_major() {
        let col_major: Array2<f64> = Array2::zeros((3, 4).f());
        let row_major: Array2<f64> = Array2::zeros((3, 4));

        assert!(is_column_major(&col_major));
        assert!(!is_column_major(&row_major));
        assert!(is_row_major(&row_major));
        assert!(!is_row_major(&col_major));
    }

    #[test]
    fn test_to_column_major() {
        let row_major = Array2::from_shape_fn((3, 4), |(i, j)| (i * 4 + j) as f64);
        let col_major = to_column_major(&row_major);

        assert!(is_column_major(&col_major));
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(row_major[[i, j]], col_major[[i, j]]);
            }
        }
    }

    #[test]
    fn test_array_view_to_mat_ref_or_transposed() {
        // Column-major
        let col_major: Array2<f64> = Array2::from_shape_fn((3, 4).f(), |(i, j)| (i * 4 + j) as f64);
        let view = col_major.view();
        let (mat_ref, transposed) = array_view_to_mat_ref_or_transposed(&view).unwrap();
        assert!(!transposed);
        assert_eq!(mat_ref.shape(), (3, 4));

        // Row-major
        let row_major: Array2<f64> = Array2::from_shape_fn((3, 4), |(i, j)| (i * 4 + j) as f64);
        let view = row_major.view();
        let (mat_ref, transposed) = array_view_to_mat_ref_or_transposed(&view).unwrap();
        assert!(transposed);
        // Transposed: original is 3x4, so MatRef should be 4x3
        assert_eq!(mat_ref.shape(), (4, 3));
    }

    // =========================================================================
    // ArrayD (Dynamic Dimension) Tests
    // =========================================================================

    #[test]
    fn test_arrayd_to_mat() {
        let arr = ArrayD::from_shape_fn(IxDyn(&[3, 4]), |idx| (idx[0] * 4 + idx[1]) as f64);
        let mat = arrayd_to_mat(&arr);

        assert_eq!(mat.shape(), (3, 4));
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(mat[(i, j)], arr[[i, j].as_ref()]);
            }
        }
    }

    #[test]
    fn test_mat_to_arrayd() {
        let mat: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let arr = mat_to_arrayd(&mat);

        assert_eq!(arr.ndim(), 2);
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr[[0, 0].as_ref()], 1.0);
        assert_eq!(arr[[1, 2].as_ref()], 6.0);
    }

    #[test]
    fn test_arrayd_roundtrip() {
        let original = ArrayD::from_shape_fn(IxDyn(&[5, 7]), |idx| (idx[0] * 7 + idx[1]) as f64);
        let mat = arrayd_to_mat(&original);
        let recovered = mat_to_arrayd(&mat);

        assert_eq!(recovered.shape(), original.shape());
        for i in 0..5 {
            for j in 0..7 {
                assert!((original[[i, j].as_ref()] - recovered[[i, j].as_ref()]).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn test_arrayd_to_array2() {
        let arr_d = ArrayD::from_shape_fn(IxDyn(&[3, 4]), |idx| (idx[0] * 4 + idx[1]) as f64);
        let arr_2 = arrayd_to_array2(&arr_d);

        assert_eq!(arr_2.dim(), (3, 4));
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(arr_2[[i, j]], arr_d[[i, j].as_ref()]);
            }
        }
    }

    #[test]
    fn test_array2_to_arrayd() {
        let arr_2: Array2<f64> = Array2::from_shape_fn((3, 4), |(i, j)| (i * 4 + j) as f64);
        let arr_d = array2_to_arrayd(&arr_2);

        assert_eq!(arr_d.ndim(), 2);
        assert_eq!(arr_d.shape(), &[3, 4]);
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(arr_d[[i, j].as_ref()], arr_2[[i, j]]);
            }
        }
    }

    #[test]
    fn test_array2_to_arrayd_empty_does_not_panic() {
        // Regression test: array2_to_arrayd used to index arr[[0, 0]] as a
        // "template" element before checking for emptiness, which panicked
        // on any Array2 with a zero dimension.
        let empty_rows: Array2<f64> = Array2::from_shape_fn((0, 4), |_| 0.0);
        let arr_d = array2_to_arrayd(&empty_rows);
        assert_eq!(arr_d.shape(), &[0, 4]);

        let empty_cols: Array2<f64> = Array2::from_shape_fn((3, 0), |_| 0.0);
        let arr_d = array2_to_arrayd(&empty_cols);
        assert_eq!(arr_d.shape(), &[3, 0]);

        let empty_both: Array2<f64> = Array2::from_shape_fn((0, 0), |_| 0.0);
        let arr_d = array2_to_arrayd(&empty_both);
        assert_eq!(arr_d.shape(), &[0, 0]);
    }

    #[test]
    fn test_array_viewd_to_mat_ref() {
        let arr = ArrayD::from_shape_fn(IxDyn(&[3, 4]), |idx| (idx[0] * 4 + idx[1]) as f64);
        let view = arr.view();

        // Default ndarray layout is C-order (row-major), so column-major view should fail
        // unless we specifically create it that way
        let result = array_viewd_to_mat_ref(&view);
        // Row-major, so this should return None
        assert!(result.is_none());
    }

    #[test]
    fn test_array_viewd_to_mat_ref_or_transposed() {
        // Row-major ArrayD
        let arr = ArrayD::from_shape_fn(IxDyn(&[3, 4]), |idx| (idx[0] * 4 + idx[1]) as f64);
        let view = arr.view();

        let result = array_viewd_to_mat_ref_or_transposed(&view);
        assert!(result.is_some());
        let (mat_ref, transposed) = result.unwrap();
        assert!(transposed); // Row-major should be transposed
        assert_eq!(mat_ref.shape(), (4, 3)); // Transposed dimensions
    }

    #[test]
    #[should_panic(expected = "2-dimensional")]
    fn test_arrayd_to_mat_wrong_dim() {
        let arr = ArrayD::from_shape_fn(IxDyn(&[2, 3, 4]), |idx| idx[0] as f64);
        let _ = arrayd_to_mat(&arr);
    }

    #[test]
    fn test_array_viewd_wrong_dim() {
        // 3D array
        let arr = ArrayD::from_shape_fn(IxDyn(&[2, 3, 4]), |idx| idx[0] as f64);
        let view = arr.view();

        // Should return None for non-2D arrays
        assert!(array_viewd_to_mat_ref(&view).is_none());
        assert!(array_viewd_to_mat_ref_or_transposed(&view).is_none());
    }

    // =========================================================================
    // Negative-stride rejection tests (see `non_negative_stride`)
    // =========================================================================
    //
    // Regression tests: the view-based *-to-mat_ref/mat_mut conversions used
    // to cast a possibly-negative ndarray stride to `usize` unchecked. For a
    // reversed-axis view that cast silently wraps around into an enormous
    // bogus stride, which is safe-code UB the moment it feeds pointer
    // arithmetic. They must now report such layouts as `None` instead.

    #[test]
    fn test_array_view_to_mat_ref_rejects_negative_stride() {
        use ndarray::s;

        // Column-major 3x4 array: strides = [1, 3]. Reversing the column
        // axis flips strides[1] negative while strides[0] stays 1.
        let col_major: Array2<f64> = Array2::from_shape_fn((3, 4).f(), |(i, j)| (i * 4 + j) as f64);
        let reversed = col_major.slice(s![.., ..;-1]);
        assert_eq!(reversed.strides()[0], 1);
        assert!(reversed.strides()[1] < 0);

        assert!(array_view_to_mat_ref(&reversed).is_none());
        assert!(array_view_to_mat_ref_or_transposed(&reversed).is_none());
    }

    #[test]
    fn test_array_view_mut_to_mat_mut_rejects_negative_stride() {
        use ndarray::s;

        let mut col_major: Array2<f64> =
            Array2::from_shape_fn((3, 4).f(), |(i, j)| (i * 4 + j) as f64);
        let mut reversed = col_major.slice_mut(s![.., ..;-1]);
        assert_eq!(reversed.strides()[0], 1);
        assert!(reversed.strides()[1] < 0);

        assert!(array_view_mut_to_mat_mut(&mut reversed).is_none());
    }

    #[test]
    fn test_array_viewd_to_mat_ref_rejects_negative_stride() {
        use ndarray::s;

        // Build the negative-stride view on a statically 2D array first
        // (so the `s![]` macro produces a fixed Ix2 output dimension), then
        // erase the dimension to IxDyn while preserving the strides/layout.
        let col_major: Array2<f64> = Array2::from_shape_fn((3, 4).f(), |(i, j)| (i * 4 + j) as f64);
        let reversed = col_major.slice(s![.., ..;-1]).into_dyn();
        assert_eq!(reversed.strides()[0], 1);
        assert!(reversed.strides()[1] < 0);

        assert!(array_viewd_to_mat_ref(&reversed).is_none());
        assert!(array_viewd_to_mat_ref_or_transposed(&reversed).is_none());
    }
}
