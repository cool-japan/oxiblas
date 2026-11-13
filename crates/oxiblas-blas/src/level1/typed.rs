//! Compile-time routing from generic `T` to concrete float kernels.
//!
//! The generic entry points (`dot<T>`, `nrm2<T>`, `axpy<T>`, `cnrm2<T>`,
//! `gemv<T>`, the small-GEMM and thin-GEMM paths of level 3) hand their data
//! to typed SIMD kernels when `T` *is* `f64` or `f32`. These helpers perform
//! that hand-off: each one compares the two `TypeId`s and only reinterprets
//! when they are equal, i.e. when the "cast" is the identity. After
//! monomorphisation the comparison is a constant, so the routing costs nothing
//! at run time and the untaken branch is removed.
//!
//! This is the one implementation of that hand-off in the crate, compiled on
//! every target. The value and matrix-view helpers serve every target; the
//! slice helpers are compiled where a caller uses them (the aarch64 NEON
//! level-1 and level-2 kernels).

use core::any::TypeId;

use oxiblas_core::scalar::Scalar;
use oxiblas_matrix::{MatMut, MatRef};

/// Returns `a` viewed as a `MatRef<U>` (same pointer, shape and strides) when
/// `T` and `U` are the same type.
#[inline(always)]
pub(crate) fn same_matref<'a, T: Scalar, U: Scalar>(a: MatRef<'a, T>) -> Option<MatRef<'a, U>> {
    if TypeId::of::<T>() == TypeId::of::<U>() {
        // SAFETY: `T` and `U` are the same type (equal `TypeId`s), so the view's
        // pointer, shape and strides describe exactly the elements `a` already
        // borrows immutably for `'a`.
        Some(unsafe {
            MatRef::new_strided(
                a.as_ptr().cast::<U>(),
                a.nrows(),
                a.ncols(),
                a.row_stride(),
                a.col_stride(),
            )
        })
    } else {
        None
    }
}

/// Returns `c` viewed as a `MatMut<U>` (same pointer, shape and leading
/// dimension) when `T` and `U` are the same type; the exclusive borrow moves
/// into the result.
#[inline(always)]
pub(crate) fn same_matmut<'a, T: Scalar, U: Scalar>(mut c: MatMut<'a, T>) -> Option<MatMut<'a, U>> {
    if TypeId::of::<T>() == TypeId::of::<U>() {
        // SAFETY: `T` and `U` are the same type (equal `TypeId`s), so the view's
        // pointer, shape and leading dimension describe exactly the elements
        // `c` borrows exclusively for `'a`; `c` is consumed, so no other access
        // to them remains.
        Some(unsafe {
            MatMut::new(
                c.as_mut_ptr().cast::<U>(),
                c.nrows(),
                c.ncols(),
                c.row_stride(),
            )
        })
    } else {
        None
    }
}

/// Returns `x` viewed as `&[U]` when `T` and `U` are the same type.
#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
#[inline(always)]
pub(crate) fn same_slice<T: 'static, U: 'static>(x: &[T]) -> Option<&[U]> {
    if TypeId::of::<T>() == TypeId::of::<U>() {
        // SAFETY: `T` and `U` are the same type (equal `TypeId`s), so the
        // pointer, length, alignment and initialisation of `x` describe a valid
        // `[U]` for the same lifetime.
        Some(unsafe { core::slice::from_raw_parts(x.as_ptr().cast::<U>(), x.len()) })
    } else {
        None
    }
}

/// Returns `x` viewed as `&mut [U]` when `T` and `U` are the same type.
#[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
#[inline(always)]
pub(crate) fn same_slice_mut<T: 'static, U: 'static>(x: &mut [T]) -> Option<&mut [U]> {
    if TypeId::of::<T>() == TypeId::of::<U>() {
        // SAFETY: as in `same_slice`; the exclusive borrow of `x` is moved into
        // the returned slice, so no aliasing access to the data remains.
        Some(unsafe { core::slice::from_raw_parts_mut(x.as_mut_ptr().cast::<U>(), x.len()) })
    } else {
        None
    }
}

/// Returns `v` as a `U` when `T` and `U` are the same type.
#[inline(always)]
pub(crate) fn same_value<T: 'static + Copy, U: 'static + Copy>(v: T) -> Option<U> {
    if TypeId::of::<T>() == TypeId::of::<U>() {
        // SAFETY: `T` and `U` are the same type (equal `TypeId`s), so this reads
        // a `T` as itself (same size, alignment and validity).
        Some(unsafe { core::mem::transmute_copy::<T, U>(&v) })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(target_arch = "aarch64", not(feature = "force-scalar")))]
    #[test]
    fn test_same_type_slice_casts_are_identity() {
        let x = [1.5f64, -2.0, 3.25];
        let view: Option<&[f64]> = same_slice::<f64, f64>(&x);
        assert_eq!(view, Some(&x[..]));
        assert!(same_slice::<f64, f32>(&x).is_none());

        let mut y = [1.0f32, 2.0];
        if let Some(v) = same_slice_mut::<f32, f32>(&mut y) {
            v[1] = 7.0;
        }
        assert_eq!(y, [1.0, 7.0]);
        assert!(same_slice_mut::<f32, f64>(&mut y).is_none());
    }

    #[test]
    fn test_same_type_value_and_view_casts_are_identity() {
        assert_eq!(same_value::<f64, f64>(-0.5), Some(-0.5));
        assert_eq!(same_value::<f64, f32>(-0.5), None);
        assert_eq!(
            same_value::<f64, f64>(-0.0).map(f64::to_bits),
            Some((-0.0f64).to_bits())
        );

        let data = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = MatRef::from_column_major(&data, 2, 3).expect("2x3 view");
        let view = same_matref::<f64, f64>(a).expect("same type");
        assert_eq!((view.nrows(), view.ncols()), (2, 3));
        assert_eq!(view[(1, 2)], 6.0);
        assert!(same_matref::<f64, f32>(a).is_none());

        let mut store = [0.0f32; 6];
        let c = MatMut::from_strided(&mut store, 2, 2, 3).expect("2x2 view, ld 3");
        let mut view = same_matmut::<f32, f32>(c).expect("same type");
        assert_eq!((view.nrows(), view.ncols(), view.row_stride()), (2, 2, 3));
        view.set(1, 1, 7.5);
        assert_eq!(store, [0.0, 0.0, 0.0, 0.0, 7.5, 0.0]);
        let c = MatMut::from_strided(&mut store, 2, 2, 3).expect("2x2 view, ld 3");
        assert!(same_matmut::<f32, f64>(c).is_none());
    }
}
