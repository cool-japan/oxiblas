//! Compile-time routing from generic `T` to concrete float kernels.
//!
//! The generic entry points (`dot<T>`, `nrm2<T>`, `axpy<T>`, `cnrm2<T>`,
//! `gemv<T>`) hand their data to typed SIMD kernels when `T` *is* `f64` or
//! `f32`. These helpers perform that hand-off: each one compares the two
//! `TypeId`s and only reinterprets when they are equal, i.e. when the "cast" is
//! the identity. After monomorphisation the comparison is a constant, so the
//! routing costs nothing at run time and the untaken branch is removed.

use core::any::TypeId;

use oxiblas_core::scalar::Scalar;
use oxiblas_matrix::MatRef;

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

/// Returns `x` viewed as `&[U]` when `T` and `U` are the same type.
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

    #[test]
    fn test_same_type_casts_are_identity() {
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

        assert_eq!(same_value::<f64, f64>(-0.5), Some(-0.5));
        assert_eq!(same_value::<f64, f32>(-0.5), None);
        assert_eq!(
            same_value::<f64, f64>(-0.0).map(f64::to_bits),
            Some((-0.0f64).to_bits())
        );
    }
}
