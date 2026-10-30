//! Strided GEMV: general matrix-vector product with strided `x`/`y` vectors.
//!
//! Reference BLAS `?gemv` takes `incx`/`incy` increments for its vector
//! operands (the matrix keeps its own separate leading-dimension stride, which
//! is unrelated). This module restores that contract on top of the crate's
//! [`MatRef`] matrix view by accepting [`StridedSlice`] / [`StridedSliceMut`]
//! for the vectors. A unit-stride call is forwarded to the optimised
//! contiguous [`gemv`](super::gemv) so the fast path is unchanged; otherwise a
//! straightforward strided kernel is used.
//!
//! Negative and zero increments follow the same reference convention as the
//! Level 1 strided routines (see [`crate::level1::StridedSlice`]).

use oxiblas_core::scalar::Field;
use oxiblas_matrix::MatRef;

use super::GemvTrans;
use crate::level1::{StridedSlice, StridedSliceMut};

/// Computes `y ← α·op(A)·x + β·y` with strided `x` and `y`.
///
/// `op(A)` is `A`, `Aᵀ`, or `Aᴴ` per `trans`. The matrix `a` supplies its own
/// leading-dimension stride via [`MatRef`]; `x` and `y` supply their BLAS
/// increments via the strided views. Reference BLAS requires the vector
/// increments to be non-zero, but a zero increment is handled coherently here
/// (it aliases logical element 0, matching the Level 1 routines).
///
/// As in reference BLAS, an empty `A` (`m == 0` or `n == 0`) is a quick return
/// that leaves `y` untouched (it is not scaled by `beta`).
///
/// # Panics
///
/// Panics if the vector lengths do not match `op(A)`'s dimensions.
pub fn gemv_strided<T: Field>(
    trans: GemvTrans,
    alpha: T,
    a: MatRef<'_, T>,
    x: StridedSlice<'_, T>,
    beta: T,
    mut y: StridedSliceMut<'_, T>,
) {
    let (m, n) = (a.nrows(), a.ncols());
    let (rows, cols) = match trans {
        GemvTrans::NoTrans => (m, n),
        GemvTrans::Trans | GemvTrans::ConjTrans => (n, m),
    };

    assert_eq!(
        x.len(),
        cols,
        "x length must match number of columns of op(A)"
    );
    assert_eq!(y.len(), rows, "y length must match number of rows of op(A)");

    // Reference quick return: for an empty A (m == 0 or n == 0) y is left
    // untouched -- not even beta-scaled or zero-filled.
    if m == 0 || n == 0 {
        return;
    }

    // Fast path: both vectors unit-stride -> optimised contiguous kernel.
    if let (Some(xc), Some(yc)) = (x.as_contiguous(), y.as_contiguous_mut()) {
        super::gemv(trans, alpha, a, xc, beta, yc);
        return;
    }

    // Apply beta to y first (strided), mirroring the contiguous gemv.
    if beta == T::zero() {
        for i in 0..rows {
            y.set(i, T::zero());
        }
    } else if beta != T::one() {
        for i in 0..rows {
            let scaled = beta * y.get(i);
            y.set(i, scaled);
        }
    }

    if alpha == T::zero() {
        return;
    }

    match trans {
        GemvTrans::NoTrans => {
            for i in 0..m {
                let mut sum = T::zero();
                for j in 0..n {
                    sum += a[(i, j)] * x.get(j);
                }
                let updated = y.get(i) + alpha * sum;
                y.set(i, updated);
            }
        }
        GemvTrans::Trans => {
            for i in 0..m {
                let alpha_xi = alpha * x.get(i);
                for j in 0..n {
                    let updated = y.get(j) + a[(i, j)] * alpha_xi;
                    y.set(j, updated);
                }
            }
        }
        GemvTrans::ConjTrans => {
            for i in 0..m {
                let alpha_xi = alpha * x.get(i);
                for j in 0..n {
                    let updated = y.get(j) + a[(i, j)].conj() * alpha_xi;
                    y.set(j, updated);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiblas_matrix::Mat;

    /// Physical offset of logical element `i` under reference BLAS rules.
    fn ref_offset(i: usize, n: usize, inc: isize) -> usize {
        if inc >= 0 {
            i * inc.unsigned_abs()
        } else {
            (n - 1 - i) * inc.unsigned_abs()
        }
    }

    fn view(data: &[f64], n: usize, inc: isize) -> StridedSlice<'_, f64> {
        StridedSlice::try_new(data, n, inc).expect("view fits")
    }

    fn view_mut(data: &mut [f64], n: usize, inc: isize) -> StridedSliceMut<'_, f64> {
        StridedSliceMut::try_new(data, n, inc).expect("view fits")
    }

    #[test]
    fn test_gemv_strided_unit_matches_contiguous() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let x = [1.0, 2.0, 3.0];
        let mut y_legacy = [10.0, 20.0];
        let mut y_strided = y_legacy;

        super::super::gemv(GemvTrans::NoTrans, 2.0, a.as_ref(), &x, 3.0, &mut y_legacy);
        gemv_strided(
            GemvTrans::NoTrans,
            2.0,
            a.as_ref(),
            StridedSlice::from_slice(&x),
            3.0,
            StridedSliceMut::from_slice(&mut y_strided),
        );
        assert_eq!(y_strided, y_legacy);
    }

    #[test]
    fn test_gemv_strided_notrans_stride2() {
        // A is 2x3. x has 3 logical elements at stride 2; y has 2 at stride 2.
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let xbuf = [1.0, -9.0, 2.0, -9.0, 3.0]; // logical 1,2,3
        let mut ybuf = [0.0, -9.0, 0.0]; // logical y at offsets 0,2
        let n_x = 3;
        let n_y = 2;

        // Naive reference into ybuf.
        let mut expected = ybuf;
        for i in 0..n_y {
            let mut sum = 0.0;
            for j in 0..n_x {
                sum += a.as_ref()[(i, j)] * xbuf[ref_offset(j, n_x, 2)];
            }
            let oy = ref_offset(i, n_y, 2);
            expected[oy] = 1.0 * sum; // beta = 0
        }

        gemv_strided(
            GemvTrans::NoTrans,
            1.0,
            a.as_ref(),
            view(&xbuf, n_x, 2),
            0.0,
            view_mut(&mut ybuf, n_y, 2),
        );

        for k in 0..ybuf.len() {
            assert!((ybuf[k] - expected[k]).abs() < 1e-12, "index {k}");
        }
        // logical y = [1*1+2*2+3*3, 4*1+5*2+6*3] = [14, 32]
        assert!((ybuf[0] - 14.0).abs() < 1e-12);
        assert!((ybuf[2] - 32.0).abs() < 1e-12);
    }

    #[test]
    fn test_gemv_strided_trans_stride2() {
        // A is 2x3; A^T is 3x2. x has 2 logical elems, y has 3.
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let xbuf = [1.0, 0.0, 2.0]; // logical 1,2 at stride 2
        let mut ybuf = [0.0, 0.0, 0.0, 0.0, 0.0]; // 3 at stride 2
        let n_x = 2;
        let n_y = 3;

        gemv_strided(
            GemvTrans::Trans,
            1.0,
            a.as_ref(),
            view(&xbuf, n_x, 2),
            0.0,
            view_mut(&mut ybuf, n_y, 2),
        );

        // y = A^T x, A^T = [[1,4],[2,5],[3,6]], x=[1,2]
        // y = [1+8, 2+10, 3+12] = [9, 12, 15]
        assert!((ybuf[0] - 9.0).abs() < 1e-12);
        assert!((ybuf[2] - 12.0).abs() < 1e-12);
        assert!((ybuf[4] - 15.0).abs() < 1e-12);
    }

    #[test]
    fn test_gemv_strided_negative_incx() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        // x logical [1,2,3] with incx = -1 (buffer reversed): buffer = [3,2,1].
        let xbuf = [3.0, 2.0, 1.0];
        let mut ybuf = [0.0, 0.0];
        gemv_strided(
            GemvTrans::NoTrans,
            1.0,
            a.as_ref(),
            view(&xbuf, 3, -1),
            0.0,
            view_mut(&mut ybuf, 2, 1),
        );
        // logical x = [1,2,3] -> y = [14, 32]
        assert!((ybuf[0] - 14.0).abs() < 1e-12);
        assert!((ybuf[1] - 32.0).abs() < 1e-12);
    }

    #[test]
    fn test_gemv_strided_beta_scaling_strided_y() {
        let a = Mat::from_rows(&[&[1.0f64, 1.0], &[1.0, 1.0]]);
        let xbuf = [1.0, 0.0, 1.0]; // logical [1,1] at stride 2
        let mut ybuf = [10.0, -9.0, 20.0]; // logical y [10,20] at stride 2
        gemv_strided(
            GemvTrans::NoTrans,
            1.0,
            a.as_ref(),
            view(&xbuf, 2, 2),
            2.0,
            view_mut(&mut ybuf, 2, 2),
        );
        // A*x = [2,2]; y = 1*[2,2] + 2*[10,20] = [22, 42]
        assert!((ybuf[0] - 22.0).abs() < 1e-12);
        assert!((ybuf[1] - -9.0).abs() < 1e-12, "gap element untouched");
        assert!((ybuf[2] - 42.0).abs() < 1e-12);
    }

    // Issue #13: reference ?GEMV quick-returns when m == 0 or n == 0, leaving y
    // untouched (no beta scaling / zero-fill).

    #[test]
    fn test_issue_13_gemv_strided_empty_a_unit_stride_leaves_y_untouched() {
        // A is 0x1, op(A) = A^T is 1x0: x is empty, y has one element.
        let a: Mat<f64> = Mat::zeros(0, 1);
        let x: [f64; 0] = [];
        let mut y = [1.0f64];
        gemv_strided(
            GemvTrans::Trans,
            1.0,
            a.as_ref(),
            StridedSlice::from_slice(&x),
            0.5,
            StridedSliceMut::from_slice(&mut y),
        );
        assert_eq!(y, [1.0]);

        let mut y = [f64::NAN];
        gemv_strided(
            GemvTrans::Trans,
            1.0,
            a.as_ref(),
            StridedSlice::from_slice(&x),
            0.0,
            StridedSliceMut::from_slice(&mut y),
        );
        assert!(
            y[0].is_nan(),
            "beta = 0 must not zero-fill y when A is empty"
        );
    }

    #[test]
    fn test_issue_13_gemv_strided_empty_a_stride2_leaves_y_untouched() {
        // Same shapes, but with stride-2 views so the strided kernel (not the
        // contiguous fast path) is the code under test.
        let a: Mat<f64> = Mat::zeros(0, 1);
        let xbuf: [f64; 0] = [];
        let mut ybuf = [1.0f64, -9.0];
        gemv_strided(
            GemvTrans::Trans,
            1.0,
            a.as_ref(),
            view(&xbuf, 0, 2),
            0.5,
            view_mut(&mut ybuf, 1, 2),
        );
        assert_eq!(ybuf, [1.0, -9.0]);

        // n == 0, NoTrans: A is 2x0, x empty, y has two elements at stride 2.
        let a: Mat<f64> = Mat::zeros(2, 0);
        let mut ybuf = [f64::NAN, -9.0, 3.0];
        gemv_strided(
            GemvTrans::NoTrans,
            1.0,
            a.as_ref(),
            view(&xbuf, 0, 2),
            0.0,
            view_mut(&mut ybuf, 2, 2),
        );
        assert!(ybuf[0].is_nan());
        assert_eq!(ybuf[1], -9.0);
        assert_eq!(ybuf[2], 3.0);
    }
}
