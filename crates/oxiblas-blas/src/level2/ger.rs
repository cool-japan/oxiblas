//! GER: Rank-1 update.
//!
//! Computes A = α·x·y^T + A

use oxiblas_core::scalar::Field;
use oxiblas_matrix::MatMut;

/// `A[:, j] += x * temp`, element by element as `A(i,j) + x(i)*temp`: the
/// reference ?GER association (two roundings, no fused multiply-add) over the
/// contiguous column `j` of the column-major view.
#[inline]
fn add_scaled_column<T: Field>(a: &mut MatMut<'_, T>, j: usize, x: &[T], temp: T) {
    if let Some(col) = a.col_as_slice_mut(j) {
        for (aij, &xi) in col.iter_mut().zip(x) {
            *aij += xi * temp;
        }
    } else {
        // Unreachable today: `MatMut` is column-major with a unit row step, so
        // `col_as_slice_mut` only returns `None` for `j >= ncols`, which the callers
        // never pass. Should a layout change ever make a column non-contiguous,
        // update it element by element instead of silently dropping it.
        debug_assert!(
            false,
            "column {j} of the rank-1 update target is not contiguous"
        );
        let m = a.nrows();
        for (i, &xi) in x.iter().enumerate().take(m) {
            a[(i, j)] += xi * temp;
        }
    }
}

/// Performs the rank-1 update: A = α·x·y^T + A
///
/// As in reference BLAS `?ger` / `?geru`, the update runs column by column with
/// `temp = alpha * y[j]` and `A[i,j] += x[i] * temp`, and a column with
/// `y[j] == 0` is skipped entirely, so an infinite or NaN `x[i]` cannot reach it
/// as `x[i] * 0`. If `m == 0`, `n == 0` or `alpha == 0` the matrix is left
/// untouched.
///
/// # Arguments
///
/// * `alpha` - Scalar multiplier
/// * `x` - Column vector (m elements)
/// * `y` - Row vector (n elements)
/// * `a` - Matrix A (m×n), modified in place
///
/// # Panics
///
/// Panics if dimensions don't match.
///
/// # Example
///
/// ```
/// use oxiblas_blas::level2::ger;
/// use oxiblas_matrix::Mat;
///
/// let x = [1.0f64, 2.0, 3.0];
/// let y = [1.0f64, 2.0];
/// let mut a = Mat::zeros(3, 2);
///
/// // A = 1.0 * x * y^T + A
/// ger(1.0, &x, &y, a.as_mut());
///
/// // A = [[1, 2], [2, 4], [3, 6]]
/// assert!((a[(0, 0)] - 1.0).abs() < 1e-10);
/// assert!((a[(0, 1)] - 2.0).abs() < 1e-10);
/// assert!((a[(1, 0)] - 2.0).abs() < 1e-10);
/// assert!((a[(1, 1)] - 4.0).abs() < 1e-10);
/// assert!((a[(2, 0)] - 3.0).abs() < 1e-10);
/// assert!((a[(2, 1)] - 6.0).abs() < 1e-10);
/// ```
pub fn ger<T: Field>(alpha: T, x: &[T], y: &[T], mut a: MatMut<'_, T>) {
    let m = a.nrows();
    let n = a.ncols();

    assert_eq!(x.len(), m, "x length must match number of rows");
    assert_eq!(y.len(), n, "y length must match number of columns");

    if m == 0 || n == 0 || alpha == T::zero() {
        return;
    }

    // Compute A += α * x * y^T column by column, like reference ?GER: column j
    // is skipped when y(j) == 0 (so an infinite or NaN x(i) cannot reach it as
    // `x(i) * 0`), otherwise A(i,j) += x(i) * (alpha * y(j)).
    for (j, &yj) in y.iter().enumerate() {
        if yj == T::zero() {
            continue;
        }
        add_scaled_column(&mut a, j, x, alpha * yj);
    }
}

/// Performs the conjugated rank-1 update: A = α·x·conj(y)^T + A
///
/// For real types, this is identical to ger.
/// For complex types, y is conjugated. As in reference BLAS `?gerc`, the update
/// runs column by column with `temp = alpha * conj(y[j])` and
/// `A[i,j] += x[i] * temp`, and a column with `y[j] == 0` is skipped entirely,
/// so an infinite or NaN `x[i]` cannot reach it. If `m == 0`, `n == 0` or
/// `alpha == 0` the matrix is left untouched.
///
/// # Panics
///
/// Panics if dimensions don't match.
pub fn gerc<T: Field>(alpha: T, x: &[T], y: &[T], mut a: MatMut<'_, T>) {
    let m = a.nrows();
    let n = a.ncols();

    assert_eq!(x.len(), m, "x length must match number of rows");
    assert_eq!(y.len(), n, "y length must match number of columns");

    if m == 0 || n == 0 || alpha == T::zero() {
        return;
    }

    // Compute A += α * x * conj(y)^T column by column, like reference ?GERC:
    // column j is skipped when y(j) == 0, otherwise
    // A(i,j) += x(i) * (alpha * conj(y(j))).
    for (j, &yj) in y.iter().enumerate() {
        if yj == T::zero() {
            continue;
        }
        add_scaled_column(&mut a, j, x, alpha * yj.conj());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use oxiblas_matrix::Mat;

    #[test]
    fn test_ger_basic() {
        let x = [1.0f64, 2.0, 3.0];
        let y = [1.0f64, 2.0];
        let mut a = Mat::zeros(3, 2);

        ger(1.0, &x, &y, a.as_mut());

        // A = x * y^T = [[1,2], [2,4], [3,6]]
        assert!((a[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((a[(0, 1)] - 2.0).abs() < 1e-10);
        assert!((a[(1, 0)] - 2.0).abs() < 1e-10);
        assert!((a[(1, 1)] - 4.0).abs() < 1e-10);
        assert!((a[(2, 0)] - 3.0).abs() < 1e-10);
        assert!((a[(2, 1)] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_ger_with_alpha() {
        let x = [1.0f64, 2.0];
        let y = [1.0f64, 2.0];
        let mut a = Mat::zeros(2, 2);

        ger(2.0, &x, &y, a.as_mut());

        // A = 2 * x * y^T = [[2,4], [4,8]]
        assert!((a[(0, 0)] - 2.0).abs() < 1e-10);
        assert!((a[(0, 1)] - 4.0).abs() < 1e-10);
        assert!((a[(1, 0)] - 4.0).abs() < 1e-10);
        assert!((a[(1, 1)] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_ger_accumulate() {
        let x = [1.0f64, 1.0];
        let y = [1.0f64, 1.0];
        let mut a = Mat::filled(2, 2, 10.0);

        ger(1.0, &x, &y, a.as_mut());

        // A = [[10,10], [10,10]] + [[1,1], [1,1]] = [[11,11], [11,11]]
        for i in 0..2 {
            for j in 0..2 {
                assert!((a[(i, j)] - 11.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_ger_alpha_zero() {
        let x = [1.0f64, 2.0];
        let y = [1.0f64, 2.0];
        let mut a = Mat::filled(2, 2, 5.0);

        ger(0.0, &x, &y, a.as_mut());

        // A should be unchanged
        for i in 0..2 {
            for j in 0..2 {
                assert!((a[(i, j)] - 5.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_ger_f32() {
        let x = [1.0f32, 2.0];
        let y = [3.0f32, 4.0];
        let mut a = Mat::zeros(2, 2);

        ger(1.0, &x, &y, a.as_mut());

        assert!((a[(0, 0)] - 3.0).abs() < 1e-5);
        assert!((a[(0, 1)] - 4.0).abs() < 1e-5);
        assert!((a[(1, 0)] - 6.0).abs() < 1e-5);
        assert!((a[(1, 1)] - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_ger_single() {
        let x = [3.0f64];
        let y = [4.0f64];
        let mut a = Mat::zeros(1, 1);

        ger(2.0, &x, &y, a.as_mut());

        assert!((a[(0, 0)] - 24.0).abs() < 1e-10);
    }

    #[test]
    fn test_ger_tall() {
        let x = [1.0f64, 2.0, 3.0, 4.0];
        let y = [1.0f64];
        let mut a = Mat::zeros(4, 1);

        ger(1.0, &x, &y, a.as_mut());

        assert!((a[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((a[(1, 0)] - 2.0).abs() < 1e-10);
        assert!((a[(2, 0)] - 3.0).abs() < 1e-10);
        assert!((a[(3, 0)] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_ger_wide() {
        let x = [2.0f64];
        let y = [1.0f64, 2.0, 3.0, 4.0];
        let mut a = Mat::zeros(1, 4);

        ger(1.0, &x, &y, a.as_mut());

        assert!((a[(0, 0)] - 2.0).abs() < 1e-10);
        assert!((a[(0, 1)] - 4.0).abs() < 1e-10);
        assert!((a[(0, 2)] - 6.0).abs() < 1e-10);
        assert!((a[(0, 3)] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_gerc_real() {
        // For real types, gerc is the same as ger
        let x = [1.0f64, 2.0];
        let y = [3.0f64, 4.0];
        let mut a = Mat::zeros(2, 2);

        gerc(1.0, &x, &y, a.as_mut());

        assert!((a[(0, 0)] - 3.0).abs() < 1e-10);
        assert!((a[(0, 1)] - 4.0).abs() < 1e-10);
        assert!((a[(1, 0)] - 6.0).abs() < 1e-10);
        assert!((a[(1, 1)] - 8.0).abs() < 1e-10);
    }

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    fn bits(z: Complex64) -> (u64, u64) {
        (z.re.to_bits(), z.im.to_bits())
    }

    /// Issue #14: reference DGER / ZGERU skip column `j` when `y(j) == 0`, so an
    /// infinite `x(i)` cannot reach it as `x(i) * (alpha * 0) = NaN`, and form the
    /// update as `A(i,j) + x(i) * (alpha * y(j))`.
    #[test]
    fn test_issue_14_ger_zero_column_skip() {
        let inf = f64::INFINITY;
        // y(1) = 0 and y(2) = -0 (equal to zero, like Fortran's `.NE.ZERO`) skip
        // columns 1 and 2 although x(0) is infinite.
        let x = [inf, 1.0, 2.0];
        let y = [2.0, 0.0, -0.0];
        let mut a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);
        let before = a.clone();
        ger(1.5, &x, &y, a.as_mut());
        for j in 1..3 {
            for i in 0..3 {
                assert_eq!(a[(i, j)].to_bits(), before[(i, j)].to_bits(), "({i},{j})");
            }
        }
        // Column 0: temp = 1.5 * 2 = 3, so A00 = 1 + Inf * 3 = Inf.
        assert_eq!(a[(0, 0)], inf);
        assert_eq!(a[(1, 0)], 4.0 + 1.0 * 3.0);
        assert_eq!(a[(2, 0)], 7.0 + 2.0 * 3.0);

        // Reference association x(i) * (alpha * y(j)): here (alpha * x(i)) * y(j)
        // would overflow to Inf, while the reference product is exactly 2^600.
        let big = 2f64.powi(600);
        let mut a = Mat::filled(1, 1, 1.0f64);
        ger(big, &[big], &[1.0 / big], a.as_mut());
        assert_eq!(a[(0, 0)], 1.0 + big);

        // Complex (ZGERU): column 0 (y(0) == 0) is skipped and
        // A01 = 1 + x0 * (alpha * y1) = 1 + (Inf, 0) * (1, 0) = (Inf, NaN); the
        // (alpha * x0) * y1 association would give (NaN, NaN).
        let mut a = Mat::filled(2, 2, c(1.0, 0.0));
        let x = [c(inf, 0.0), c(1.0, 0.0)];
        let y = [c(0.0, 0.0), c(1.0, 0.0)];
        ger(c(1.0, 0.0), &x, &y, a.as_mut());
        assert_eq!(bits(a[(0, 0)]), bits(c(1.0, 0.0)));
        assert_eq!(bits(a[(1, 0)]), bits(c(1.0, 0.0)));
        assert_eq!(a[(0, 1)].re, inf);
        assert!(a[(0, 1)].im.is_nan());
        assert_eq!(bits(a[(1, 1)]), bits(c(2.0, 0.0)));
    }

    /// Issue #14: reference ZGERC skips column `j` when `y(j) == 0` and forms the
    /// update as `A(i,j) + x(i) * (alpha * conj(y(j)))`.
    #[test]
    fn test_issue_14_gerc_zero_column_skip() {
        let inf = f64::INFINITY;
        // y(0) = -0 + 0i compares equal to zero: column 0 is skipped although x(0)
        // is infinite.
        let x = [c(inf, 0.0), c(1.0, -1.0)];
        let y = [c(-0.0, 0.0), c(0.0, 1.0)];
        let rows = [[c(1.0, 0.5), c(2.0, -1.0)], [c(3.0, 0.25), c(4.0, 2.0)]];
        let mut a = Mat::from_rows(&[&rows[0], &rows[1]]);
        gerc(c(1.0, 0.0), &x, &y, a.as_mut());
        assert_eq!(bits(a[(0, 0)]), bits(rows[0][0]));
        assert_eq!(bits(a[(1, 0)]), bits(rows[1][0]));
        // Column 1: temp = (1, 0) * conj(i) = (0, -1). A01 = (2, -1) + (Inf, 0) * (0, -1)
        // = (2, -1) + (NaN, -Inf): the reference association keeps the imaginary part
        // at -Inf, where (alpha * x0) * conj(y1) would give (NaN, NaN).
        assert!(a[(0, 1)].re.is_nan());
        assert_eq!(a[(0, 1)].im, -inf);
        // A11 = (4, 2) + (1, -1) * (0, -1) = (4, 2) + (-1, -1).
        assert_eq!(bits(a[(1, 1)]), bits(c(3.0, 1.0)));
    }
}
