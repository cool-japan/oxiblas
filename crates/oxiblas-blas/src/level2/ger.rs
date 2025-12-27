//! GER: Rank-1 update.
//!
//! Computes A = α·x·y^T + A

use oxiblas_core::scalar::Field;
use oxiblas_matrix::MatMut;

/// Performs the rank-1 update: A = α·x·y^T + A
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

    if alpha == T::zero() {
        return;
    }

    // Compute A += α * x * y^T
    for i in 0..m {
        let alpha_xi = alpha * x[i];
        for j in 0..n {
            a[(i, j)] += alpha_xi * y[j];
        }
    }
}

/// Performs the conjugated rank-1 update: A = α·x·conj(y)^T + A
///
/// For real types, this is identical to ger.
/// For complex types, y is conjugated.
pub fn gerc<T: Field>(alpha: T, x: &[T], y: &[T], mut a: MatMut<'_, T>) {
    let m = a.nrows();
    let n = a.ncols();

    assert_eq!(x.len(), m, "x length must match number of rows");
    assert_eq!(y.len(), n, "y length must match number of columns");

    if alpha == T::zero() {
        return;
    }

    // Compute A += α * x * conj(y)^T
    for i in 0..m {
        let alpha_xi = alpha * x[i];
        for j in 0..n {
            a[(i, j)] += alpha_xi * y[j].conj();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
