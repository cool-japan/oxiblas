//! SCAL: x = α·x
//!
//! Scales a vector by a constant.

use oxiblas_core::scalar::Field;

/// Scales a vector by a constant: x = α·x
///
/// # Example
///
/// ```
/// use oxiblas_blas::level1::scal;
///
/// let mut x = [1.0f64, 2.0, 3.0];
/// scal(2.0, &mut x);
///
/// assert!((x[0] - 2.0).abs() < 1e-10);
/// assert!((x[1] - 4.0).abs() < 1e-10);
/// assert!((x[2] - 6.0).abs() < 1e-10);
/// ```
pub fn scal<T: Field>(alpha: T, x: &mut [T]) {
    let n = x.len();
    if n == 0 {
        return;
    }

    // Fast path for common cases
    if alpha == T::zero() {
        for xi in x.iter_mut() {
            *xi = T::zero();
        }
        return;
    }

    if alpha == T::one() {
        return;
    }

    // Unroll by 4 for better pipelining
    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 4;
        x[base] = alpha * x[base];
        x[base + 1] = alpha * x[base + 1];
        x[base + 2] = alpha * x[base + 2];
        x[base + 3] = alpha * x[base + 3];
    }

    // Handle remainder
    let base = chunks * 4;
    for i in 0..remainder {
        x[base + i] = alpha * x[base + i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scal_basic() {
        let mut x = [1.0, 2.0, 3.0, 4.0];
        scal(2.0, &mut x);
        assert_eq!(x, [2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_scal_zero() {
        let mut x = [1.0, 2.0, 3.0];
        scal(0.0, &mut x);
        assert_eq!(x, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_scal_one() {
        let mut x = [1.0, 2.0, 3.0];
        let x_orig = x;
        scal(1.0, &mut x);
        assert_eq!(x, x_orig);
    }

    #[test]
    fn test_scal_negative() {
        let mut x = [1.0, 2.0, 3.0];
        scal(-1.0, &mut x);
        assert_eq!(x, [-1.0, -2.0, -3.0]);
    }

    #[test]
    fn test_scal_f32() {
        let mut x = [1.0f32, 2.0, 3.0];
        scal(0.5f32, &mut x);
        assert!((x[0] - 0.5).abs() < 1e-5);
        assert!((x[1] - 1.0).abs() < 1e-5);
        assert!((x[2] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_scal_odd_length() {
        let mut x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        scal(3.0, &mut x);
        for i in 0..7 {
            assert!((x[i] - 3.0 * (i + 1) as f64).abs() < 1e-10);
        }
    }
}
