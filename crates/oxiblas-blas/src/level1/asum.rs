//! ASUM: Sum of absolute values (L1 norm)
//!
//! Computes ||x||_1 = Σ |x\[i\]|

use oxiblas_core::scalar::{Real, Scalar};

/// Computes the sum of absolute values (L1 norm) for real vectors.
///
/// ||x||_1 = Σ |x\[i\]|
///
/// # Example
///
/// ```
/// use oxiblas_blas::level1::asum;
///
/// let x = [1.0f64, -2.0, 3.0, -4.0];
/// let sum = asum(&x);
///
/// // |1| + |-2| + |3| + |-4| = 10
/// assert!((sum - 10.0).abs() < 1e-10);
/// ```
pub fn asum<T: Real>(x: &[T]) -> T {
    let n = x.len();
    if n == 0 {
        return T::zero();
    }

    // Use 4-way accumulation
    let mut acc0 = T::zero();
    let mut acc1 = T::zero();
    let mut acc2 = T::zero();
    let mut acc3 = T::zero();

    let chunks = n / 4;
    let remainder = n % 4;

    for i in 0..chunks {
        let base = i * 4;
        acc0 += Scalar::abs(x[base]);
        acc1 += Scalar::abs(x[base + 1]);
        acc2 += Scalar::abs(x[base + 2]);
        acc3 += Scalar::abs(x[base + 3]);
    }

    let base = chunks * 4;
    for i in 0..remainder {
        acc0 += Scalar::abs(x[base + i]);
    }

    (acc0 + acc1) + (acc2 + acc3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asum_basic() {
        let x = [1.0, -2.0, 3.0, -4.0];
        let sum = asum(&x);
        assert!((sum - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_asum_positive() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let sum = asum(&x);
        assert!((sum - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_asum_empty() {
        let x: [f64; 0] = [];
        let sum = asum(&x);
        assert_eq!(sum, 0.0);
    }

    #[test]
    fn test_asum_single() {
        let x = [-5.0];
        let sum = asum(&x);
        assert!((sum - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_asum_f32() {
        let x = [1.0f32, -2.0, 3.0];
        let sum = asum(&x);
        assert!((sum - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_asum_zeros() {
        let x = [0.0; 5];
        let sum = asum(&x);
        assert_eq!(sum, 0.0);
    }
}
