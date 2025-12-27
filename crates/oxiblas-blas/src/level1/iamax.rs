//! IAMAX: Index of maximum absolute value
//!
//! Finds the index of the first element with maximum absolute value.

use oxiblas_core::scalar::Scalar;

/// Finds the index of the element with maximum absolute value.
///
/// Returns the index of the first element having the maximum |x\[i\]|.
/// Returns 0 for empty vectors.
///
/// # Example
///
/// ```
/// use oxiblas_blas::level1::iamax;
///
/// let x = [1.0, -5.0, 3.0, 2.0];
/// let idx = iamax(&x);
///
/// // |-5| = 5 is the maximum absolute value
/// assert_eq!(idx, 1);
/// ```
pub fn iamax<T: Scalar>(x: &[T]) -> usize {
    let n = x.len();
    if n == 0 {
        return 0;
    }

    let mut max_idx = 0;
    let mut max_val = Scalar::abs(x[0]);

    for (i, &xi) in x.iter().enumerate().skip(1) {
        let abs_xi = Scalar::abs(xi);
        if abs_xi > max_val {
            max_val = abs_xi;
            max_idx = i;
        }
    }

    max_idx
}

/// Finds the index of the element with minimum absolute value.
///
/// Returns the index of the first element having the minimum |x\[i\]|.
/// Returns 0 for empty vectors.
///
/// # Example
///
/// ```
/// use oxiblas_blas::level1::iamin;
///
/// let x = [5.0, -1.0, 3.0, 2.0];
/// let idx = iamin(&x);
///
/// // |-1| = 1 is the minimum absolute value
/// assert_eq!(idx, 1);
/// ```
pub fn iamin<T: Scalar>(x: &[T]) -> usize {
    let n = x.len();
    if n == 0 {
        return 0;
    }

    let mut min_idx = 0;
    let mut min_val = Scalar::abs(x[0]);

    for (i, &xi) in x.iter().enumerate().skip(1) {
        let abs_xi = Scalar::abs(xi);
        if abs_xi < min_val {
            min_val = abs_xi;
            min_idx = i;
        }
    }

    min_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iamax_basic() {
        let x = [1.0, -5.0, 3.0, 2.0];
        let idx = iamax(&x);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_iamax_first() {
        let x = [10.0, 5.0, 3.0, 2.0];
        let idx = iamax(&x);
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_iamax_last() {
        let x = [1.0, 2.0, 3.0, 10.0];
        let idx = iamax(&x);
        assert_eq!(idx, 3);
    }

    #[test]
    fn test_iamax_ties() {
        // When there are ties, return the first one
        let x = [5.0, -5.0, 3.0];
        let idx = iamax(&x);
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_iamax_empty() {
        let x: [f64; 0] = [];
        let idx = iamax(&x);
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_iamax_single() {
        let x = [42.0];
        let idx = iamax(&x);
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_iamax_negative() {
        let x = [-10.0, -5.0, -3.0];
        let idx = iamax(&x);
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_iamax_f32() {
        let x = [1.0f32, -5.0, 3.0];
        let idx = iamax(&x);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_iamin_basic() {
        let x = [5.0, -1.0, 3.0, 2.0];
        let idx = iamin(&x);
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_iamin_zeros() {
        let x = [5.0, 0.0, 3.0];
        let idx = iamin(&x);
        assert_eq!(idx, 1);
    }
}
