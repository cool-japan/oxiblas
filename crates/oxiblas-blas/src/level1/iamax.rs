//! IAMAX: Index of maximum absolute value
//!
//! Finds the index of the first element with maximum absolute value.
//!
//! For complex scalar types, the comparison uses the BLAS `cabs1` metric
//! (`|Re(z)| + |Im(z)|`), matching reference `ICAMAX`/`IZAMAX` exactly --
//! **not** the complex modulus `sqrt(re^2 + im^2)`. For real types this
//! reduces to the ordinary `|x[i]|`, so the metric change is invisible
//! there.

use oxiblas_core::scalar::Scalar;

use super::asum::cabs1;

/// Finds the index of the element with maximum absolute value.
///
/// Returns the index of the first element having the maximum |x\[i\]|.
/// Returns 0 for empty vectors.
///
/// For complex scalar types, elements are compared using `cabs1(z) =
/// |Re(z)| + |Im(z)|`, matching reference BLAS `ICAMAX`/`IZAMAX` exactly
/// (this is **not** the complex modulus). For real types this reduces to
/// the ordinary `|x[i]|`.
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
    let mut max_val = cabs1(x[0]);

    for (i, &xi) in x.iter().enumerate().skip(1) {
        let abs_xi = cabs1(xi);
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
/// For complex scalar types, elements are compared using the same
/// `cabs1(z) = |Re(z)| + |Im(z)|` metric as [`iamax`], for consistency
/// with the BLAS `ICAMAX`/`IZAMAX` convention. For real types this reduces
/// to the ordinary `|x[i]|`.
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
    let mut min_val = cabs1(x[0]);

    for (i, &xi) in x.iter().enumerate().skip(1) {
        let abs_xi = cabs1(xi);
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

    // =========================================================================
    // Complex iamax/iamin -- cabs1 metric (|Re|+|Im|), not the modulus
    // =========================================================================

    #[test]
    fn test_iamax_complex_cabs1_counterexample() {
        // Genuine counterexample where cabs1 and the complex modulus
        // disagree on which element is "larger":
        //
        //   z0 = 3+0i  -> cabs1 = |3|+|0| = 3      modulus = sqrt(9)     = 3.0
        //   z1 = 2+2i  -> cabs1 = |2|+|2| = 4      modulus = sqrt(8)     = 2.8284...
        //
        // Under the (incorrect) modulus metric, z0 (index 0) would win
        // (3.0 > 2.8284). Under the correct BLAS cabs1 metric, z1 (index
        // 1) wins (4 > 3). This distinguishes the two implementations.
        use oxiblas_core::scalar::c64;

        let x = [c64(3.0, 0.0), c64(2.0, 2.0)];
        assert_eq!(
            iamax(&x),
            1,
            "iamax must use cabs1 (|re|+|im|), not the complex modulus"
        );
    }

    #[test]
    fn test_iamax_complex_hand_verified() {
        use oxiblas_core::scalar::c64;

        // cabs1: (1+2)=3, (3+4)=7, (0+1)=1, (5+0)=5 -> max is index 1
        let x = [c64(1.0, 2.0), c64(-3.0, 4.0), c64(0.0, -1.0), c64(5.0, 0.0)];
        assert_eq!(iamax(&x), 1);
    }

    #[test]
    fn test_iamin_complex_cabs1() {
        use oxiblas_core::scalar::c64;

        // cabs1: (1+2)=3, (3+4)=7, (0+1)=1, (5+0)=5 -> min is index 2
        let x = [c64(1.0, 2.0), c64(-3.0, 4.0), c64(0.0, -1.0), c64(5.0, 0.0)];
        assert_eq!(iamin(&x), 2);
    }
}
