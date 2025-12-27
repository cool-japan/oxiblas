//! Extended precision dot product operations.
//!
//! Provides dot products with extended precision accumulation for improved
//! numerical accuracy, similar to BLAS sdsdot and dsdot routines.

/// Dot product of single precision vectors with double precision accumulation.
///
/// Computes: result = α + Σ(x\[i\] * y\[i\]) in double precision, then converts to f32.
///
/// This provides better accuracy than standard single precision dot product
/// by accumulating in double precision internally.
///
/// # Arguments
///
/// * `alpha` - Initial value to add to the dot product
/// * `x` - First input vector
/// * `y` - Second input vector
///
/// # Panics
///
/// Panics if `x` and `y` have different lengths.
///
/// # Example
///
/// ```
/// use oxiblas_blas::level1::sdsdot;
///
/// let x = [1.0f32, 2.0, 3.0];
/// let y = [4.0f32, 5.0, 6.0];
///
/// // result = 0.0 + (1*4 + 2*5 + 3*6) = 32.0
/// let result = sdsdot(0.0, &x, &y);
/// assert!((result - 32.0).abs() < 1e-5);
///
/// // With alpha
/// let result_alpha = sdsdot(10.0, &x, &y);
/// assert!((result_alpha - 42.0).abs() < 1e-5);
/// ```
#[inline]
#[must_use]
pub fn sdsdot(alpha: f32, x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    let mut sum = f64::from(alpha);

    // Accumulate in double precision for better accuracy
    for i in 0..x.len() {
        sum += f64::from(x[i]) * f64::from(y[i]);
    }

    sum as f32
}

/// Dot product of single precision vectors with double precision result.
///
/// Computes: result = Σ(x\[i\] * y\[i\]) in double precision.
///
/// This is useful when you need the full double precision result from
/// single precision inputs.
///
/// # Arguments
///
/// * `x` - First input vector
/// * `y` - Second input vector
///
/// # Panics
///
/// Panics if `x` and `y` have different lengths.
///
/// # Example
///
/// ```
/// use oxiblas_blas::level1::dsdot;
///
/// let x = [1.0f32, 2.0, 3.0];
/// let y = [4.0f32, 5.0, 6.0];
///
/// // result = 1*4 + 2*5 + 3*6 = 32.0 (in f64)
/// let result = dsdot(&x, &y);
/// assert!((result - 32.0).abs() < 1e-12);
/// ```
#[inline]
#[must_use]
pub fn dsdot(x: &[f32], y: &[f32]) -> f64 {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    let mut sum = 0.0f64;

    // Accumulate in double precision
    for i in 0..x.len() {
        sum += f64::from(x[i]) * f64::from(y[i]);
    }

    sum
}

/// Extended precision dot product using Kahan summation.
///
/// Computes the dot product with compensated summation to reduce
/// numerical error accumulation. This is more accurate than standard
/// dot product for very long vectors or when dealing with values
/// of vastly different magnitudes.
///
/// # Arguments
///
/// * `x` - First input vector
/// * `y` - Second input vector
///
/// # Panics
///
/// Panics if `x` and `y` have different lengths.
///
/// # Example
///
/// ```
/// use oxiblas_blas::level1::dot_kahan;
///
/// let x = [1.0f64, 2.0, 3.0];
/// let y = [4.0f64, 5.0, 6.0];
///
/// let result = dot_kahan(&x, &y);
/// assert!((result - 32.0).abs() < 1e-14);
/// ```
#[inline]
#[must_use]
pub fn dot_kahan(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    let mut sum = 0.0;
    let mut c = 0.0; // Compensation for lost low-order bits

    for i in 0..x.len() {
        let product = x[i] * y[i];
        let y_val = product - c;
        let t = sum + y_val;
        c = (t - sum) - y_val;
        sum = t;
    }

    sum
}

/// Pairwise summation dot product for improved numerical stability.
///
/// Uses a pairwise (recursive) summation strategy that reduces
/// rounding error from O(n) to O(log n) compared to naive summation.
///
/// This is faster than Kahan summation while still providing good accuracy.
///
/// # Arguments
///
/// * `x` - First input vector
/// * `y` - Second input vector
///
/// # Panics
///
/// Panics if `x` and `y` have different lengths.
///
/// # Example
///
/// ```
/// use oxiblas_blas::level1::dot_pairwise;
///
/// let x = [1.0f64, 2.0, 3.0, 4.0];
/// let y = [5.0f64, 6.0, 7.0, 8.0];
///
/// let result = dot_pairwise(&x, &y);
/// // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
/// assert!((result - 70.0).abs() < 1e-14);
/// ```
#[must_use]
pub fn dot_pairwise(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    fn pairwise_sum(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len();

        if n == 0 {
            return 0.0;
        }
        if n == 1 {
            return x[0] * y[0];
        }
        if n <= 128 {
            // Base case: simple accumulation for small arrays
            let mut sum = 0.0;
            for i in 0..n {
                sum += x[i] * y[i];
            }
            return sum;
        }

        // Recursive case: split and sum
        let mid = n / 2;
        pairwise_sum(&x[..mid], &y[..mid]) + pairwise_sum(&x[mid..], &y[mid..])
    }

    pairwise_sum(x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdsdot_basic() {
        let x = [1.0f32, 2.0, 3.0];
        let y = [4.0f32, 5.0, 6.0];

        let result = sdsdot(0.0, &x, &y);
        assert!((result - 32.0).abs() < 1e-5);

        let result_alpha = sdsdot(10.0, &x, &y);
        assert!((result_alpha - 42.0).abs() < 1e-5);
    }

    #[test]
    fn test_dsdot_basic() {
        let x = [1.0f32, 2.0, 3.0];
        let y = [4.0f32, 5.0, 6.0];

        let result = dsdot(&x, &y);
        assert!((result - 32.0).abs() < 1e-12);
    }

    #[test]
    fn test_sdsdot_accuracy() {
        // Test with many small numbers where precision matters
        let n = 10000;
        let x: Vec<f32> = (0..n).map(|i| 1e-4 * (i as f32)).collect();
        let y: Vec<f32> = (0..n).map(|i| 1e-4 * ((n - i) as f32)).collect();

        let result = sdsdot(0.0, &x, &y);

        // Should not be zero due to double precision accumulation
        assert!(result > 0.0);
    }

    #[test]
    fn test_dot_kahan_basic() {
        let x = [1.0, 2.0, 3.0];
        let y = [4.0, 5.0, 6.0];

        let result = dot_kahan(&x, &y);
        assert!((result - 32.0).abs() < 1e-14);
    }

    #[test]
    fn test_dot_kahan_accuracy() {
        // Test with values that cause catastrophic cancellation
        let n = 1000;
        let x: Vec<f64> = (0..n).map(|i| 1.0 + 1e-10 * (i as f64)).collect();
        let y: Vec<f64> = (0..n).map(|_| 1.0).collect();

        let result_kahan = dot_kahan(&x, &y);
        let expected = n as f64 + 1e-10 * (n * (n - 1) / 2) as f64;

        assert!((result_kahan - expected).abs() < 1e-8);
    }

    #[test]
    fn test_dot_pairwise_basic() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [5.0, 6.0, 7.0, 8.0];

        let result = dot_pairwise(&x, &y);
        assert!((result - 70.0).abs() < 1e-14);
    }

    #[test]
    fn test_dot_pairwise_large() {
        let n = 1000;
        let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        let result = dot_pairwise(&x, &y);

        // Sum of i^2 from 1 to n = n(n+1)(2n+1)/6
        let expected = (n * (n + 1) * (2 * n + 1) / 6) as f64;
        assert!((result - expected).abs() < 1e-8);
    }

    #[test]
    fn test_dot_pairwise_empty() {
        let x: Vec<f64> = vec![];
        let y: Vec<f64> = vec![];

        let result = dot_pairwise(&x, &y);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_extended_precision_comparison() {
        // Compare different methods on a challenging case
        let n = 1000;
        let x: Vec<f32> = (0..n).map(|i| 1e-3 * (i as f32)).collect();
        let y: Vec<f32> = (0..n).map(|i| 1e-3 * ((n - i) as f32)).collect();

        let result_sdsdot = sdsdot(0.0, &x, &y);
        let result_dsdot = dsdot(&x, &y);

        // Both should give similar results with extended precision
        // Relaxed tolerance due to f32 conversion in sdsdot
        assert!((result_sdsdot as f64 - result_dsdot).abs() / result_dsdot.abs() < 1e-3);
    }
}
