//! SCAL: x = α·x
//!
//! Scales a vector by a constant.

use oxiblas_core::scalar::Field;

/// Scales a vector by a constant: x = α·x
///
/// Matches reference BLAS `DSCAL`/`SSCAL`, which unconditionally computes
/// the literal product `x[i] * alpha` for every element -- including when
/// `alpha == 0`. In IEEE-754 arithmetic, `0 * NaN == NaN` and
/// `0 * Inf == NaN`, so a vector containing `NaN`/`Inf` is **not** scrubbed
/// to a clean zero by `scal(0, x)`; only finite entries become exactly
/// `0.0`.
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
///
/// ```
/// use oxiblas_blas::level1::scal;
///
/// // alpha == 0 does NOT scrub NaN/Inf -- it propagates per IEEE-754.
/// let mut x = [1.0f64, f64::NAN, f64::INFINITY];
/// scal(0.0, &mut x);
///
/// assert_eq!(x[0], 0.0);
/// assert!(x[1].is_nan());
/// assert!(x[2].is_nan());
/// ```
pub fn scal<T: Field>(alpha: T, x: &mut [T]) {
    let n = x.len();
    if n == 0 {
        return;
    }

    // Fast path: multiplying by 1 is an exact no-op for every IEEE-754
    // value (finite, NaN, or Inf), so skipping the loop is always
    // numerically identical to running it.
    if alpha == T::one() {
        return;
    }

    // NOTE: we deliberately do NOT special-case alpha == 0 with a memset.
    // Reference BLAS DSCAL/SSCAL always perform the literal multiply
    // `x[i] = alpha * x[i]`; for alpha == 0 that yields NaN whenever
    // x[i] is NaN or +/-Inf (per IEEE-754: 0*NaN = NaN, 0*Inf = NaN), not
    // a scrubbed zero. A memset fast path would silently corrupt that
    // semantics, so alpha == 0 falls through to the same multiply loop as
    // every other alpha value.

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

    #[test]
    fn test_scal_zero_preserves_nan_inf() {
        // Reference DSCAL always computes the literal product
        // `x[i] * alpha`. For alpha == 0, IEEE-754 says 0*NaN = NaN and
        // 0*Inf = NaN -- the result must NOT be scrubbed to a clean zero
        // by a memset fast path. This vector is 5 elements long so it
        // exercises both the unrolled chunk loop and the remainder loop.
        let mut x = [1.0f64, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -2.5];
        scal(0.0, &mut x);

        assert_eq!(x[0], 0.0, "finite * 0 must be exactly 0");
        assert!(x[1].is_nan(), "NaN * 0 must stay NaN, got {}", x[1]);
        assert!(x[2].is_nan(), "+Inf * 0 must become NaN, got {}", x[2]);
        assert!(x[3].is_nan(), "-Inf * 0 must become NaN, got {}", x[3]);
        assert_eq!(x[4], 0.0, "negative finite * 0 must be exactly 0");
    }

    #[test]
    fn test_scal_zero_f32_preserves_nan_inf() {
        let mut x = [f32::NAN, f32::INFINITY, 3.0f32];
        scal(0.0f32, &mut x);

        assert!(x[0].is_nan());
        assert!(x[1].is_nan());
        assert_eq!(x[2], 0.0);
    }
}
