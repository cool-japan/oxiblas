//! Plane rotation (Givens rotation) operations.
//!
//! This module provides:
//! - [`rotg`]: Generate a Givens plane rotation
//! - [`rot`]: Apply a plane rotation
//! - [`rotmg`]: Generate a modified Givens rotation
//! - [`rotm`]: Apply a modified Givens rotation

use num_traits::Float;

/// Result of the ROTG operation (Givens rotation).
#[derive(Debug, Clone, Copy)]
pub struct RotgResult<T> {
    /// The rotated value r = √(a² + b²)
    pub r: T,
    /// Cosine of the rotation angle
    pub c: T,
    /// Sine of the rotation angle
    pub s: T,
    /// The z parameter for reconstruction
    pub z: T,
}

/// Generates a plane rotation (Givens rotation).
///
/// Given scalars a and b, computes the cosine c and sine s of a rotation
/// that transforms (a, b) to (r, 0) where r = ±√(a² + b²).
///
/// The rotation satisfies:
/// ```text
/// [ c  s ] [ a ]   [ r ]
/// [-s  c ] [ b ] = [ 0 ]
/// ```
///
/// # Algorithm
///
/// Uses the stable BLAS algorithm:
/// - If |b| > |a|, compute tan(θ) = a/b, then c and s
/// - Otherwise compute cot(θ) = b/a, then c and s
/// - Special handling for a = 0 and b = 0
///
/// The z parameter encodes either c or 1/s for reconstruction:
/// - z = 0 if a = b = 0
/// - z = 1 if |b| > |a| (c = 1/z, s = √(1 - c²))
/// - z = s if |b| ≤ |a| (c = √(1 - z²), s = z)
///
/// # Example
///
/// ```
/// use oxiblas_blas::level1::rotg;
///
/// let result = rotg(3.0f64, 4.0);
/// // r = 5.0, c = 0.6, s = 0.8
/// assert!((result.r - 5.0).abs() < 1e-10);
/// assert!((result.c - 0.6).abs() < 1e-10);
/// assert!((result.s - 0.8).abs() < 1e-10);
/// ```
pub fn rotg<T: Float>(a: T, b: T) -> RotgResult<T> {
    let zero = T::zero();
    let one = T::one();

    // Handle the case where both are zero
    if a.is_zero() && b.is_zero() {
        return RotgResult {
            r: zero,
            c: one,
            s: zero,
            z: zero,
        };
    }

    let abs_a = a.abs();
    let abs_b = b.abs();
    let scale = abs_a + abs_b;

    // Compute |r| = scale * sqrt((a/scale)^2 + (b/scale)^2)
    let r_magnitude = scale * ((a / scale).powi(2) + (b / scale).powi(2)).sqrt();

    // Determine sign of r based on larger magnitude input
    let r = if abs_a > abs_b {
        r_magnitude.copysign(a)
    } else {
        r_magnitude.copysign(b)
    };

    let c = a / r;
    let s = b / r;

    // Compute z for reconstruction
    let z = if abs_a > abs_b {
        s
    } else if c != zero {
        one / c
    } else {
        one
    };

    RotgResult { r, c, s, z }
}

/// Applies a plane rotation to vectors x and y.
///
/// For each element i:
/// ```text
/// [ x[i] ]     [ c  s ] [ x[i] ]
/// [ y[i] ]  =  [-s  c ] [ y[i] ]
/// ```
///
/// This is equivalent to:
/// ```text
/// temp  = c * x[i] + s * y[i]
/// y[i] = c * y[i] - s * x[i]
/// x[i] = temp
/// ```
///
/// # Arguments
///
/// * `c` - Cosine of the rotation angle
/// * `s` - Sine of the rotation angle
/// * `x` - First vector (modified in place)
/// * `y` - Second vector (modified in place)
///
/// # Panics
///
/// Panics if x and y have different lengths.
///
/// # Example
///
/// ```
/// use oxiblas_blas::level1::rot;
///
/// let mut x = [1.0f64, 0.0];
/// let mut y = [0.0f64, 1.0];
/// let c = 0.6;
/// let s = 0.8;
///
/// rot(c, s, &mut x, &mut y);
/// // After rotation: x = [0.6, 0.8], y = [-0.8, 0.6]
/// ```
pub fn rot<T: Float>(c: T, s: T, x: &mut [T], y: &mut [T]) {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    for i in 0..x.len() {
        let xi = x[i];
        let yi = y[i];
        x[i] = c * xi + s * yi;
        y[i] = c * yi - s * xi;
    }
}

/// Parameters for the modified Givens rotation.
///
/// The flag indicates which form the H matrix takes:
/// - `flag = -1.0`: H = [[h11, h12], [h21, h22]]
/// - `flag = 0.0`: H = [[1.0, h12], [h21, 1.0]]
/// - `flag = 1.0`: H = [[h11, 1.0], [-1.0, h22]]
/// - `flag = -2.0`: H = I (identity)
#[derive(Debug, Clone, Copy)]
pub struct RotmParams<T> {
    /// Flag indicating the form of H (-2, -1, 0, or 1)
    pub flag: T,
    /// H\[0,0\] element (used when flag = -1 or 1)
    pub h11: T,
    /// H\[0,1\] element (used when flag = -1 or 0)
    pub h12: T,
    /// H\[1,0\] element (used when flag = -1 or 0)
    pub h21: T,
    /// H\[1,1\] element (used when flag = -1 or 1)
    pub h22: T,
}

impl<T: Float> RotmParams<T> {
    /// Creates identity rotation parameters.
    #[must_use]
    pub fn identity() -> Self {
        let zero = T::zero();
        let one = T::one();
        let minus_two = -(one + one);
        Self {
            flag: minus_two,
            h11: zero,
            h12: zero,
            h21: zero,
            h22: zero,
        }
    }
}

/// Generates parameters for a modified Givens rotation.
///
/// This is a more numerically stable version of ROTG for scaled vectors.
/// Given scalars d1, d2, x1, y1, computes the transformation:
///
/// ```text
/// [ x1 ]     [ h11  h12 ] [ x1 * sqrt(d1) ]
/// [ 0  ]  =  [ h21  h22 ] [ y1 * sqrt(d2) ]
/// ```
///
/// The modified rotation avoids explicit computation of c and s,
/// instead working with scaled quantities.
///
/// # Arguments
///
/// * `d1` - Scaling factor for x1 (modified in place)
/// * `d2` - Scaling factor for y1 (modified in place)
/// * `x1` - First element of vector (modified in place)
/// * `y1` - Second element of vector
///
/// # Returns
///
/// Parameters for the modified rotation matrix H.
///
/// # Example
///
/// ```
/// use oxiblas_blas::level1::rotmg;
///
/// let mut d1 = 1.0f64;
/// let mut d2 = 1.0f64;
/// let mut x1 = 3.0f64;
/// let y1 = 4.0f64;
///
/// let params = rotmg(&mut d1, &mut d2, &mut x1, y1);
/// ```
pub fn rotmg<T: Float>(d1: &mut T, d2: &mut T, x1: &mut T, y1: T) -> RotmParams<T> {
    let zero = T::zero();
    let one = T::one();
    let _two = one + one;
    let minus_one = -one;

    // GAM and GAMSQ are the machine-dependent constants
    // GAM = 2^(base/2) where base is the number of bits in the mantissa
    // For simplicity, use reasonable constants
    let gam: T = T::from(4096.0).unwrap(); // 2^12
    let gamsq: T = gam * gam;
    let rgamsq: T = one / gamsq;

    // Handle special cases
    if *d1 < zero {
        // Set flag = -1 and all parameters to zero, set d1 to zero
        *d1 = zero;
        *d2 = zero;
        *x1 = zero;
        return RotmParams {
            flag: minus_one,
            h11: zero,
            h12: zero,
            h21: zero,
            h22: zero,
        };
    }

    let p2 = *d2 * y1;

    if p2.is_zero() {
        return RotmParams::identity();
    }

    let p1 = *d1 * *x1;
    let q2 = p2 * y1;
    let q1 = p1 * *x1;

    let abs_q1 = q1.abs();
    let abs_q2 = q2.abs();

    let (mut flag, mut h11, mut h12, mut h21, mut h22);

    if abs_q1 > abs_q2 {
        h21 = -y1 / *x1;
        h12 = p2 / p1;

        let u = one - h12 * h21;

        if u > zero {
            flag = zero;
            *d1 = *d1 / u;
            *d2 = *d2 / u;
            *x1 = *x1 * u;
            h11 = one;
            h22 = one;
        } else {
            // Rescaling needed
            flag = minus_one;
            h11 = zero;
            h12 = zero;
            h21 = zero;
            h22 = zero;
            *d1 = zero;
            *d2 = zero;
            *x1 = zero;
        }
    } else if q2 < zero {
        flag = minus_one;
        h11 = zero;
        h12 = zero;
        h21 = zero;
        h22 = zero;
        *d1 = zero;
        *d2 = zero;
        *x1 = zero;
    } else {
        flag = one;
        h11 = p1 / p2;
        h22 = *x1 / y1;
        let u = one + h11 * h22;
        let temp = *d2 / u;
        *d2 = *d1 / u;
        *d1 = temp;
        *x1 = y1 * u;
        h12 = one;
        h21 = minus_one;
    }

    // Scale d1 if necessary
    if *d1 != zero {
        while *d1 <= rgamsq || *d1 >= gamsq {
            if flag == zero {
                h11 = one;
                h22 = one;
                flag = minus_one;
            } else {
                h21 = minus_one;
                h12 = one;
                flag = minus_one;
            }
            if *d1 <= rgamsq {
                *d1 = *d1 * gamsq;
                *x1 = *x1 / gam;
                h11 = h11 / gam;
                h12 = h12 / gam;
            } else {
                *d1 = *d1 / gamsq;
                *x1 = *x1 * gam;
                h11 = h11 * gam;
                h12 = h12 * gam;
            }
        }
    }

    // Scale d2 if necessary
    if *d2 != zero {
        while d2.abs() <= rgamsq || d2.abs() >= gamsq {
            if flag == zero {
                h11 = one;
                h22 = one;
                flag = minus_one;
            } else {
                h21 = minus_one;
                h12 = one;
                flag = minus_one;
            }
            if d2.abs() <= rgamsq {
                *d2 = *d2 * gamsq;
                h21 = h21 / gam;
                h22 = h22 / gam;
            } else {
                *d2 = *d2 / gamsq;
                h21 = h21 * gam;
                h22 = h22 * gam;
            }
        }
    }

    RotmParams {
        flag,
        h11,
        h12,
        h21,
        h22,
    }
}

/// Applies a modified Givens rotation to vectors x and y.
///
/// Applies the rotation defined by the parameters from [`rotmg`].
///
/// The form of the rotation matrix H depends on the flag:
/// - `flag = -1.0`: H = [[h11, h12], [h21, h22]]
/// - `flag = 0.0`: H = [[1.0, h12], [h21, 1.0]]
/// - `flag = 1.0`: H = [[h11, 1.0], [-1.0, h22]]
/// - `flag = -2.0`: H = I (identity, no operation)
///
/// For each element i:
/// ```text
/// [ x[i] ]     [ h11  h12 ] [ x[i] ]
/// [ y[i] ]  =  [ h21  h22 ] [ y[i] ]
/// ```
///
/// # Arguments
///
/// * `params` - Parameters from [`rotmg`]
/// * `x` - First vector (modified in place)
/// * `y` - Second vector (modified in place)
///
/// # Panics
///
/// Panics if x and y have different lengths.
///
/// # Example
///
/// ```
/// use oxiblas_blas::level1::{rotmg, rotm, RotmParams};
///
/// let mut d1 = 1.0f64;
/// let mut d2 = 1.0f64;
/// let mut x1 = 3.0f64;
/// let y1 = 4.0f64;
///
/// let params = rotmg(&mut d1, &mut d2, &mut x1, y1);
///
/// let mut x = [1.0, 2.0, 3.0];
/// let mut y = [4.0, 5.0, 6.0];
/// rotm(&params, &mut x, &mut y);
/// ```
pub fn rotm<T: Float>(params: &RotmParams<T>, x: &mut [T], y: &mut [T]) {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    let zero = T::zero();
    let one = T::one();
    let two = one + one;
    let minus_one = -one;
    let minus_two = -two;

    // Check flag to determine operation
    if params.flag == minus_two {
        // Identity - no operation needed
        return;
    }

    let n = x.len();

    if params.flag == minus_one {
        // H = [[h11, h12], [h21, h22]]
        for i in 0..n {
            let xi = x[i];
            let yi = y[i];
            x[i] = params.h11 * xi + params.h12 * yi;
            y[i] = params.h21 * xi + params.h22 * yi;
        }
    } else if params.flag == zero {
        // H = [[1, h12], [h21, 1]]
        for i in 0..n {
            let xi = x[i];
            let yi = y[i];
            x[i] = xi + params.h12 * yi;
            y[i] = params.h21 * xi + yi;
        }
    } else if params.flag == one {
        // H = [[h11, 1], [-1, h22]]
        for i in 0..n {
            let xi = x[i];
            let yi = y[i];
            x[i] = params.h11 * xi + yi;
            y[i] = -xi + params.h22 * yi;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotg_basic() {
        // Test with (3, 4) -> r = 5, c = 0.6, s = 0.8
        let result = rotg(3.0f64, 4.0);
        assert!((result.r - 5.0).abs() < 1e-10);
        assert!((result.c - 0.6).abs() < 1e-10);
        assert!((result.s - 0.8).abs() < 1e-10);

        // Verify the rotation: c*a + s*b = r, -s*a + c*b = 0
        assert!((result.c * 3.0 + result.s * 4.0 - result.r).abs() < 1e-10);
        assert!((-result.s * 3.0 + result.c * 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotg_zero() {
        let result = rotg(0.0f64, 0.0);
        assert_eq!(result.r, 0.0);
        assert_eq!(result.c, 1.0);
        assert_eq!(result.s, 0.0);
    }

    #[test]
    fn test_rotg_a_zero() {
        let result = rotg(0.0f64, 3.0);
        assert!((result.r.abs() - 3.0).abs() < 1e-10);
        assert!(result.c.abs() < 1e-10);
        assert!((result.s.abs() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotg_b_zero() {
        let result = rotg(3.0f64, 0.0);
        assert!((result.r - 3.0).abs() < 1e-10);
        assert!((result.c - 1.0).abs() < 1e-10);
        assert!(result.s.abs() < 1e-10);
    }

    #[test]
    fn test_rotg_negative() {
        let result = rotg(-3.0f64, 4.0);
        // r has sign of larger magnitude input (b in this case)
        assert!((result.r.abs() - 5.0).abs() < 1e-10);
        // Verify rotation property
        assert!((result.c * (-3.0) + result.s * 4.0 - result.r).abs() < 1e-10);
        assert!((-result.s * (-3.0) + result.c * 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_rot_basic() {
        let mut x = [3.0f64, 0.0];
        let mut y = [4.0f64, 5.0];
        let c = 0.6;
        let s = 0.8;

        rot(c, s, &mut x, &mut y);

        // x[0] = c*3 + s*4 = 1.8 + 3.2 = 5.0
        // y[0] = c*4 - s*3 = 2.4 - 2.4 = 0.0
        assert!((x[0] - 5.0).abs() < 1e-10);
        assert!(y[0].abs() < 1e-10);

        // x[1] = c*0 + s*5 = 4.0
        // y[1] = c*5 - s*0 = 3.0
        assert!((x[1] - 4.0).abs() < 1e-10);
        assert!((y[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_rot_identity() {
        let mut x = [1.0, 2.0, 3.0];
        let mut y = [4.0, 5.0, 6.0];
        let x_orig = x;
        let y_orig = y;

        // c = 1, s = 0 is identity
        rot(1.0, 0.0, &mut x, &mut y);

        for i in 0..3 {
            assert!((x[i] - x_orig[i]).abs() < 1e-10);
            assert!((y[i] - y_orig[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_rot_90_degrees() {
        let mut x = [1.0f64, 0.0];
        let mut y = [0.0f64, 1.0];

        // 90 degree rotation: c = 0, s = 1
        rot(0.0, 1.0, &mut x, &mut y);

        // [0, 1][1, 0] = [0, 1]
        // [-1, 0][0, 1] = [-1, 0]
        assert!((x[0] - 0.0).abs() < 1e-10);
        assert!((x[1] - 1.0).abs() < 1e-10);
        assert!((y[0] - (-1.0)).abs() < 1e-10);
        assert!((y[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotg_and_rot_combined() {
        // Generate rotation to eliminate 4 from (3, 4)
        let result = rotg(3.0f64, 4.0);

        // Apply to the same vector
        let mut x = [3.0f64];
        let mut y = [4.0f64];
        rot(result.c, result.s, &mut x, &mut y);

        // After rotation, x should be r=5, y should be 0
        assert!((x[0] - result.r).abs() < 1e-10);
        assert!(y[0].abs() < 1e-10);
    }

    #[test]
    fn test_rotm_identity() {
        let params = RotmParams::<f64>::identity();
        let mut x = [1.0, 2.0, 3.0];
        let mut y = [4.0, 5.0, 6.0];
        let x_orig = x;
        let y_orig = y;

        rotm(&params, &mut x, &mut y);

        for i in 0..3 {
            assert!((x[i] - x_orig[i]).abs() < 1e-10);
            assert!((y[i] - y_orig[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_rotmg_basic() {
        let mut d1 = 1.0f64;
        let mut d2 = 1.0f64;
        let mut x1 = 3.0f64;
        let y1 = 4.0f64;

        let _params = rotmg(&mut d1, &mut d2, &mut x1, y1);

        // The result should satisfy certain properties
        // d1 and d2 should remain positive (or zero if degenerate)
        assert!(d1 >= 0.0);
    }

    #[test]
    fn test_rotmg_zero_y() {
        let mut d1 = 1.0f64;
        let mut d2 = 1.0f64;
        let mut x1 = 3.0f64;
        let y1 = 0.0f64;

        let params = rotmg(&mut d1, &mut d2, &mut x1, y1);

        // Should return identity (flag = -2)
        assert!((params.flag - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_rotm_flag_minus_one() {
        // Test with explicit H matrix
        let params = RotmParams {
            flag: -1.0,
            h11: 0.6,
            h12: 0.8,
            h21: -0.8,
            h22: 0.6,
        };

        let mut x = [1.0f64, 0.0];
        let mut y = [0.0f64, 1.0];

        rotm(&params, &mut x, &mut y);

        // x = 0.6*1 + 0.8*0 = 0.6
        // y = -0.8*1 + 0.6*0 = -0.8
        assert!((x[0] - 0.6).abs() < 1e-10);
        assert!((y[0] - (-0.8)).abs() < 1e-10);

        // x = 0.6*0 + 0.8*1 = 0.8
        // y = -0.8*0 + 0.6*1 = 0.6
        assert!((x[1] - 0.8).abs() < 1e-10);
        assert!((y[1] - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_rotm_flag_zero() {
        // H = [[1, h12], [h21, 1]]
        let params = RotmParams {
            flag: 0.0,
            h11: 0.0, // unused
            h12: 2.0,
            h21: 3.0,
            h22: 0.0, // unused
        };

        let mut x = [1.0f64];
        let mut y = [1.0f64];

        rotm(&params, &mut x, &mut y);

        // x = 1*1 + 2*1 = 3
        // y = 3*1 + 1*1 = 4
        assert!((x[0] - 3.0).abs() < 1e-10);
        assert!((y[0] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotm_flag_one() {
        // H = [[h11, 1], [-1, h22]]
        let params = RotmParams {
            flag: 1.0,
            h11: 2.0,
            h12: 0.0, // unused (implicitly 1)
            h21: 0.0, // unused (implicitly -1)
            h22: 3.0,
        };

        let mut x = [1.0f64];
        let mut y = [1.0f64];

        rotm(&params, &mut x, &mut y);

        // x = 2*1 + 1*1 = 3
        // y = -1*1 + 3*1 = 2
        assert!((x[0] - 3.0).abs() < 1e-10);
        assert!((y[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_rot_f32() {
        let mut x = [3.0f32, 0.0];
        let mut y = [4.0f32, 5.0];
        let c = 0.6f32;
        let s = 0.8f32;

        rot(c, s, &mut x, &mut y);

        assert!((x[0] - 5.0).abs() < 1e-5);
        assert!(y[0].abs() < 1e-5);
    }
}
