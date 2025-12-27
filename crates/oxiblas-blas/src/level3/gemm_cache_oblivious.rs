//! Cache-oblivious matrix multiplication
//!
//! Implements a recursive divide-and-conquer GEMM that automatically adapts
//! to any cache hierarchy without explicit cache size parameters.
//!
//! The algorithm recursively subdivides matrices until reaching a base case,
//! ensuring optimal data locality across all cache levels.
//!
//! Reference: Frigo, M., Leiserson, C. E., Prokop, H., & Ramachandran, S. (1999).
//! "Cache-oblivious algorithms". In FOCS.

use oxiblas_core::scalar::{Field, Real};
use oxiblas_matrix::{MatMut, MatRef};

/// Minimum size for recursion (base case threshold)
const BASE_CASE_SIZE: usize = 32;

/// Cache-oblivious matrix multiplication using recursive divide-and-conquer.
///
/// Automatically adapts to cache hierarchy without tuning parameters.
/// Works well for any matrix size but is most beneficial for large matrices
/// where cache effects dominate.
///
/// # Arguments
///
/// * `alpha` - Scalar multiplier for A*B
/// * `a` - Left matrix (m × k)
/// * `b` - Right matrix (k × n)
/// * `beta` - Scalar multiplier for C
/// * `c` - Output matrix (m × n), modified in place
///
/// # Example
///
/// ```
/// use oxiblas_matrix::Mat;
/// use oxiblas_blas::level3::gemm_cache_oblivious;
///
/// let a = Mat::from_slice(4, 4, &[1.0; 16]);
/// let b = Mat::from_slice(4, 4, &[2.0; 16]);
/// let mut c = Mat::zeros(4, 4);
///
/// gemm_cache_oblivious(1.0, a.as_ref(), b.as_ref(), 0.0, &mut c.as_mut());
/// ```
pub fn gemm_cache_oblivious<T>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: &mut MatMut<'_, T>,
) where
    T: Field + Real,
{
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

    assert_eq!(b.nrows(), k, "Inner dimensions must match");
    assert_eq!(c.nrows(), m, "Output rows must match");
    assert_eq!(c.ncols(), n, "Output cols must match");

    // Scale C by beta
    if beta == T::zero() {
        for i in 0..m {
            for j in 0..n {
                c[(i, j)] = T::zero();
            }
        }
    } else if beta != T::one() {
        for i in 0..m {
            for j in 0..n {
                c[(i, j)] *= beta;
            }
        }
    }

    // Recursive computation
    gemm_recursive(alpha, a, b, c, 0, m, 0, n, 0, k);
}

/// Recursive cache-oblivious GEMM helper
///
/// Computes C[i_start..i_end, j_start..j_end] += alpha * A[i_start..i_end, k_start..k_end] * B[k_start..k_end, j_start..j_end]
#[inline]
fn gemm_recursive<T>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    c: &mut MatMut<'_, T>,
    i_start: usize,
    i_end: usize,
    j_start: usize,
    j_end: usize,
    k_start: usize,
    k_end: usize,
) where
    T: Field + Real,
{
    let m = i_end - i_start;
    let n = j_end - j_start;
    let k = k_end - k_start;

    // Base case: use simple triple loop
    if m <= BASE_CASE_SIZE && n <= BASE_CASE_SIZE && k <= BASE_CASE_SIZE {
        for i in i_start..i_end {
            for j in j_start..j_end {
                let mut sum = T::zero();
                for l in k_start..k_end {
                    sum += a[(i, l)] * b[(l, j)];
                }
                c[(i, j)] += alpha * sum;
            }
        }
        return;
    }

    // Recursive case: subdivide along largest dimension
    if m >= n && m >= k {
        // Split along m dimension (rows of A and C)
        let i_mid = i_start + m / 2;

        gemm_recursive(
            alpha, a, b, c, i_start, i_mid, j_start, j_end, k_start, k_end,
        );
        gemm_recursive(alpha, a, b, c, i_mid, i_end, j_start, j_end, k_start, k_end);
    } else if n >= m && n >= k {
        // Split along n dimension (cols of B and C)
        let j_mid = j_start + n / 2;

        gemm_recursive(
            alpha, a, b, c, i_start, i_end, j_start, j_mid, k_start, k_end,
        );
        gemm_recursive(alpha, a, b, c, i_start, i_end, j_mid, j_end, k_start, k_end);
    } else {
        // Split along k dimension (cols of A, rows of B)
        let k_mid = k_start + k / 2;

        gemm_recursive(
            alpha, a, b, c, i_start, i_end, j_start, j_end, k_start, k_mid,
        );
        gemm_recursive(alpha, a, b, c, i_start, i_end, j_start, j_end, k_mid, k_end);
    }
}

/// Cache-oblivious GEMM with configurable base case threshold.
///
/// Allows tuning the recursion base case for different architectures.
pub fn gemm_cache_oblivious_with_threshold<T>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: &mut MatMut<'_, T>,
    threshold: usize,
) where
    T: Field + Real,
{
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

    assert_eq!(b.nrows(), k);
    assert_eq!(c.nrows(), m);
    assert_eq!(c.ncols(), n);

    // Scale C by beta
    if beta == T::zero() {
        for i in 0..m {
            for j in 0..n {
                c[(i, j)] = T::zero();
            }
        }
    } else if beta != T::one() {
        for i in 0..m {
            for j in 0..n {
                c[(i, j)] *= beta;
            }
        }
    }

    gemm_recursive_threshold(alpha, a, b, c, 0, m, 0, n, 0, k, threshold);
}

/// Recursive helper with custom threshold
#[inline]
fn gemm_recursive_threshold<T>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    c: &mut MatMut<'_, T>,
    i_start: usize,
    i_end: usize,
    j_start: usize,
    j_end: usize,
    k_start: usize,
    k_end: usize,
    threshold: usize,
) where
    T: Field + Real,
{
    let m = i_end - i_start;
    let n = j_end - j_start;
    let k = k_end - k_start;

    if m <= threshold && n <= threshold && k <= threshold {
        for i in i_start..i_end {
            for j in j_start..j_end {
                let mut sum = T::zero();
                for l in k_start..k_end {
                    sum += a[(i, l)] * b[(l, j)];
                }
                c[(i, j)] += alpha * sum;
            }
        }
        return;
    }

    // Split along largest dimension
    if m >= n && m >= k {
        let i_mid = i_start + m / 2;
        gemm_recursive_threshold(
            alpha, a, b, c, i_start, i_mid, j_start, j_end, k_start, k_end, threshold,
        );
        gemm_recursive_threshold(
            alpha, a, b, c, i_mid, i_end, j_start, j_end, k_start, k_end, threshold,
        );
    } else if n >= m && n >= k {
        let j_mid = j_start + n / 2;
        gemm_recursive_threshold(
            alpha, a, b, c, i_start, i_end, j_start, j_mid, k_start, k_end, threshold,
        );
        gemm_recursive_threshold(
            alpha, a, b, c, i_start, i_end, j_mid, j_end, k_start, k_end, threshold,
        );
    } else {
        let k_mid = k_start + k / 2;
        gemm_recursive_threshold(
            alpha, a, b, c, i_start, i_end, j_start, j_end, k_start, k_mid, threshold,
        );
        gemm_recursive_threshold(
            alpha, a, b, c, i_start, i_end, j_start, j_end, k_mid, k_end, threshold,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiblas_matrix::Mat;

    fn gemm_reference<T>(
        alpha: T,
        a: MatRef<'_, T>,
        b: MatRef<'_, T>,
        beta: T,
        c: &mut MatMut<'_, T>,
    ) where
        T: Field + Real,
    {
        let m = a.nrows();
        let k = a.ncols();
        let n = b.ncols();

        if beta == T::zero() {
            for i in 0..m {
                for j in 0..n {
                    c[(i, j)] = T::zero();
                }
            }
        } else if beta != T::one() {
            for i in 0..m {
                for j in 0..n {
                    c[(i, j)] *= beta;
                }
            }
        }

        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for l in 0..k {
                    sum += a[(i, l)] * b[(l, j)];
                }
                c[(i, j)] += alpha * sum;
            }
        }
    }

    #[test]
    fn test_cache_oblivious_square() {
        let n = 64;
        let mut a_data = vec![0.0; n * n];
        let mut b_data = vec![0.0; n * n];

        for i in 0..n {
            for j in 0..n {
                a_data[i * n + j] = (i + j) as f64 * 0.5;
                b_data[i * n + j] = ((i as i32 - j as i32).abs()) as f64 * 0.3;
            }
        }

        let a = Mat::from_slice(n, n, &a_data);
        let b = Mat::from_slice(n, n, &b_data);
        let mut c_co = Mat::zeros(n, n);
        let mut c_ref = Mat::zeros(n, n);

        gemm_cache_oblivious(1.0, a.as_ref(), b.as_ref(), 0.0, &mut c_co.as_mut());
        gemm_reference(1.0, a.as_ref(), b.as_ref(), 0.0, &mut c_ref.as_mut());

        for i in 0..n {
            for j in 0..n {
                let diff: f64 = c_co[(i, j)] - c_ref[(i, j)];
                assert!(
                    diff.abs() < 1e-9,
                    "Mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    c_co[(i, j)],
                    c_ref[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_cache_oblivious_rectangular() {
        let a = Mat::from_slice(8, 12, &(1..=96).map(|x| x as f64).collect::<Vec<_>>());
        let b = Mat::from_slice(
            12,
            10,
            &(1..=120).map(|x| x as f64 * 0.5).collect::<Vec<_>>(),
        );
        let mut c_co = Mat::zeros(8, 10);
        let mut c_ref = Mat::zeros(8, 10);

        gemm_cache_oblivious(1.0, a.as_ref(), b.as_ref(), 0.0, &mut c_co.as_mut());
        gemm_reference(1.0, a.as_ref(), b.as_ref(), 0.0, &mut c_ref.as_mut());

        for i in 0..8 {
            for j in 0..10 {
                let diff: f64 = c_co[(i, j)] - c_ref[(i, j)];
                assert!(diff.abs() < 1e-8, "Rectangular mismatch at ({}, {})", i, j);
            }
        }
    }

    #[test]
    fn test_cache_oblivious_with_beta() {
        let a = Mat::from_slice(
            4,
            4,
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        );
        let b = Mat::from_slice(
            4,
            4,
            &[
                16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0,
                1.0,
            ],
        );
        let mut c = Mat::from_slice(
            4,
            4,
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        );

        let c_original = c.clone();
        let mut c_ref = c_original.clone();

        gemm_cache_oblivious(2.0, a.as_ref(), b.as_ref(), 3.0, &mut c.as_mut());
        gemm_reference(2.0, a.as_ref(), b.as_ref(), 3.0, &mut c_ref.as_mut());

        for i in 0..4 {
            for j in 0..4 {
                let diff: f64 = c[(i, j)] - c_ref[(i, j)];
                assert!(diff.abs() < 1e-9, "Beta scaling failed at ({}, {})", i, j);
            }
        }
    }

    #[test]
    fn test_cache_oblivious_power_of_two() {
        // Test with power-of-2 dimensions (optimal for recursion)
        let n = 128;
        let a = Mat::filled(n, n, 2.0);
        let b = Mat::filled(n, n, 3.0);
        let mut c_co = Mat::zeros(n, n);
        let mut c_ref = Mat::zeros(n, n);

        gemm_cache_oblivious(1.0, a.as_ref(), b.as_ref(), 0.0, &mut c_co.as_mut());
        gemm_reference(1.0, a.as_ref(), b.as_ref(), 0.0, &mut c_ref.as_mut());

        // All entries should be 2.0 * 3.0 * n = 6n
        let expected = 6.0 * n as f64;
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (c_co[(i, j)] - expected).abs() < 1e-8,
                    "Power-of-2 test failed at ({}, {})",
                    i,
                    j
                );
                assert!(
                    (c_ref[(i, j)] - expected).abs() < 1e-8,
                    "Reference failed at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_cache_oblivious_custom_threshold() {
        let a = Mat::from_slice(16, 16, &(0..256).map(|x| x as f64).collect::<Vec<_>>());
        let b = Mat::from_slice(
            16,
            16,
            &(0..256).map(|x| (255 - x) as f64).collect::<Vec<_>>(),
        );
        let mut c_t16 = Mat::zeros(16, 16);
        let mut c_t64 = Mat::zeros(16, 16);
        let mut c_ref = Mat::zeros(16, 16);

        gemm_cache_oblivious_with_threshold(
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            &mut c_t16.as_mut(),
            16,
        );
        gemm_cache_oblivious_with_threshold(
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            &mut c_t64.as_mut(),
            64,
        );
        gemm_reference(1.0, a.as_ref(), b.as_ref(), 0.0, &mut c_ref.as_mut());

        for i in 0..16 {
            for j in 0..16 {
                assert!(
                    (c_t16[(i, j)] - c_ref[(i, j)]).abs() < 1e-8,
                    "Threshold 16 failed at ({}, {})",
                    i,
                    j
                );
                assert!(
                    (c_t64[(i, j)] - c_ref[(i, j)]).abs() < 1e-8,
                    "Threshold 64 failed at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_cache_oblivious_f32() {
        let n = 32;
        let a = Mat::filled(n, n, 1.5f32);
        let b = Mat::filled(n, n, 2.5f32);
        let mut c = Mat::zeros(n, n);

        gemm_cache_oblivious(1.0, a.as_ref(), b.as_ref(), 0.0, &mut c.as_mut());

        // All entries should be 1.5 * 2.5 * n = 3.75n
        let expected = 3.75 * n as f32;
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (c[(i, j)] - expected).abs() < 1e-4,
                    "f32 test failed at ({}, {}): got {}, expected {}",
                    i,
                    j,
                    c[(i, j)],
                    expected
                );
            }
        }
    }
}
