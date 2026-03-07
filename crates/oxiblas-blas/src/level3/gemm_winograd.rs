//! Winograd's algorithm for matrix multiplication
//!
//! Winograd's algorithm reduces the number of multiplications at the cost of
//! more additions. For large matrices, this can be beneficial on architectures
//! where multiplication is significantly more expensive than addition.
//!
//! The classic Winograd algorithm works on 2×2 blocks and reduces multiplications
//! from 8 to 7 per block (12.5% reduction).
//!
//! For matrix multiplication C = A * B, the algorithm computes:
//! - Row factors: r\[i\] = A\[i,0\] * A\[i,1\] (for each row of A)
//! - Column factors: c\[j\] = B\[0,j\] * B\[1,j\] (for each column of B)
//! - Then C\[i,j\] = -r\[i\] - c\[j\] + (A\[i,0\] + B\[1,j\]) * (A\[i,1\] + B\[0,j\])

use oxiblas_core::scalar::{Field, Real};
use oxiblas_matrix::{MatMut, MatRef};

/// Winograd matrix multiplication for square matrices with even dimensions.
///
/// Uses the classic 2×2 Winograd algorithm to reduce multiplications.
/// Works best for matrices where n is even and n >= 64.
///
/// # Arguments
///
/// * `alpha` - Scalar multiplier for A*B
/// * `a` - Left matrix (m × k)
/// * `b` - Right matrix (k × n)
/// * `beta` - Scalar multiplier for C
/// * `c` - Output matrix (m × n), modified in place
///
/// # Panics
///
/// Panics if dimensions don't match or if k is not even.
pub fn gemm_winograd<T>(
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
    assert_eq!(c.nrows(), m, "Output rows must match A rows");
    assert_eq!(c.ncols(), n, "Output cols must match B cols");

    // Winograd algorithm requires even k
    if k % 2 != 0 {
        // Fall back to standard algorithm for odd k
        gemm_standard(alpha, a, b, beta, c);
        return;
    }

    let half_k = k / 2;

    // Precompute row factors for A: r[i] = sum_{j=0}^{k/2-1} A[i, 2j] * A[i, 2j+1]
    let mut row_factors = vec![T::zero(); m];
    for i in 0..m {
        let mut sum = T::zero();
        for j in 0..half_k {
            sum += a[(i, 2 * j)] * a[(i, 2 * j + 1)];
        }
        row_factors[i] = sum;
    }

    // Precompute column factors for B: c[j] = sum_{i=0}^{k/2-1} B[2i, j] * B[2i+1, j]
    let mut col_factors = vec![T::zero(); n];
    for j in 0..n {
        let mut sum = T::zero();
        for i in 0..half_k {
            sum += b[(2 * i, j)] * b[(2 * i + 1, j)];
        }
        col_factors[j] = sum;
    }

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

    // Compute C using Winograd's formula
    for i in 0..m {
        for j in 0..n {
            let mut sum = -row_factors[i] - col_factors[j];

            for l in 0..half_k {
                let temp1 = a[(i, 2 * l)] + b[(2 * l + 1, j)];
                let temp2 = a[(i, 2 * l + 1)] + b[(2 * l, j)];
                sum += temp1 * temp2;
            }

            c[(i, j)] += alpha * sum;
        }
    }
}

/// Standard GEMM fallback for when Winograd doesn't apply
fn gemm_standard<T>(alpha: T, a: MatRef<'_, T>, b: MatRef<'_, T>, beta: T, c: &mut MatMut<'_, T>)
where
    T: Field + Real,
{
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

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

    // Compute C += alpha * A * B
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

/// Blocked Winograd algorithm for better cache utilization.
///
/// Applies Winograd's algorithm to blocks of the matrix, combining the
/// reduced multiplication count with good cache behavior.
///
/// # Arguments
///
/// * `alpha` - Scalar multiplier for A*B
/// * `a` - Left matrix (m × k)
/// * `b` - Right matrix (k × n)
/// * `beta` - Scalar multiplier for C
/// * `c` - Output matrix (m × n), modified in place
/// * `block_size` - Block size for tiling (should be even)
pub fn gemm_winograd_blocked<T>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: &mut MatMut<'_, T>,
    block_size: usize,
) where
    T: Field + Real,
{
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

    assert_eq!(b.nrows(), k);
    assert_eq!(c.nrows(), m);
    assert_eq!(c.ncols(), n);
    assert!(block_size % 2 == 0, "Block size must be even");

    // Scale C by beta first
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

    // Apply Winograd to blocks
    for i_block in (0..m).step_by(block_size) {
        let i_end = (i_block + block_size).min(m);

        for j_block in (0..n).step_by(block_size) {
            let j_end = (j_block + block_size).min(n);

            for k_block in (0..k).step_by(block_size) {
                let k_end = (k_block + block_size).min(k);
                let block_k_size = k_end - k_block;

                // Only use Winograd if block k dimension is even
                if block_k_size % 2 == 0 {
                    gemm_winograd_block(
                        alpha, a, b, c, i_block, i_end, j_block, j_end, k_block, k_end,
                    );
                } else {
                    // Fall back to standard for odd k
                    gemm_standard_block(
                        alpha, a, b, c, i_block, i_end, j_block, j_end, k_block, k_end,
                    );
                }
            }
        }
    }
}

/// Apply Winograd algorithm to a single block
#[inline]
fn gemm_winograd_block<T>(
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
    let block_k = k_end - k_start;
    let half_k = block_k / 2;

    for i in i_start..i_end {
        for j in j_start..j_end {
            // Compute row and column factors for this specific i,j pair
            let mut row_factor = T::zero();
            let mut col_factor = T::zero();

            for l in 0..half_k {
                row_factor += a[(i, k_start + 2 * l)] * a[(i, k_start + 2 * l + 1)];
                col_factor += b[(k_start + 2 * l, j)] * b[(k_start + 2 * l + 1, j)];
            }

            let mut sum = -row_factor - col_factor;

            for l in 0..half_k {
                let temp1 = a[(i, k_start + 2 * l)] + b[(k_start + 2 * l + 1, j)];
                let temp2 = a[(i, k_start + 2 * l + 1)] + b[(k_start + 2 * l, j)];
                sum += temp1 * temp2;
            }

            c[(i, j)] += alpha * sum;
        }
    }
}

/// Standard GEMM for a single block
#[inline]
fn gemm_standard_block<T>(
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
    for i in i_start..i_end {
        for j in j_start..j_end {
            let mut sum = T::zero();
            for l in k_start..k_end {
                sum += a[(i, l)] * b[(l, j)];
            }
            c[(i, j)] += alpha * sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiblas_matrix::Mat;

    #[test]
    fn test_winograd_small() {
        // Test 4×4 matrices (k=4 is even)
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
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        );
        let mut c_winograd = Mat::zeros(4, 4);
        let mut c_standard = Mat::zeros(4, 4);

        gemm_winograd(1.0, a.as_ref(), b.as_ref(), 0.0, &mut c_winograd.as_mut());
        gemm_standard(1.0, a.as_ref(), b.as_ref(), 0.0, &mut c_standard.as_mut());

        // Results should match
        for i in 0..4 {
            for j in 0..4 {
                let diff: f64 = c_winograd[(i, j)] - c_standard[(i, j)];
                assert!(
                    diff.abs() < 1e-10,
                    "Mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    c_winograd[(i, j)],
                    c_standard[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_winograd_rectangular() {
        // Test rectangular matrices
        let a = Mat::from_slice(
            3,
            4,
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let b = Mat::from_slice(
            4,
            5,
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0,
            ],
        );
        let mut c_winograd = Mat::zeros(3, 5);
        let mut c_standard = Mat::zeros(3, 5);

        gemm_winograd(1.0, a.as_ref(), b.as_ref(), 0.0, &mut c_winograd.as_mut());
        gemm_standard(1.0, a.as_ref(), b.as_ref(), 0.0, &mut c_standard.as_mut());

        for i in 0..3 {
            for j in 0..5 {
                let diff: f64 = c_winograd[(i, j)] - c_standard[(i, j)];
                assert!(
                    diff.abs() < 1e-9,
                    "Mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    c_winograd[(i, j)],
                    c_standard[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_winograd_with_beta() {
        let a = Mat::from_slice(2, 4, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = Mat::from_slice(
            4,
            3,
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let mut c = Mat::from_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let c_original = c.clone();

        gemm_winograd(2.0, a.as_ref(), b.as_ref(), 3.0, &mut c.as_mut());

        // Verify with manual computation
        let mut c_expected = Mat::zeros(2, 3);
        gemm_standard(2.0, a.as_ref(), b.as_ref(), 0.0, &mut c_expected.as_mut());

        for i in 0..2 {
            for j in 0..3 {
                let expected = 3.0 * c_original[(i, j)] + c_expected[(i, j)];
                let diff: f64 = c[(i, j)] - expected;
                assert!(diff.abs() < 1e-9, "Beta scaling failed at ({}, {})", i, j);
            }
        }
    }

    #[test]
    fn test_winograd_blocked() {
        use oxiblas_matrix::Mat;

        // Test blocked variant
        let n = 16;
        let mut a_data = vec![0.0; n * n];
        let mut b_data = vec![0.0; n * n];

        for i in 0..n {
            for j in 0..n {
                a_data[i * n + j] = (i + j) as f64;
                b_data[i * n + j] = (i * 2 + j) as f64;
            }
        }

        let a = Mat::from_slice(n, n, &a_data);
        let b = Mat::from_slice(n, n, &b_data);
        let mut c_blocked = Mat::zeros(n, n);
        let mut c_standard = Mat::zeros(n, n);

        gemm_winograd_blocked(1.0, a.as_ref(), b.as_ref(), 0.0, &mut c_blocked.as_mut(), 8);
        gemm_standard(1.0, a.as_ref(), b.as_ref(), 0.0, &mut c_standard.as_mut());

        for i in 0..n {
            for j in 0..n {
                let diff: f64 = c_blocked[(i, j)] - c_standard[(i, j)];
                assert!(
                    diff.abs() < 1e-8,
                    "Blocked mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    c_blocked[(i, j)],
                    c_standard[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_winograd_odd_k_fallback() {
        // Test that odd k falls back to standard algorithm
        let a = Mat::from_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Mat::from_slice(3, 2, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let mut c = Mat::zeros(2, 2);

        // Should not panic, should fall back to standard
        gemm_winograd(1.0, a.as_ref(), b.as_ref(), 0.0, &mut c.as_mut());

        // Verify correctness
        let mut c_expected = Mat::zeros(2, 2);
        gemm_standard(1.0, a.as_ref(), b.as_ref(), 0.0, &mut c_expected.as_mut());

        for i in 0..2 {
            for j in 0..2 {
                let diff: f64 = c[(i, j)] - c_expected[(i, j)];
                assert!(diff.abs() < 1e-10, "Fallback failed at ({}, {})", i, j);
            }
        }
    }

    #[test]
    fn test_winograd_f32() {
        // Test with f32
        let a = Mat::from_slice(
            4,
            4,
            &[
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        );
        let b = Mat::from_slice(
            4,
            4,
            &[
                16.0f32, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0,
                2.0, 1.0,
            ],
        );
        let mut c_winograd = Mat::zeros(4, 4);
        let mut c_standard = Mat::zeros(4, 4);

        gemm_winograd(1.0, a.as_ref(), b.as_ref(), 0.0, &mut c_winograd.as_mut());
        gemm_standard(1.0, a.as_ref(), b.as_ref(), 0.0, &mut c_standard.as_mut());

        for i in 0..4 {
            for j in 0..4 {
                let diff: f32 = c_winograd[(i, j)] - c_standard[(i, j)];
                assert!(diff.abs() < 1e-5, "f32 mismatch at ({}, {})", i, j);
            }
        }
    }
}
