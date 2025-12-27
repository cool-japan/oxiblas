//! Complex GEMM using the 3M method.
//!
//! The 3M method (Gauss's complex multiplication algorithm) reduces the number
//! of real multiplications from 4 to 3 at the cost of additional additions.
//!
//! For complex multiplication (a + bi)(c + di):
//! - Standard: 4 muls, 2 adds: ac - bd + i(ad + bc)
//! - 3M method: 3 muls, 5 adds:
//!   k1 = a * c
//!   k2 = b * d
//!   k3 = (a + b)(c + d)
//!   real = k1 - k2
//!   imag = k3 - k1 - k2
//!
//! This module uses optimized real GEMM kernels for the three matrix multiplications.

use crate::level3::gemm::gemm;
use num_complex::{Complex32, Complex64};
use oxiblas_matrix::{Mat, MatMut, MatRef};

/// Complex GEMM using the 3M method for Complex64.
///
/// C = alpha * A * B + beta * C
///
/// This uses 3 real matrix multiplications instead of 4, which can be
/// significantly faster when real GEMM is highly optimized.
///
/// # Arguments
///
/// * `alpha` - Scalar multiplier for A*B
/// * `a` - First input matrix (m×k)
/// * `b` - Second input matrix (k×n)
/// * `beta` - Scalar multiplier for C
/// * `c` - Output matrix (m×n), overwritten with result
///
/// # Panics
///
/// Panics if matrix dimensions are incompatible.
pub fn gemm3m_c64(
    alpha: Complex64,
    a: MatRef<Complex64>,
    b: MatRef<Complex64>,
    beta: Complex64,
    mut c: MatMut<Complex64>,
) {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

    assert_eq!(b.nrows(), k, "Inner dimensions must match");
    assert_eq!(c.nrows(), m, "Output rows must match A rows");
    assert_eq!(c.ncols(), n, "Output cols must match B cols");

    // Handle trivial cases
    if m == 0 || n == 0 {
        return;
    }

    if k == 0 {
        // C = beta * C
        if beta == Complex64::new(0.0, 0.0) {
            for i in 0..m {
                for j in 0..n {
                    c[(i, j)] = Complex64::new(0.0, 0.0);
                }
            }
        } else if beta != Complex64::new(1.0, 0.0) {
            for i in 0..m {
                for j in 0..n {
                    c[(i, j)] = beta * c[(i, j)];
                }
            }
        }
        return;
    }

    // For small matrices, use naive implementation
    if m * n * k <= 32 * 32 * 32 {
        gemm3m_c64_naive(alpha, a, b, beta, c);
        return;
    }

    // Extract real and imaginary parts of A and B
    let mut a_re: Mat<f64> = Mat::zeros(m, k);
    let mut a_im: Mat<f64> = Mat::zeros(m, k);
    let mut b_re: Mat<f64> = Mat::zeros(k, n);
    let mut b_im: Mat<f64> = Mat::zeros(k, n);

    for i in 0..m {
        for j in 0..k {
            let val = a[(i, j)];
            a_re[(i, j)] = val.re;
            a_im[(i, j)] = val.im;
        }
    }

    for i in 0..k {
        for j in 0..n {
            let val = b[(i, j)];
            b_re[(i, j)] = val.re;
            b_im[(i, j)] = val.im;
        }
    }

    // Compute helper matrices for 3M method
    // a_sum = a_re + a_im
    // b_sum = b_re + b_im
    let mut a_sum: Mat<f64> = Mat::zeros(m, k);
    let mut b_sum: Mat<f64> = Mat::zeros(k, n);

    for i in 0..m {
        for j in 0..k {
            a_sum[(i, j)] = a_re[(i, j)] + a_im[(i, j)];
        }
    }

    for i in 0..k {
        for j in 0..n {
            b_sum[(i, j)] = b_re[(i, j)] + b_im[(i, j)];
        }
    }

    // Three real matrix multiplications using optimized GEMM
    // k1 = a_re * b_re
    // k2 = a_im * b_im
    // k3 = a_sum * b_sum = (a_re + a_im) * (b_re + b_im)
    let mut k1: Mat<f64> = Mat::zeros(m, n);
    let mut k2: Mat<f64> = Mat::zeros(m, n);
    let mut k3: Mat<f64> = Mat::zeros(m, n);

    // Use optimized real GEMM kernels
    gemm(1.0f64, a_re.as_ref(), b_re.as_ref(), 0.0f64, k1.as_mut());
    gemm(1.0f64, a_im.as_ref(), b_im.as_ref(), 0.0f64, k2.as_mut());
    gemm(1.0f64, a_sum.as_ref(), b_sum.as_ref(), 0.0f64, k3.as_mut());

    // Combine results: real = k1 - k2, imag = k3 - k1 - k2
    // C = alpha * (real + i*imag) + beta * C
    for i in 0..m {
        for j in 0..n {
            let real_part = k1[(i, j)] - k2[(i, j)];
            let imag_part = k3[(i, j)] - k1[(i, j)] - k2[(i, j)];

            let product = Complex64::new(real_part, imag_part);
            let old_val = c[(i, j)];

            c[(i, j)] = alpha * product + beta * old_val;
        }
    }
}

/// Naive 3M implementation for small matrices.
fn gemm3m_c64_naive(
    alpha: Complex64,
    a: MatRef<Complex64>,
    b: MatRef<Complex64>,
    beta: Complex64,
    mut c: MatMut<Complex64>,
) {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

    // Extract real and imaginary parts
    let mut a_re = vec![0.0f64; m * k];
    let mut a_im = vec![0.0f64; m * k];
    let mut b_re = vec![0.0f64; k * n];
    let mut b_im = vec![0.0f64; k * n];

    for i in 0..m {
        for j in 0..k {
            let val = a[(i, j)];
            a_re[i * k + j] = val.re;
            a_im[i * k + j] = val.im;
        }
    }

    for i in 0..k {
        for j in 0..n {
            let val = b[(i, j)];
            b_re[i * n + j] = val.re;
            b_im[i * n + j] = val.im;
        }
    }

    // Compute helper matrices for 3M method
    let mut a_sum = vec![0.0f64; m * k]; // a_re + a_im
    let mut b_sum = vec![0.0f64; k * n]; // b_re + b_im

    for i in 0..m * k {
        a_sum[i] = a_re[i] + a_im[i];
    }

    for i in 0..k * n {
        b_sum[i] = b_re[i] + b_im[i];
    }

    // Three real matrix multiplications
    let mut k1 = vec![0.0f64; m * n]; // a_re * b_re
    let mut k2 = vec![0.0f64; m * n]; // a_im * b_im
    let mut k3 = vec![0.0f64; m * n]; // (a_re + a_im) * (b_re + b_im)

    // k1 = a_re * b_re
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a_re[i * k + p] * b_re[p * n + j];
            }
            k1[i * n + j] = sum;
        }
    }

    // k2 = a_im * b_im
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a_im[i * k + p] * b_im[p * n + j];
            }
            k2[i * n + j] = sum;
        }
    }

    // k3 = a_sum * b_sum
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a_sum[i * k + p] * b_sum[p * n + j];
            }
            k3[i * n + j] = sum;
        }
    }

    // Combine results: real = k1 - k2, imag = k3 - k1 - k2
    for i in 0..m {
        for j in 0..n {
            let idx = i * n + j;
            let real_part = k1[idx] - k2[idx];
            let imag_part = k3[idx] - k1[idx] - k2[idx];

            let product = Complex64::new(real_part, imag_part);
            let old_val = c[(i, j)];

            c[(i, j)] = alpha * product + beta * old_val;
        }
    }
}

/// Complex GEMM using the 3M method for Complex32.
///
/// C = alpha * A * B + beta * C
///
/// This uses 3 real matrix multiplications instead of 4.
pub fn gemm3m_c32(
    alpha: Complex32,
    a: MatRef<Complex32>,
    b: MatRef<Complex32>,
    beta: Complex32,
    mut c: MatMut<Complex32>,
) {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

    assert_eq!(b.nrows(), k, "Inner dimensions must match");
    assert_eq!(c.nrows(), m, "Output rows must match A rows");
    assert_eq!(c.ncols(), n, "Output cols must match B cols");

    // Handle trivial cases
    if m == 0 || n == 0 {
        return;
    }

    if k == 0 {
        // C = beta * C
        if beta == Complex32::new(0.0, 0.0) {
            for i in 0..m {
                for j in 0..n {
                    c[(i, j)] = Complex32::new(0.0, 0.0);
                }
            }
        } else if beta != Complex32::new(1.0, 0.0) {
            for i in 0..m {
                for j in 0..n {
                    c[(i, j)] = beta * c[(i, j)];
                }
            }
        }
        return;
    }

    // For small matrices, use naive implementation
    if m * n * k <= 32 * 32 * 32 {
        gemm3m_c32_naive(alpha, a, b, beta, c);
        return;
    }

    // Extract real and imaginary parts of A and B
    let mut a_re: Mat<f32> = Mat::zeros(m, k);
    let mut a_im: Mat<f32> = Mat::zeros(m, k);
    let mut b_re: Mat<f32> = Mat::zeros(k, n);
    let mut b_im: Mat<f32> = Mat::zeros(k, n);

    for i in 0..m {
        for j in 0..k {
            let val = a[(i, j)];
            a_re[(i, j)] = val.re;
            a_im[(i, j)] = val.im;
        }
    }

    for i in 0..k {
        for j in 0..n {
            let val = b[(i, j)];
            b_re[(i, j)] = val.re;
            b_im[(i, j)] = val.im;
        }
    }

    // Compute helper matrices
    let mut a_sum: Mat<f32> = Mat::zeros(m, k);
    let mut b_sum: Mat<f32> = Mat::zeros(k, n);

    for i in 0..m {
        for j in 0..k {
            a_sum[(i, j)] = a_re[(i, j)] + a_im[(i, j)];
        }
    }

    for i in 0..k {
        for j in 0..n {
            b_sum[(i, j)] = b_re[(i, j)] + b_im[(i, j)];
        }
    }

    // Three real matrix multiplications using optimized GEMM
    let mut k1: Mat<f32> = Mat::zeros(m, n);
    let mut k2: Mat<f32> = Mat::zeros(m, n);
    let mut k3: Mat<f32> = Mat::zeros(m, n);

    gemm(1.0f32, a_re.as_ref(), b_re.as_ref(), 0.0f32, k1.as_mut());
    gemm(1.0f32, a_im.as_ref(), b_im.as_ref(), 0.0f32, k2.as_mut());
    gemm(1.0f32, a_sum.as_ref(), b_sum.as_ref(), 0.0f32, k3.as_mut());

    // Combine results
    for i in 0..m {
        for j in 0..n {
            let real_part = k1[(i, j)] - k2[(i, j)];
            let imag_part = k3[(i, j)] - k1[(i, j)] - k2[(i, j)];

            let product = Complex32::new(real_part, imag_part);
            let old_val = c[(i, j)];

            c[(i, j)] = alpha * product + beta * old_val;
        }
    }
}

/// Naive 3M implementation for Complex32.
fn gemm3m_c32_naive(
    alpha: Complex32,
    a: MatRef<Complex32>,
    b: MatRef<Complex32>,
    beta: Complex32,
    mut c: MatMut<Complex32>,
) {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

    // Extract real and imaginary parts
    let mut a_re = vec![0.0f32; m * k];
    let mut a_im = vec![0.0f32; m * k];
    let mut b_re = vec![0.0f32; k * n];
    let mut b_im = vec![0.0f32; k * n];

    for i in 0..m {
        for j in 0..k {
            let val = a[(i, j)];
            a_re[i * k + j] = val.re;
            a_im[i * k + j] = val.im;
        }
    }

    for i in 0..k {
        for j in 0..n {
            let val = b[(i, j)];
            b_re[i * n + j] = val.re;
            b_im[i * n + j] = val.im;
        }
    }

    // Compute helper matrices
    let mut a_sum = vec![0.0f32; m * k];
    let mut b_sum = vec![0.0f32; k * n];

    for i in 0..m * k {
        a_sum[i] = a_re[i] + a_im[i];
    }

    for i in 0..k * n {
        b_sum[i] = b_re[i] + b_im[i];
    }

    // Three real matrix multiplications (fused for better cache usage)
    for i in 0..m {
        for j in 0..n {
            let mut sum1 = 0.0f32;
            let mut sum2 = 0.0f32;
            let mut sum3 = 0.0f32;
            for p in 0..k {
                sum1 += a_re[i * k + p] * b_re[p * n + j];
                sum2 += a_im[i * k + p] * b_im[p * n + j];
                sum3 += a_sum[i * k + p] * b_sum[p * n + j];
            }

            let real_part = sum1 - sum2;
            let imag_part = sum3 - sum1 - sum2;

            let product = Complex32::new(real_part, imag_part);
            let old_val = c[(i, j)];

            c[(i, j)] = alpha * product + beta * old_val;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemm3m_c64_simple() {
        let a = Mat::from_rows(&[
            &[Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
            &[Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)],
        ]);

        let b = Mat::from_rows(&[
            &[Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)],
            &[Complex64::new(0.0, 1.0), Complex64::new(1.0, 0.0)],
        ]);

        let mut c = Mat::zeros(2, 2);

        gemm3m_c64(
            Complex64::new(1.0, 0.0),
            a.as_ref(),
            b.as_ref(),
            Complex64::new(0.0, 0.0),
            c.as_mut(),
        );

        // Verify result
        // C[0,0] = (1+2i)*(1) + (3+4i)*(i) = (1+2i) + (3i-4) = -3+5i
        assert!((c[(0, 0)].re - (-3.0)).abs() < 1e-10);
        assert!((c[(0, 0)].im - 5.0).abs() < 1e-10);

        // C[0,1] = (1+2i)*(i) + (3+4i)*(1) = (i-2) + (3+4i) = 1+5i
        assert!((c[(0, 1)].re - 1.0).abs() < 1e-10);
        assert!((c[(0, 1)].im - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemm3m_c64_identity() {
        let a = Mat::from_rows(&[
            &[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            &[Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ]);

        let b = Mat::from_rows(&[
            &[Complex64::new(5.0, 3.0), Complex64::new(2.0, 1.0)],
            &[Complex64::new(4.0, 2.0), Complex64::new(6.0, 3.0)],
        ]);

        let mut c = Mat::zeros(2, 2);

        gemm3m_c64(
            Complex64::new(1.0, 0.0),
            a.as_ref(),
            b.as_ref(),
            Complex64::new(0.0, 0.0),
            c.as_mut(),
        );

        // Should give b back
        assert!((c[(0, 0)] - b[(0, 0)]).norm() < 1e-10);
        assert!((c[(0, 1)] - b[(0, 1)]).norm() < 1e-10);
        assert!((c[(1, 0)] - b[(1, 0)]).norm() < 1e-10);
        assert!((c[(1, 1)] - b[(1, 1)]).norm() < 1e-10);
    }

    #[test]
    fn test_gemm3m_c32_basic() {
        let a = Mat::from_rows(&[&[Complex32::new(2.0, 1.0), Complex32::new(1.0, 2.0)]]);

        let b = Mat::from_rows(&[&[Complex32::new(1.0, 1.0)], &[Complex32::new(2.0, 1.0)]]);

        let mut c = Mat::zeros(1, 1);

        gemm3m_c32(
            Complex32::new(1.0, 0.0),
            a.as_ref(),
            b.as_ref(),
            Complex32::new(0.0, 0.0),
            c.as_mut(),
        );

        // (2+i)(1+i) + (1+2i)(2+i)
        // = (2+2i+i+i²) + (2+i+4i+2i²)
        // = (2+3i-1) + (2+5i-2)
        // = (1+3i) + (0+5i)
        // = 1+8i
        assert!((c[(0, 0)].re - 1.0).abs() < 1e-5);
        assert!((c[(0, 0)].im - 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_gemm3m_c64_larger() {
        // Test with a larger matrix to use the optimized path
        let n = 64;
        let a: Mat<Complex64> = Mat::filled(n, n, Complex64::new(1.0, 1.0));
        let b: Mat<Complex64> = Mat::filled(n, n, Complex64::new(1.0, 0.0));
        let mut c: Mat<Complex64> = Mat::zeros(n, n);

        gemm3m_c64(
            Complex64::new(1.0, 0.0),
            a.as_ref(),
            b.as_ref(),
            Complex64::new(0.0, 0.0),
            c.as_mut(),
        );

        // Each element should be n * (1+i) * (1+0i) = n * (1+i)
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (c[(i, j)].re - n as f64).abs() < 1e-8,
                    "Real part mismatch at ({}, {}): got {}, expected {}",
                    i,
                    j,
                    c[(i, j)].re,
                    n
                );
                assert!(
                    (c[(i, j)].im - n as f64).abs() < 1e-8,
                    "Imag part mismatch at ({}, {}): got {}, expected {}",
                    i,
                    j,
                    c[(i, j)].im,
                    n
                );
            }
        }
    }

    #[test]
    fn test_gemm3m_c32_larger() {
        // Test with a larger matrix to use the optimized path
        let n = 64;
        let a: Mat<Complex32> = Mat::filled(n, n, Complex32::new(1.0, 0.0));
        let b: Mat<Complex32> = Mat::filled(n, n, Complex32::new(1.0, 0.0));
        let mut c: Mat<Complex32> = Mat::zeros(n, n);

        gemm3m_c32(
            Complex32::new(1.0, 0.0),
            a.as_ref(),
            b.as_ref(),
            Complex32::new(0.0, 0.0),
            c.as_mut(),
        );

        // Each element should be n (pure real)
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (c[(i, j)].re - n as f32).abs() < 1e-4,
                    "Real part mismatch at ({}, {}): got {}, expected {}",
                    i,
                    j,
                    c[(i, j)].re,
                    n
                );
                assert!(
                    c[(i, j)].im.abs() < 1e-4,
                    "Imag part should be zero at ({}, {}): got {}",
                    i,
                    j,
                    c[(i, j)].im
                );
            }
        }
    }

    #[test]
    fn test_gemm3m_with_alpha_beta() {
        let a = Mat::from_rows(&[
            &[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            &[Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ]);

        let b = Mat::from_rows(&[
            &[Complex64::new(2.0, 1.0), Complex64::new(3.0, 2.0)],
            &[Complex64::new(4.0, 3.0), Complex64::new(5.0, 4.0)],
        ]);

        let mut c = Mat::from_rows(&[
            &[Complex64::new(1.0, 1.0), Complex64::new(1.0, 1.0)],
            &[Complex64::new(1.0, 1.0), Complex64::new(1.0, 1.0)],
        ]);

        // C = 2 * A * B + 3 * C
        // A * B = B (identity)
        // C = 2 * B + 3 * C = 2 * B + 3 * [1+i, ...]
        gemm3m_c64(
            Complex64::new(2.0, 0.0),
            a.as_ref(),
            b.as_ref(),
            Complex64::new(3.0, 0.0),
            c.as_mut(),
        );

        // C[0,0] = 2*(2+i) + 3*(1+i) = (4+2i) + (3+3i) = 7+5i
        assert!((c[(0, 0)].re - 7.0).abs() < 1e-10);
        assert!((c[(0, 0)].im - 5.0).abs() < 1e-10);
    }
}
