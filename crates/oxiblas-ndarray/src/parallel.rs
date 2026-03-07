//! Parallel BLAS operations on ndarray types.
//!
//! This module provides parallelized BLAS operations using Rayon,
//! gated behind the `parallel` feature flag.
//!
//! # Example
//!
//! ```ignore
//! use oxiblas_ndarray::parallel::{gemm_par_ndarray, matmul_par};
//! use ndarray::Array2;
//!
//! let a = Array2::from_shape_fn((100, 100), |(i, j)| (i + j) as f64);
//! let b = Array2::from_shape_fn((100, 100), |(i, j)| (i * j) as f64);
//!
//! // Parallel matrix multiply
//! let c = matmul_par(&a, &b);
//! ```

use crate::conversions::array2_to_mat;
use ndarray::{Array2, ShapeBuilder};
use oxiblas_blas::level3::{GemmKernel, gemm_with_par};
use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::Field;
use oxiblas_matrix::Mat;

/// Parallel general matrix-matrix multiplication: C = alpha * A * B + beta * C
///
/// Uses Rayon to parallelize the GEMM computation across available threads.
///
/// # Arguments
/// * `alpha` - Scalar multiplier for A * B
/// * `a` - Left matrix (m x k)
/// * `b` - Right matrix (k x n)
/// * `beta` - Scalar multiplier for C
/// * `c` - Output matrix (m x n), modified in place
///
/// # Panics
/// Panics if matrix dimensions are incompatible.
pub fn gemm_par_ndarray<T: Field + GemmKernel>(
    alpha: T,
    a: &Array2<T>,
    b: &Array2<T>,
    beta: T,
    c: &mut Array2<T>,
) where
    T: bytemuck::Zeroable + Clone,
{
    let a_mat = array2_to_mat(a);
    let b_mat = array2_to_mat(b);

    let (m, n) = c.dim();
    let mut c_mat: Mat<T> = Mat::zeros(m, n);

    // Copy existing C values if beta != 0
    if beta != T::zero() {
        for i in 0..m {
            for j in 0..n {
                c_mat[(i, j)] = c[[i, j]];
            }
        }
    }

    gemm_with_par(
        alpha,
        a_mat.as_ref(),
        b_mat.as_ref(),
        beta,
        c_mat.as_mut(),
        Par::Rayon,
    );

    // Copy result back
    for i in 0..m {
        for j in 0..n {
            c[[i, j]] = c_mat[(i, j)];
        }
    }
}

/// Parallel matrix multiplication: C = A * B
///
/// Simplified parallel version that allocates a new output matrix.
/// Uses all available Rayon threads for computation.
///
/// # Arguments
/// * `a` - Left matrix (m x k)
/// * `b` - Right matrix (k x n)
///
/// # Returns
/// New matrix C = A * B in column-major order
///
/// # Panics
/// Panics if inner dimensions do not match.
pub fn matmul_par<T: Field + GemmKernel>(a: &Array2<T>, b: &Array2<T>) -> Array2<T>
where
    T: bytemuck::Zeroable + Clone,
{
    let (m, k1) = a.dim();
    let (k2, n) = b.dim();
    assert_eq!(k1, k2, "Inner dimensions must match: {} vs {}", k1, k2);

    let a_mat = array2_to_mat(a);
    let b_mat = array2_to_mat(b);
    let mut c_mat: Mat<T> = Mat::zeros(m, n);

    gemm_with_par(
        T::one(),
        a_mat.as_ref(),
        b_mat.as_ref(),
        T::zero(),
        c_mat.as_mut(),
        Par::Rayon,
    );

    Array2::from_shape_fn((m, n).f(), |(i, j)| c_mat[(i, j)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_matmul_par_basic() {
        let a = Array2::from_shape_fn((2, 3), |_| 1.0f64);
        let b = Array2::from_shape_fn((3, 2), |_| 2.0f64);
        let c = matmul_par(&a, &b);

        assert_eq!(c.dim(), (2, 2));
        for i in 0..2 {
            for j in 0..2 {
                assert!((c[[i, j]] - 6.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_matmul_par_identity() {
        let n = 50;
        let a = Array2::from_shape_fn((n, n), |(i, j)| (i * n + j + 1) as f64);
        let id = {
            let mut m = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                m[[i, i]] = 1.0;
            }
            m
        };

        let c = matmul_par(&a, &id);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (c[[i, j]] - a[[i, j]]).abs() < 1e-10,
                    "Mismatch at ({}, {})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_gemm_par_ndarray_with_beta() {
        let a = Array2::from_shape_fn((2, 3), |_| 1.0f64);
        let b = Array2::from_shape_fn((3, 2), |_| 2.0f64);
        let mut c = Array2::from_shape_fn((2, 2), |_| 1.0f64);

        gemm_par_ndarray(1.0, &a, &b, 1.0, &mut c);

        // C = 1 * A * B + 1 * C = 6 + 1 = 7
        for i in 0..2 {
            for j in 0..2 {
                assert!((c[[i, j]] - 7.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_matmul_par_rectangular() {
        let a = array![[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b = array![[7.0f64, 8.0], [9.0, 10.0], [11.0, 12.0]];
        let c = matmul_par(&a, &b);

        assert_eq!(c.dim(), (2, 2));
        // c[0,0] = 1*7 + 2*9 + 3*11 = 7+18+33 = 58
        assert!((c[[0, 0]] - 58.0).abs() < 1e-10);
        // c[0,1] = 1*8 + 2*10 + 3*12 = 8+20+36 = 64
        assert!((c[[0, 1]] - 64.0).abs() < 1e-10);
        // c[1,0] = 4*7 + 5*9 + 6*11 = 28+45+66 = 139
        assert!((c[[1, 0]] - 139.0).abs() < 1e-10);
        // c[1,1] = 4*8 + 5*10 + 6*12 = 32+50+72 = 154
        assert!((c[[1, 1]] - 154.0).abs() < 1e-10);
    }

    #[test]
    fn test_matmul_par_f32() {
        let a = Array2::from_shape_fn((3, 3), |(i, j)| (i * 3 + j + 1) as f32);
        let b = Array2::from_shape_fn((3, 3), |(i, j)| if i == j { 1.0f32 } else { 0.0f32 });
        let c = matmul_par(&a, &b);

        for i in 0..3 {
            for j in 0..3 {
                assert!((c[[i, j]] - a[[i, j]]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_matmul_par_larger() {
        let n = 100;
        let a = Array2::from_shape_fn((n, n), |(i, j)| if i == j { 2.0f64 } else { 0.0 });
        let b = Array2::from_shape_fn((n, n), |(i, j)| (i + j) as f64);
        let c = matmul_par(&a, &b);

        // C = 2*I * B = 2*B
        for i in 0..n {
            for j in 0..n {
                let expected = 2.0 * (i + j) as f64;
                assert!(
                    (c[[i, j]] - expected).abs() < 1e-10,
                    "Mismatch at ({}, {}): got {} expected {}",
                    i,
                    j,
                    c[[i, j]],
                    expected
                );
            }
        }
    }
}
