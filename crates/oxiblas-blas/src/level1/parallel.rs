//! Parallel Level 1 BLAS operations.
//!
//! This module provides parallelized versions of Level 1 operations
//! for large vectors where parallelization provides a performance benefit.
//!
//! # Usage
//!
//! These functions are available when the `parallel` feature is enabled.
//! They automatically decide whether to parallelize based on vector size.

use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::{Field, Real};

#[cfg(feature = "parallel")]
use oxiblas_core::parallel::ParThreshold;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Parallel threshold for Level 1 operations.
/// Operations with fewer elements use sequential execution.
#[cfg(feature = "parallel")]
const LEVEL1_PAR_THRESHOLD: usize = 65536;

/// Minimum work per thread for Level 1 operations.
#[cfg(feature = "parallel")]
const LEVEL1_MIN_WORK_PER_THREAD: usize = 4096;

/// Parallel AXPY: y = α·x + y with parallelization control.
///
/// Uses Rayon for parallel execution when the vector is large enough.
///
/// # Example
///
/// ```
/// use oxiblas_blas::level1::axpy_par;
/// use oxiblas_core::parallel::Par;
///
/// let x: Vec<f64> = vec![1.0; 100000];
/// let mut y: Vec<f64> = vec![2.0; 100000];
///
/// #[cfg(feature = "parallel")]
/// axpy_par(3.0, &x, &mut y, Par::Rayon);
/// #[cfg(not(feature = "parallel"))]
/// axpy_par(3.0, &x, &mut y, Par::Seq);
/// ```
pub fn axpy_par<T: Field + Send + Sync>(alpha: T, x: &[T], y: &mut [T], par: Par) {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    let n = x.len();
    if n == 0 || alpha == T::zero() {
        return;
    }

    #[cfg(feature = "parallel")]
    {
        let threshold = ParThreshold::new(LEVEL1_PAR_THRESHOLD, LEVEL1_MIN_WORK_PER_THREAD);
        if threshold.should_parallelize(n, par) {
            axpy_parallel(alpha, x, y);
            return;
        }
    }

    let _ = par;
    // Sequential fallback
    super::axpy(alpha, x, y);
}

/// Internal parallel axpy implementation using Rayon.
#[cfg(feature = "parallel")]
fn axpy_parallel<T: Field + Send + Sync>(alpha: T, x: &[T], y: &mut [T]) {
    // Use chunk size for better cache utilization
    const CHUNK_SIZE: usize = 4096;

    y.par_chunks_mut(CHUNK_SIZE)
        .zip(x.par_chunks(CHUNK_SIZE))
        .for_each(|(y_chunk, x_chunk)| {
            // Process each chunk with 8-way unrolling
            let n = y_chunk.len();
            let chunks = n / 8;
            let remainder = n % 8;

            for i in 0..chunks {
                let base = i * 8;
                y_chunk[base] = alpha * x_chunk[base] + y_chunk[base];
                y_chunk[base + 1] = alpha * x_chunk[base + 1] + y_chunk[base + 1];
                y_chunk[base + 2] = alpha * x_chunk[base + 2] + y_chunk[base + 2];
                y_chunk[base + 3] = alpha * x_chunk[base + 3] + y_chunk[base + 3];
                y_chunk[base + 4] = alpha * x_chunk[base + 4] + y_chunk[base + 4];
                y_chunk[base + 5] = alpha * x_chunk[base + 5] + y_chunk[base + 5];
                y_chunk[base + 6] = alpha * x_chunk[base + 6] + y_chunk[base + 6];
                y_chunk[base + 7] = alpha * x_chunk[base + 7] + y_chunk[base + 7];
            }

            let base = chunks * 8;
            for i in 0..remainder {
                y_chunk[base + i] = alpha * x_chunk[base + i] + y_chunk[base + i];
            }
        });
}

/// Parallel dot product with parallelization control.
///
/// Uses parallel reduction for large vectors.
///
/// # Example
///
/// ```
/// use oxiblas_blas::level1::dot_par;
/// use oxiblas_core::parallel::Par;
///
/// let x: Vec<f64> = vec![1.0; 100000];
/// let y: Vec<f64> = vec![2.0; 100000];
///
/// #[cfg(feature = "parallel")]
/// let result = dot_par(&x, &y, Par::Rayon);
/// #[cfg(not(feature = "parallel"))]
/// let result = dot_par(&x, &y, Par::Seq);
/// ```
pub fn dot_par<T: Field + Send + Sync>(x: &[T], y: &[T], par: Par) -> T {
    assert_eq!(x.len(), y.len(), "Vector lengths must match");

    let n = x.len();
    if n == 0 {
        return T::zero();
    }

    #[cfg(feature = "parallel")]
    {
        let threshold = ParThreshold::new(LEVEL1_PAR_THRESHOLD, LEVEL1_MIN_WORK_PER_THREAD);
        if threshold.should_parallelize(n, par) {
            return dot_parallel(x, y);
        }
    }

    let _ = par;
    // Sequential fallback
    super::dot(x, y)
}

/// Internal parallel dot implementation using Rayon reduction.
#[cfg(feature = "parallel")]
fn dot_parallel<T: Field + Send + Sync>(x: &[T], y: &[T]) -> T {
    const CHUNK_SIZE: usize = 4096;

    x.par_chunks(CHUNK_SIZE)
        .zip(y.par_chunks(CHUNK_SIZE))
        .map(|(x_chunk, y_chunk)| {
            // Compute local dot product with 4-way accumulation
            let n = x_chunk.len();
            let mut acc0 = T::zero();
            let mut acc1 = T::zero();
            let mut acc2 = T::zero();
            let mut acc3 = T::zero();

            let chunks = n / 4;
            let remainder = n % 4;

            for i in 0..chunks {
                let base = i * 4;
                acc0 = acc0 + x_chunk[base] * y_chunk[base];
                acc1 = acc1 + x_chunk[base + 1] * y_chunk[base + 1];
                acc2 = acc2 + x_chunk[base + 2] * y_chunk[base + 2];
                acc3 = acc3 + x_chunk[base + 3] * y_chunk[base + 3];
            }

            let base = chunks * 4;
            for i in 0..remainder {
                acc0 = acc0 + x_chunk[base + i] * y_chunk[base + i];
            }

            (acc0 + acc1) + (acc2 + acc3)
        })
        .reduce(T::zero, |a, b| a + b)
}

/// Parallel nrm2: ||x||_2 with parallelization control.
///
/// Uses parallel reduction for large vectors.
pub fn nrm2_par<T: Real + Send + Sync>(x: &[T], par: Par) -> T {
    let n = x.len();
    if n == 0 {
        return T::zero();
    }

    #[cfg(feature = "parallel")]
    {
        let threshold = ParThreshold::new(LEVEL1_PAR_THRESHOLD, LEVEL1_MIN_WORK_PER_THREAD);
        if threshold.should_parallelize(n, par) {
            return nrm2_parallel(x);
        }
    }

    let _ = par;
    super::nrm2(x)
}

/// Internal parallel nrm2 implementation.
#[cfg(feature = "parallel")]
fn nrm2_parallel<T: Real + Send + Sync>(x: &[T]) -> T {
    const CHUNK_SIZE: usize = 4096;

    // Compute sum of squares in parallel
    let sum_sq: T = x
        .par_chunks(CHUNK_SIZE)
        .map(|chunk| {
            let mut sum = T::zero();
            for xi in chunk {
                sum = sum + *xi * *xi;
            }
            sum
        })
        .reduce(T::zero, |a, b| a + b);

    Real::sqrt(sum_sq)
}

/// Parallel asum: ||x||_1 with parallelization control.
///
/// Uses parallel reduction for large vectors.
pub fn asum_par<T: Real + Send + Sync>(x: &[T], par: Par) -> T {
    let n = x.len();
    if n == 0 {
        return T::zero();
    }

    #[cfg(feature = "parallel")]
    {
        let threshold = ParThreshold::new(LEVEL1_PAR_THRESHOLD, LEVEL1_MIN_WORK_PER_THREAD);
        if threshold.should_parallelize(n, par) {
            return asum_parallel(x);
        }
    }

    let _ = par;
    super::asum(x)
}

/// Internal parallel asum implementation.
#[cfg(feature = "parallel")]
fn asum_parallel<T: Real + Send + Sync>(x: &[T]) -> T {
    use oxiblas_core::scalar::Scalar;

    const CHUNK_SIZE: usize = 4096;

    x.par_chunks(CHUNK_SIZE)
        .map(|chunk| {
            let mut sum = T::zero();
            for xi in chunk {
                sum = sum + Scalar::abs(*xi);
            }
            sum
        })
        .reduce(T::zero, |a, b| a + b)
}

/// Parallel scal: x = α·x with parallelization control.
pub fn scal_par<T: Field + Send + Sync>(alpha: T, x: &mut [T], par: Par) {
    let n = x.len();
    if n == 0 {
        return;
    }

    if alpha == T::zero() {
        // Fast path: zero out the vector
        for xi in x.iter_mut() {
            *xi = T::zero();
        }
        return;
    }

    if alpha == T::one() {
        return;
    }

    #[cfg(feature = "parallel")]
    {
        let threshold = ParThreshold::new(LEVEL1_PAR_THRESHOLD, LEVEL1_MIN_WORK_PER_THREAD);
        if threshold.should_parallelize(n, par) {
            scal_parallel(alpha, x);
            return;
        }
    }

    let _ = par;
    super::scal(alpha, x);
}

/// Internal parallel scal implementation.
#[cfg(feature = "parallel")]
fn scal_parallel<T: Field + Send + Sync>(alpha: T, x: &mut [T]) {
    const CHUNK_SIZE: usize = 4096;

    x.par_chunks_mut(CHUNK_SIZE).for_each(|chunk| {
        for xi in chunk.iter_mut() {
            *xi = alpha * *xi;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_axpy_par_sequential() {
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let mut y: Vec<f64> = vec![5.0, 6.0, 7.0, 8.0];

        axpy_par(2.0, &x, &mut y, Par::Seq);

        assert!((y[0] - 7.0).abs() < 1e-10);
        assert!((y[1] - 10.0).abs() < 1e-10);
        assert!((y[2] - 13.0).abs() < 1e-10);
        assert!((y[3] - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot_par_sequential() {
        let x: Vec<f64> = vec![1.0, 2.0, 3.0];
        let y: Vec<f64> = vec![4.0, 5.0, 6.0];

        let result = dot_par(&x, &y, Par::Seq);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((result - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_nrm2_par_sequential() {
        let x: Vec<f64> = vec![3.0, 4.0];

        let result = nrm2_par(&x, Par::Seq);
        // sqrt(9 + 16) = 5
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_asum_par_sequential() {
        let x: Vec<f64> = vec![1.0, -2.0, 3.0, -4.0];

        let result = asum_par(&x, Par::Seq);
        // |1| + |-2| + |3| + |-4| = 10
        assert!((result - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_scal_par_sequential() {
        let mut x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];

        scal_par(2.0, &mut x, Par::Seq);

        assert!((x[0] - 2.0).abs() < 1e-10);
        assert!((x[1] - 4.0).abs() < 1e-10);
        assert!((x[2] - 6.0).abs() < 1e-10);
        assert!((x[3] - 8.0).abs() < 1e-10);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_axpy_par_large() {
        let n = 100000;
        let x: Vec<f64> = vec![1.0; n];
        let mut y: Vec<f64> = vec![2.0; n];

        axpy_par(3.0, &x, &mut y, Par::Rayon);

        // y = 3*1 + 2 = 5
        for yi in y.iter() {
            assert!((*yi - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_dot_par_large() {
        let n = 100000;
        let x: Vec<f64> = vec![1.0; n];
        let y: Vec<f64> = vec![1.0; n];

        let result = dot_par(&x, &y, Par::Rayon);

        assert!(
            (result - n as f64).abs() < 1e-6,
            "Expected {}, got {}",
            n,
            result
        );
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_nrm2_par_large() {
        let n = 100000;
        let x: Vec<f64> = vec![1.0; n];

        let result = nrm2_par(&x, Par::Rayon);
        let expected = (n as f64).sqrt();

        assert!(
            (result - expected).abs() < 1e-6,
            "Expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_asum_par_large() {
        let n = 100000;
        let x: Vec<f64> = vec![1.0; n];

        let result = asum_par(&x, Par::Rayon);

        assert!(
            (result - n as f64).abs() < 1e-6,
            "Expected {}, got {}",
            n,
            result
        );
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_scal_par_large() {
        let n = 100000;
        let mut x: Vec<f64> = vec![2.0; n];

        scal_par(3.0, &mut x, Par::Rayon);

        for xi in x.iter() {
            assert!((*xi - 6.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_parallel_empty_vectors() {
        let x: Vec<f64> = vec![];
        let mut y: Vec<f64> = vec![];

        axpy_par(1.0, &x, &mut y, Par::Seq);
        let empty: &[f64] = &[];
        assert_eq!(dot_par(empty, empty, Par::Seq), 0.0);
        assert_eq!(nrm2_par(empty, Par::Seq), 0.0);
        assert_eq!(asum_par(empty, Par::Seq), 0.0);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_consistency() {
        // Verify parallel and sequential give same results
        let n = 100000;
        let x: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001).collect();
        let y: Vec<f64> = (0..n).map(|i| ((n - i) as f64) * 0.001).collect();

        let result_seq = dot_par(&x, &y, Par::Seq);
        let result_par = dot_par(&x, &y, Par::Rayon);

        let rel_err = (result_seq - result_par).abs() / result_seq.abs();
        assert!(
            rel_err < 1e-10,
            "seq={}, par={}, rel_err={}",
            result_seq,
            result_par,
            rel_err
        );
    }
}
