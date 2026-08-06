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

/// Combines two Blue's-scaling `(scale, ssq)` accumulators, each satisfying
/// `Σ|x|² == scale² · ssq`, into one covering both partial sums.
///
/// This is the cross-chunk reduction step the parallel nrm2 needs: without it
/// each chunk's sum of squares would be summed naively, which overflows the
/// moment a single chunk holds a large value (e.g. `1e200² == 1e400`) and loses
/// precision when chunk magnitudes differ widely.
///
/// Correctness on non-finite / degenerate scales:
/// - Two equal scales (both `0` for all-zero chunks, or both `+Inf`) form ratios
///   `0/0` and `Inf/Inf` (= NaN); the `scale_a == scale_b` branch adds the
///   `ssq`s directly, so all-zero chunks stay `0` and all-infinite chunks stay
///   `+Inf` instead of collapsing to NaN.
/// - A NaN `ssq` (produced by a NaN input in `nrm2_fold`) always survives: it is
///   added in every branch, and even when scaled by `t·t` the product
///   `NaN · anything == NaN`, so a NaN input propagates to the final norm.
#[cfg(feature = "parallel")]
fn nrm2_combine<T: Real>(a: (T, T), b: (T, T)) -> (T, T) {
    let (scale_a, ssq_a) = a;
    let (scale_b, ssq_b) = b;

    if scale_a == scale_b {
        if scale_a == T::zero() {
            // Both chunks contributed nothing; keep the neutral accumulator.
            return (T::zero(), T::one());
        }
        // Equal (finite or infinite) scales: no rescaling ratio is needed.
        return (scale_a, ssq_a + ssq_b);
    }

    if scale_a > scale_b {
        // `scale_a` is the common scale; `scale_b / scale_a` is finite (and `0`
        // when `scale_a` is `+Inf`), so no spurious NaN is introduced.
        let t = scale_b / scale_a;
        (scale_a, ssq_a + ssq_b * t * t)
    } else {
        let t = scale_a / scale_b;
        (scale_b, ssq_b + ssq_a * t * t)
    }
}

/// Internal parallel nrm2 implementation.
///
/// Each chunk folds its elements into a Blue's-scaling `(scale, ssq)` pair; the
/// pairs are then reduced with [`nrm2_combine`], which rescales across chunk
/// boundaries. This keeps the whole computation overflow/underflow-safe and
/// consistent with the scalar and SIMD paths — a naive per-chunk sum of squares
/// would overflow or lose precision.
#[cfg(feature = "parallel")]
fn nrm2_parallel<T: Real + Send + Sync>(x: &[T]) -> T {
    use super::nrm2::{nrm2_finalize, nrm2_fold};

    const CHUNK_SIZE: usize = 4096;

    let state = x
        .par_chunks(CHUNK_SIZE)
        .map(|chunk| {
            let mut local = (T::zero(), T::one());
            for &xi in chunk {
                local = nrm2_fold(local, xi);
            }
            local
        })
        .reduce(|| (T::zero(), T::one()), nrm2_combine);

    nrm2_finalize(state)
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

    #[test]
    #[cfg(feature = "parallel")]
    fn test_nrm2_par_overflow_cross_chunk() {
        // A value large enough that squaring it overflows f64 lives in one chunk
        // and a tiny value in another (chunks are 4096 elements wide, so indices
        // 0 and 80_000 fall in different chunks). A naive per-chunk sum of
        // squares computes 1e200² == 1e400 == +Inf and returns +Inf; the scaled
        // cross-chunk combine must instead return the correct finite norm.
        let n = 100_000;
        let mut x = vec![0.0f64; n];
        x[0] = 1e200;
        x[80_000] = 1e-200;

        let result = nrm2_par(&x, Par::Rayon);
        assert!(
            result.is_finite(),
            "cross-chunk norm must be finite, got {}",
            result
        );
        // The 1e-200 term is ~800 orders of magnitude below 1e200, so the norm
        // rounds to exactly 1e200 in f64.
        assert!(
            (result - 1e200).abs() / 1e200 < 1e-10,
            "expected ≈1e200, got {}",
            result
        );
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_nrm2_par_two_large_cross_chunk() {
        // Two overflow-prone values in distinct chunks: correct norm is
        // sqrt(2) * 1e200. Naive per-chunk sums would overflow to +Inf.
        let n = 100_000;
        let mut x = vec![0.0f64; n];
        x[10] = 1e200;
        x[90_000] = 1e200;

        let result = nrm2_par(&x, Par::Rayon);
        let expected = (2.0f64).sqrt() * 1e200;
        assert!(
            result.is_finite(),
            "cross-chunk norm must be finite, got {}",
            result
        );
        assert!(
            (result - expected).abs() / expected < 1e-10,
            "expected {}, got {}",
            expected,
            result
        );
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_nrm2_par_matches_scalar_wide_range() {
        // Parallel scaled reduction must match the scalar reference norm even
        // when magnitudes span a wide dynamic range across many chunks.
        let n = 100_000;
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                sign * (1e-50 + (i as f64) * 1e-40)
            })
            .collect();

        let par = nrm2_par(&x, Par::Rayon);
        let seq = super::super::nrm2(&x);
        assert!(
            (par - seq).abs() / seq < 1e-12,
            "parallel {} vs scalar {}",
            par,
            seq
        );
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_nrm2_par_nan_propagates() {
        // A single NaN anywhere in a large (multi-chunk) vector must make the
        // whole parallel norm NaN.
        let n = 100_000;
        let mut x = vec![1.0f64; n];
        x[50_000] = f64::NAN;
        assert!(
            nrm2_par(&x, Par::Rayon).is_nan(),
            "parallel nrm2 must propagate NaN"
        );
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_nrm2_par_inf_yields_positive_infinity() {
        // A single infinity in a large vector yields +Inf, not NaN.
        let n = 100_000;
        let mut x = vec![1.0f64; n];
        x[70_000] = f64::INFINITY;
        let result = nrm2_par(&x, Par::Rayon);
        assert!(
            result.is_infinite() && result > 0.0,
            "parallel nrm2 of Inf must be +Inf, got {}",
            result
        );

        // Infinities in two different chunks must still give +Inf (not Inf/Inf).
        let mut x2 = vec![1.0f64; n];
        x2[1_000] = f64::INFINITY;
        x2[90_000] = f64::NEG_INFINITY;
        let result2 = nrm2_par(&x2, Par::Rayon);
        assert!(
            result2.is_infinite() && result2 > 0.0,
            "parallel nrm2 of two infinities must be +Inf, got {}",
            result2
        );

        // Inf together with NaN: NaN wins.
        let mut x3 = vec![1.0f64; n];
        x3[2_000] = f64::INFINITY;
        x3[95_000] = f64::NAN;
        assert!(
            nrm2_par(&x3, Par::Rayon).is_nan(),
            "parallel nrm2 of Inf+NaN must be NaN"
        );
    }
}
