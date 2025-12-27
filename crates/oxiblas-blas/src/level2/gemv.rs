//! GEMV: General matrix-vector multiplication.
//!
//! Computes y = α·op(A)·x + β·y where op(A) is A or A^T.
//!
//! ## Blocked Algorithm
//!
//! For large matrices, this module uses a blocked algorithm that improves
//! cache utilization by processing the matrix in blocks that fit in L1/L2 cache.
//! Block sizes are tuned for modern CPU cache hierarchies.
//!
//! ## Parallelization
//!
//! When the `parallel` feature is enabled, GEMV operations are parallelized:
//! - For `NoTrans`: parallelizes over output rows
//! - For Trans/ConjTrans: uses thread-local accumulators with reduction

use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::Field;
use oxiblas_matrix::MatRef;

/// Block size for the column dimension (K blocking).
/// Tuned for L1 cache (32KB typically holds ~4K doubles).
/// Increased to 512 for better vector reuse in x.
const KC_BLOCK: usize = 512;

/// Block size for the row dimension (M blocking).
/// Tuned for L2 cache and to enable register blocking.
/// Increased to 128 for better output locality.
const MC_BLOCK: usize = 128;

#[cfg(feature = "parallel")]
use oxiblas_core::parallel::ParThreshold;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Transpose operation for matrix-vector multiply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemvTrans {
    /// No transpose: y = α·A·x + β·y
    NoTrans,
    /// Transpose: y = α·A^T·x + β·y
    Trans,
    /// Conjugate transpose: y = α·A^H·x + β·y (for complex)
    ConjTrans,
}

/// Computes the general matrix-vector product.
///
/// y = α·op(A)·x + β·y
///
/// where op(A) is A, A^T, or A^H depending on the trans parameter.
///
/// # Arguments
///
/// * `trans` - Whether to transpose A
/// * `alpha` - Scalar multiplier for A·x
/// * `a` - The matrix A (m×n for `NoTrans`, n×m for Trans)
/// * `x` - The input vector
/// * `beta` - Scalar multiplier for y
/// * `y` - The output vector (modified in place)
///
/// # Panics
///
/// Panics if dimensions don't match.
///
/// # Example
///
/// ```
/// use oxiblas_blas::level2::{gemv, GemvTrans};
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[1.0f64, 2.0, 3.0],
///     &[4.0, 5.0, 6.0],
/// ]);
/// let x = [1.0f64, 2.0, 3.0];
/// let mut y = [0.0f64, 0.0];
///
/// // y = 1.0 * A * x + 0.0 * y
/// gemv(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, 0.0, &mut y);
///
/// // y[0] = 1*1 + 2*2 + 3*3 = 14
/// // y[1] = 4*1 + 5*2 + 6*3 = 32
/// assert!((y[0] - 14.0).abs() < 1e-10);
/// assert!((y[1] - 32.0).abs() < 1e-10);
/// ```
pub fn gemv<T: Field>(trans: GemvTrans, alpha: T, a: MatRef<'_, T>, x: &[T], beta: T, y: &mut [T]) {
    let (m, n) = (a.nrows(), a.ncols());

    // Determine effective dimensions based on transpose
    let (rows, cols) = match trans {
        GemvTrans::NoTrans => (m, n),
        GemvTrans::Trans | GemvTrans::ConjTrans => (n, m),
    };

    assert_eq!(
        x.len(),
        cols,
        "x length must match number of columns of op(A)"
    );
    assert_eq!(y.len(), rows, "y length must match number of rows of op(A)");

    // Handle beta = 0 case (avoid multiplying uninitialized values)
    if beta == T::zero() {
        for yi in y.iter_mut() {
            *yi = T::zero();
        }
    } else if beta != T::one() {
        for yi in y.iter_mut() {
            *yi = beta * *yi;
        }
    }

    // Early exit if alpha is zero
    if alpha == T::zero() {
        return;
    }

    // Use blocked algorithm for large matrices
    if m * n > 4096 {
        gemv_blocked(trans, alpha, a, x, y, m, n);
    } else {
        gemv_unblocked(trans, alpha, a, x, y, m, n);
    }
}

/// Unblocked GEMV for small matrices.
#[inline]
fn gemv_unblocked<T: Field>(
    trans: GemvTrans,
    alpha: T,
    a: MatRef<'_, T>,
    x: &[T],
    y: &mut [T],
    m: usize,
    n: usize,
) {
    match trans {
        GemvTrans::NoTrans => {
            // y = α·A·x + β·y (beta already applied)
            for i in 0..m {
                let mut sum = T::zero();
                // Unroll by 4
                let chunks = n / 4;
                let remainder = n % 4;

                for j in 0..chunks {
                    let base = j * 4;
                    sum += a[(i, base)] * x[base];
                    sum += a[(i, base + 1)] * x[base + 1];
                    sum += a[(i, base + 2)] * x[base + 2];
                    sum += a[(i, base + 3)] * x[base + 3];
                }

                let base = chunks * 4;
                for j in 0..remainder {
                    sum += a[(i, base + j)] * x[base + j];
                }

                y[i] += alpha * sum;
            }
        }
        GemvTrans::Trans => {
            // y = α·A^T·x + β·y
            for i in 0..m {
                let alpha_xi = alpha * x[i];
                for j in 0..n {
                    y[j] += a[(i, j)] * alpha_xi;
                }
            }
        }
        GemvTrans::ConjTrans => {
            // y = α·A^H·x + β·y
            for i in 0..m {
                let alpha_xi = alpha * x[i];
                for j in 0..n {
                    y[j] += a[(i, j)].conj() * alpha_xi;
                }
            }
        }
    }
}

/// Blocked GEMV for large matrices.
///
/// Uses cache-aware blocking to improve performance on large matrices.
/// The algorithm processes the matrix in blocks that fit in L1/L2 cache.
fn gemv_blocked<T: Field>(
    trans: GemvTrans,
    alpha: T,
    a: MatRef<'_, T>,
    x: &[T],
    y: &mut [T],
    m: usize,
    n: usize,
) {
    match trans {
        GemvTrans::NoTrans => {
            gemv_blocked_notrans(alpha, a, x, y, m, n);
        }
        GemvTrans::Trans => {
            // For transpose, block over rows (input x dimension)
            // Use local accumulators for each column block to reduce memory traffic
            for ib in (0..m).step_by(MC_BLOCK) {
                let im = MC_BLOCK.min(m - ib);

                for jb in (0..n).step_by(KC_BLOCK) {
                    let jn = KC_BLOCK.min(n - jb);

                    // Process block with 4-way unrolling on j
                    for i in 0..im {
                        let row_idx = ib + i;
                        let alpha_xi = alpha * x[row_idx];

                        let chunks4 = jn / 4;
                        let remainder4 = jn % 4;

                        for j in 0..chunks4 {
                            let base = jb + j * 4;
                            y[base] += a[(row_idx, base)] * alpha_xi;
                            y[base + 1] += a[(row_idx, base + 1)] * alpha_xi;
                            y[base + 2] += a[(row_idx, base + 2)] * alpha_xi;
                            y[base + 3] += a[(row_idx, base + 3)] * alpha_xi;
                        }

                        let base = jb + chunks4 * 4;
                        for j in 0..remainder4 {
                            y[base + j] += a[(row_idx, base + j)] * alpha_xi;
                        }
                    }
                }
            }
        }
        GemvTrans::ConjTrans => {
            // Same as Trans but with conjugate
            for ib in (0..m).step_by(MC_BLOCK) {
                let im = MC_BLOCK.min(m - ib);

                for jb in (0..n).step_by(KC_BLOCK) {
                    let jn = KC_BLOCK.min(n - jb);

                    for i in 0..im {
                        let row_idx = ib + i;
                        let alpha_xi = alpha * x[row_idx];

                        let chunks4 = jn / 4;
                        let remainder4 = jn % 4;

                        for j in 0..chunks4 {
                            let base = jb + j * 4;
                            y[base] += a[(row_idx, base)].conj() * alpha_xi;
                            y[base + 1] += a[(row_idx, base + 1)].conj() * alpha_xi;
                            y[base + 2] += a[(row_idx, base + 2)].conj() * alpha_xi;
                            y[base + 3] += a[(row_idx, base + 3)].conj() * alpha_xi;
                        }

                        let base = jb + chunks4 * 4;
                        for j in 0..remainder4 {
                            y[base + j] += a[(row_idx, base + j)].conj() * alpha_xi;
                        }
                    }
                }
            }
        }
    }
}

/// SIMD-optimized blocked GEMV for `NoTrans` case.
#[inline]
fn gemv_blocked_notrans<T: Field>(
    alpha: T,
    a: MatRef<'_, T>,
    x: &[T],
    y: &mut [T],
    m: usize,
    n: usize,
) {
    // Try SIMD path for f64
    #[cfg(target_arch = "aarch64")]
    {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            unsafe {
                let a_ptr = a.as_ptr().cast::<f64>();
                let x_ptr = x.as_ptr().cast::<f64>();
                let y_ptr = y.as_mut_ptr().cast::<f64>();
                let alpha_f64 = *(&raw const alpha).cast::<f64>();
                let row_stride = a.row_stride();
                gemv_blocked_notrans_f64_neon(alpha_f64, a_ptr, row_stride, x_ptr, y_ptr, m, n);
            }
            return;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>()
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            unsafe {
                let a_ptr = a.as_ptr() as *const f64;
                let x_ptr = x.as_ptr() as *const f64;
                let y_ptr = y.as_mut_ptr() as *mut f64;
                let alpha_f64 = *(&alpha as *const T as *const f64);
                let row_stride = a.row_stride();
                gemv_blocked_notrans_f64_avx2(alpha_f64, a_ptr, row_stride, x_ptr, y_ptr, m, n);
            }
            return;
        }
    }

    // Generic fallback
    gemv_blocked_notrans_generic(alpha, a, x, y, m, n);
}

/// Generic blocked GEMV for `NoTrans`.
#[inline]
fn gemv_blocked_notrans_generic<T: Field>(
    alpha: T,
    a: MatRef<'_, T>,
    x: &[T],
    y: &mut [T],
    m: usize,
    n: usize,
) {
    for jb in (0..n).step_by(KC_BLOCK) {
        let jn = KC_BLOCK.min(n - jb);

        for ib in (0..m).step_by(MC_BLOCK) {
            let im = MC_BLOCK.min(m - ib);

            for i in 0..im {
                let row_idx = ib + i;
                let mut sum = T::zero();

                let chunks8 = jn / 8;
                let remainder8 = jn % 8;

                for j in 0..chunks8 {
                    let base = jb + j * 8;
                    sum += a[(row_idx, base)] * x[base];
                    sum += a[(row_idx, base + 1)] * x[base + 1];
                    sum += a[(row_idx, base + 2)] * x[base + 2];
                    sum += a[(row_idx, base + 3)] * x[base + 3];
                    sum += a[(row_idx, base + 4)] * x[base + 4];
                    sum += a[(row_idx, base + 5)] * x[base + 5];
                    sum += a[(row_idx, base + 6)] * x[base + 6];
                    sum += a[(row_idx, base + 7)] * x[base + 7];
                }

                let base = jb + chunks8 * 8;
                for j in 0..remainder8 {
                    sum += a[(row_idx, base + j)] * x[base + j];
                }

                y[row_idx] += alpha * sum;
            }
        }
    }
}

/// NEON-optimized GEMV for f64.
///
/// For column-major matrix A with leading dimension (`row_stride)`:
/// A[row, col] = a[row + col * `row_stride`]
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn gemv_blocked_notrans_f64_neon(
    alpha: f64,
    a: *const f64,
    row_stride: usize,
    x: *const f64,
    y: *mut f64,
    m: usize,
    n: usize,
) {
    use core::arch::aarch64::{vdupq_n_f64, vfmaq_laneq_f64, vgetq_lane_f64, vld1q_f64};

    for jb in (0..n).step_by(KC_BLOCK) {
        let jn = KC_BLOCK.min(n - jb);

        for ib in (0..m).step_by(MC_BLOCK) {
            let im = MC_BLOCK.min(m - ib);

            // Process 4 rows at a time for better register utilization
            let mut i = 0;
            while i + 4 <= im {
                let row0 = ib + i;
                let row1 = ib + i + 1;
                let row2 = ib + i + 2;
                let row3 = ib + i + 3;

                let mut sum0 = vdupq_n_f64(0.0);
                let mut sum1 = vdupq_n_f64(0.0);
                let mut sum2 = vdupq_n_f64(0.0);
                let mut sum3 = vdupq_n_f64(0.0);

                // Process column by column - for column-major, each column is contiguous
                for j in 0..jn {
                    let col = jb + j;
                    let xj = vdupq_n_f64(*x.add(col));
                    let col_ptr = a.add(col * row_stride);

                    // Load 4 consecutive elements from this column (rows row0..row3)
                    let a_vec = vld1q_f64(col_ptr.add(row0));
                    let a_vec2 = vld1q_f64(col_ptr.add(row2));

                    // Extract individual values and accumulate
                    sum0 = vfmaq_laneq_f64::<0>(sum0, xj, a_vec);
                    sum1 = vfmaq_laneq_f64::<1>(sum1, xj, a_vec);
                    sum2 = vfmaq_laneq_f64::<0>(sum2, xj, a_vec2);
                    sum3 = vfmaq_laneq_f64::<1>(sum3, xj, a_vec2);
                }

                // Store results
                *y.add(row0) += alpha * vgetq_lane_f64::<0>(sum0);
                *y.add(row1) += alpha * vgetq_lane_f64::<0>(sum1);
                *y.add(row2) += alpha * vgetq_lane_f64::<0>(sum2);
                *y.add(row3) += alpha * vgetq_lane_f64::<0>(sum3);

                i += 4;
            }

            // Handle remaining rows one at a time
            while i < im {
                let row = ib + i;
                let mut sum = 0.0;

                for j in 0..jn {
                    let col = jb + j;
                    sum += *a.add(row + col * row_stride) * *x.add(col);
                }

                *y.add(row) += alpha * sum;
                i += 1;
            }
        }
    }
}

/// AVX2-optimized GEMV for f64.
///
/// For column-major matrix A with leading dimension (row_stride):
/// A[row, col] = a[row + col * row_stride]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn gemv_blocked_notrans_f64_avx2(
    alpha: f64,
    a: *const f64,
    row_stride: usize,
    x: *const f64,
    y: *mut f64,
    m: usize,
    n: usize,
) {
    use core::arch::x86_64::*;

    for jb in (0..n).step_by(KC_BLOCK) {
        let jn = KC_BLOCK.min(n - jb);

        for ib in (0..m).step_by(MC_BLOCK) {
            let im = MC_BLOCK.min(m - ib);

            // Process 4 rows at a time (one AVX register)
            let mut i = 0;
            while i + 4 <= im {
                let row0 = ib + i;

                let mut sum = _mm256_setzero_pd();

                // Process columns
                for j in 0..jn {
                    let col = jb + j;
                    let xj = _mm256_set1_pd(*x.add(col));
                    let col_ptr = a.add(col * row_stride);

                    // Load 4 consecutive elements from this column
                    let a_vec = _mm256_loadu_pd(col_ptr.add(row0));
                    sum = _mm256_fmadd_pd(a_vec, xj, sum);
                }

                // Load current y values, add scaled sum, store back
                let y_ptr = y.add(row0);
                let y_vec = _mm256_loadu_pd(y_ptr);
                let alpha_vec = _mm256_set1_pd(alpha);
                let result = _mm256_fmadd_pd(sum, alpha_vec, y_vec);
                _mm256_storeu_pd(y_ptr, result);

                i += 4;
            }

            // Handle remaining rows one at a time
            while i < im {
                let row = ib + i;
                let mut sum = 0.0;

                for j in 0..jn {
                    let col = jb + j;
                    sum += *a.add(row + col * row_stride) * *x.add(col);
                }

                *y.add(row) += alpha * sum;
                i += 1;
            }
        }
    }
}

/// Computes GEMV with parallelization control.
///
/// When the `parallel` feature is enabled and `par` is set to `Par::Rayon`,
/// this function will parallelize the computation.
pub fn gemv_with_par<T: Field + Send + Sync>(
    trans: GemvTrans,
    alpha: T,
    a: MatRef<'_, T>,
    x: &[T],
    beta: T,
    y: &mut [T],
    par: Par,
) {
    let (m, n) = (a.nrows(), a.ncols());

    // Determine effective dimensions based on transpose
    let (rows, cols) = match trans {
        GemvTrans::NoTrans => (m, n),
        GemvTrans::Trans | GemvTrans::ConjTrans => (n, m),
    };

    assert_eq!(
        x.len(),
        cols,
        "x length must match number of columns of op(A)"
    );
    assert_eq!(y.len(), rows, "y length must match number of rows of op(A)");

    // Suppress unused warning when parallel feature is disabled
    let _ = &par;

    #[cfg(feature = "parallel")]
    {
        // Check if we should parallelize
        let threshold = ParThreshold::new(256 * 256, 256);
        let total_work = rows * cols;

        if threshold.should_parallelize(total_work, par) {
            gemv_parallel(trans, alpha, a, x, beta, y, par);
            return;
        }
    }

    // Sequential fallback
    gemv_sequential(trans, alpha, a, x, beta, y);
}

/// Sequential GEMV implementation.
fn gemv_sequential<T: Field>(
    trans: GemvTrans,
    alpha: T,
    a: MatRef<'_, T>,
    x: &[T],
    beta: T,
    y: &mut [T],
) {
    let (m, n) = (a.nrows(), a.ncols());

    // Handle beta = 0 case (avoid multiplying uninitialized values)
    if beta == T::zero() {
        for yi in y.iter_mut() {
            *yi = T::zero();
        }
    } else if beta != T::one() {
        for yi in y.iter_mut() {
            *yi = beta * *yi;
        }
    }

    // Early exit if alpha is zero
    if alpha == T::zero() {
        return;
    }

    // Use blocked algorithm for large matrices
    if m * n > 4096 {
        gemv_blocked(trans, alpha, a, x, y, m, n);
    } else {
        gemv_unblocked(trans, alpha, a, x, y, m, n);
    }
}

/// Parallel GEMV implementation.
#[cfg(feature = "parallel")]
fn gemv_parallel<T: Field + Send + Sync>(
    trans: GemvTrans,
    alpha: T,
    a: MatRef<'_, T>,
    x: &[T],
    beta: T,
    y: &mut [T],
    par: Par,
) {
    let (m, n) = (a.nrows(), a.ncols());
    let (rows, _cols) = match trans {
        GemvTrans::NoTrans => (m, n),
        GemvTrans::Trans | GemvTrans::ConjTrans => (n, m),
    };

    // Parallelize beta scaling for large vectors
    if rows > 4096 {
        if beta == T::zero() {
            y.par_chunks_mut(4096).for_each(|chunk| {
                for yi in chunk.iter_mut() {
                    *yi = T::zero();
                }
            });
        } else if beta != T::one() {
            y.par_chunks_mut(4096).for_each(|chunk| {
                for yi in chunk.iter_mut() {
                    *yi = beta * *yi;
                }
            });
        }
    } else {
        // Sequential for small vectors
        if beta == T::zero() {
            for yi in y.iter_mut() {
                *yi = T::zero();
            }
        } else if beta != T::one() {
            for yi in y.iter_mut() {
                *yi = beta * *yi;
            }
        }
    }

    // Early exit if alpha is zero
    if alpha == T::zero() {
        return;
    }

    match trans {
        GemvTrans::NoTrans => {
            gemv_parallel_notrans(alpha, a, x, y, m, n, par);
        }
        GemvTrans::Trans => {
            gemv_parallel_trans(alpha, a, x, y, m, n, par);
        }
        GemvTrans::ConjTrans => {
            gemv_parallel_conjtrans(alpha, a, x, y, m, n, par);
        }
    }
}

/// Parallel NoTrans GEMV with 8-way unrolling.
#[cfg(feature = "parallel")]
fn gemv_parallel_notrans<T: Field + Send + Sync>(
    alpha: T,
    a: MatRef<'_, T>,
    x: &[T],
    y: &mut [T],
    m: usize,
    n: usize,
    par: Par,
) {
    use oxiblas_core::parallel::partition_work;

    let y_ptr = y.as_mut_ptr() as usize;
    let num_threads = par.num_threads().min(m);
    let work_ranges = partition_work(m, num_threads);

    work_ranges.par_iter().for_each(|range| {
        // Use cache-aware blocking within each thread's work
        for ib in (range.start..range.end).step_by(MC_BLOCK) {
            let i_end = (ib + MC_BLOCK).min(range.end);

            for kb in (0..n).step_by(KC_BLOCK) {
                let k_end = (kb + KC_BLOCK).min(n);

                for i in ib..i_end {
                    let mut sum = T::zero();
                    let block_n = k_end - kb;
                    let chunks8 = block_n / 8;
                    let remainder = block_n % 8;

                    // 8-way unrolled inner loop
                    for j in 0..chunks8 {
                        let base = kb + j * 8;
                        sum = sum + a[(i, base)] * x[base];
                        sum = sum + a[(i, base + 1)] * x[base + 1];
                        sum = sum + a[(i, base + 2)] * x[base + 2];
                        sum = sum + a[(i, base + 3)] * x[base + 3];
                        sum = sum + a[(i, base + 4)] * x[base + 4];
                        sum = sum + a[(i, base + 5)] * x[base + 5];
                        sum = sum + a[(i, base + 6)] * x[base + 6];
                        sum = sum + a[(i, base + 7)] * x[base + 7];
                    }

                    let base = kb + chunks8 * 8;
                    for j in 0..remainder {
                        sum = sum + a[(i, base + j)] * x[base + j];
                    }

                    // SAFETY: Each thread writes to disjoint rows
                    unsafe {
                        let y_ptr = y_ptr as *mut T;
                        let yi = &mut *y_ptr.add(i);
                        *yi = *yi + alpha * sum;
                    }
                }
            }
        }
    });
}

/// Parallel Trans GEMV with tree reduction.
#[cfg(feature = "parallel")]
fn gemv_parallel_trans<T: Field + Send + Sync>(
    alpha: T,
    a: MatRef<'_, T>,
    x: &[T],
    y: &mut [T],
    m: usize,
    n: usize,
    par: Par,
) {
    use oxiblas_core::parallel::partition_work;

    let num_threads = par.num_threads().min(m);
    let work_ranges = partition_work(m, num_threads);

    // Collect partial results from each thread with 4-way unrolled inner loop
    let partial_results: Vec<Vec<T>> = work_ranges
        .par_iter()
        .map(|range| {
            let mut partial_y = vec![T::zero(); n];

            // Process rows in blocks for better cache behavior
            for ib in (range.start..range.end).step_by(64) {
                let i_end = (ib + 64).min(range.end);

                for i in ib..i_end {
                    let xi = x[i];
                    if xi == T::zero() {
                        continue;
                    }
                    let alpha_xi = alpha * xi;

                    // 4-way unrolled inner loop
                    let chunks4 = n / 4;
                    let remainder = n % 4;

                    for j in 0..chunks4 {
                        let base = j * 4;
                        partial_y[base] = partial_y[base] + a[(i, base)] * alpha_xi;
                        partial_y[base + 1] = partial_y[base + 1] + a[(i, base + 1)] * alpha_xi;
                        partial_y[base + 2] = partial_y[base + 2] + a[(i, base + 2)] * alpha_xi;
                        partial_y[base + 3] = partial_y[base + 3] + a[(i, base + 3)] * alpha_xi;
                    }

                    let base = chunks4 * 4;
                    for j in 0..remainder {
                        partial_y[base + j] = partial_y[base + j] + a[(i, base + j)] * alpha_xi;
                    }
                }
            }

            partial_y
        })
        .collect();

    // Tree-based parallel reduction of partial results
    if partial_results.len() > 1 {
        // Reduce in parallel chunks
        let chunk_size = 4096.max(n / num_threads);
        let y_ptr = y.as_mut_ptr() as usize;

        (0..n)
            .into_par_iter()
            .step_by(chunk_size)
            .for_each(|start| {
                let end = (start + chunk_size).min(n);
                unsafe {
                    let y_ptr = y_ptr as *mut T;
                    for j in start..end {
                        let mut sum = *y_ptr.add(j);
                        for partial in &partial_results {
                            sum = sum + partial[j];
                        }
                        *y_ptr.add(j) = sum;
                    }
                }
            });
    } else if !partial_results.is_empty() {
        // Single thread result - just add directly
        for j in 0..n {
            y[j] = y[j] + partial_results[0][j];
        }
    }
}

/// Parallel ConjTrans GEMV with tree reduction.
#[cfg(feature = "parallel")]
fn gemv_parallel_conjtrans<T: Field + Send + Sync>(
    alpha: T,
    a: MatRef<'_, T>,
    x: &[T],
    y: &mut [T],
    m: usize,
    n: usize,
    par: Par,
) {
    use oxiblas_core::parallel::partition_work;

    let num_threads = par.num_threads().min(m);
    let work_ranges = partition_work(m, num_threads);

    // Collect partial results from each thread with conjugate
    let partial_results: Vec<Vec<T>> = work_ranges
        .par_iter()
        .map(|range| {
            let mut partial_y = vec![T::zero(); n];

            // Process rows in blocks for better cache behavior
            for ib in (range.start..range.end).step_by(64) {
                let i_end = (ib + 64).min(range.end);

                for i in ib..i_end {
                    let xi = x[i];
                    if xi == T::zero() {
                        continue;
                    }
                    let alpha_xi = alpha * xi;

                    // 4-way unrolled inner loop with conjugate
                    let chunks4 = n / 4;
                    let remainder = n % 4;

                    for j in 0..chunks4 {
                        let base = j * 4;
                        partial_y[base] = partial_y[base] + a[(i, base)].conj() * alpha_xi;
                        partial_y[base + 1] =
                            partial_y[base + 1] + a[(i, base + 1)].conj() * alpha_xi;
                        partial_y[base + 2] =
                            partial_y[base + 2] + a[(i, base + 2)].conj() * alpha_xi;
                        partial_y[base + 3] =
                            partial_y[base + 3] + a[(i, base + 3)].conj() * alpha_xi;
                    }

                    let base = chunks4 * 4;
                    for j in 0..remainder {
                        partial_y[base + j] =
                            partial_y[base + j] + a[(i, base + j)].conj() * alpha_xi;
                    }
                }
            }

            partial_y
        })
        .collect();

    // Tree-based parallel reduction of partial results
    if partial_results.len() > 1 {
        // Reduce in parallel chunks
        let chunk_size = 4096.max(n / num_threads);
        let y_ptr = y.as_mut_ptr() as usize;

        (0..n)
            .into_par_iter()
            .step_by(chunk_size)
            .for_each(|start| {
                let end = (start + chunk_size).min(n);
                unsafe {
                    let y_ptr = y_ptr as *mut T;
                    for j in start..end {
                        let mut sum = *y_ptr.add(j);
                        for partial in &partial_results {
                            sum = sum + partial[j];
                        }
                        *y_ptr.add(j) = sum;
                    }
                }
            });
    } else if !partial_results.is_empty() {
        // Single thread result - just add directly
        for j in 0..n {
            y[j] = y[j] + partial_results[0][j];
        }
    }
}

/// Computes y = A·x (simplified interface).
///
/// # Example
///
/// ```
/// use oxiblas_blas::level2::gemv_simple;
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[1.0f64, 2.0],
///     &[3.0, 4.0],
/// ]);
/// let x = [1.0f64, 2.0];
///
/// let y = gemv_simple(a.as_ref(), &x);
///
/// // y[0] = 1*1 + 2*2 = 5
/// // y[1] = 3*1 + 4*2 = 11
/// assert!((y[0] - 5.0).abs() < 1e-10);
/// assert!((y[1] - 11.0).abs() < 1e-10);
/// ```
pub fn gemv_simple<T: Field>(a: MatRef<'_, T>, x: &[T]) -> Vec<T> {
    let m = a.nrows();
    let mut y = vec![T::zero(); m];
    gemv(GemvTrans::NoTrans, T::one(), a, x, T::zero(), &mut y);
    y
}

// =============================================================================
// Fused GEMV operations
// =============================================================================

/// Fused GEMV + vector addition: y = α·op(A)·x + z
///
/// This is a fused operation that computes the matrix-vector product and adds
/// a vector in a single pass, reducing memory bandwidth requirements compared
/// to calling GEMV followed by AXPY.
///
/// # Arguments
///
/// * `trans` - Whether to transpose A
/// * `alpha` - Scalar multiplier for A·x
/// * `a` - The matrix A
/// * `x` - The input vector for multiplication
/// * `z` - The vector to add (consumed)
///
/// # Returns
///
/// The result vector y = α·op(A)·x + z
///
/// # Example
///
/// ```
/// use oxiblas_blas::level2::{gemv_add, GemvTrans};
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[1.0f64, 2.0],
///     &[3.0, 4.0],
/// ]);
/// let x = [1.0f64, 1.0];
/// let z = vec![10.0f64, 20.0];
///
/// // result = 1.0 * A * x + z = [3, 7] + [10, 20] = [13, 27]
/// let result = gemv_add(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, z);
///
/// assert!((result[0] - 13.0).abs() < 1e-10);
/// assert!((result[1] - 27.0).abs() < 1e-10);
/// ```
pub fn gemv_add<T: Field>(
    trans: GemvTrans,
    alpha: T,
    a: MatRef<'_, T>,
    x: &[T],
    mut z: Vec<T>,
) -> Vec<T> {
    let (m, n) = (a.nrows(), a.ncols());

    let (rows, cols) = match trans {
        GemvTrans::NoTrans => (m, n),
        GemvTrans::Trans | GemvTrans::ConjTrans => (n, m),
    };

    assert_eq!(x.len(), cols, "x length must match columns of op(A)");
    assert_eq!(z.len(), rows, "z length must match rows of op(A)");

    // Early exit if alpha is zero
    if alpha == T::zero() {
        return z;
    }

    // Fused operation: compute y = α·A·x + z in place
    gemv_add_inplace(trans, alpha, a, x, &mut z, m, n);

    z
}

/// In-place fused GEMV + vector addition: y += α·op(A)·x
///
/// Adds α·op(A)·x to the existing contents of y.
///
/// # Example
///
/// ```
/// use oxiblas_blas::level2::{gemv_add_inplace, GemvTrans};
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[1.0f64, 2.0],
///     &[3.0, 4.0],
/// ]);
/// let x = [1.0f64, 1.0];
/// let mut y = vec![10.0f64, 20.0];
///
/// // y += 1.0 * A * x -> y = [10, 20] + [3, 7] = [13, 27]
/// gemv_add_inplace(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, &mut y, 2, 2);
///
/// assert!((y[0] - 13.0).abs() < 1e-10);
/// assert!((y[1] - 27.0).abs() < 1e-10);
/// ```
pub fn gemv_add_inplace<T: Field>(
    trans: GemvTrans,
    alpha: T,
    a: MatRef<'_, T>,
    x: &[T],
    y: &mut [T],
    m: usize,
    n: usize,
) {
    if alpha == T::zero() {
        return;
    }

    match trans {
        GemvTrans::NoTrans => {
            // Use blocked approach for large matrices
            if m * n > 4096 {
                gemv_add_blocked_notrans(alpha, a, x, y, m, n);
            } else {
                gemv_add_unblocked_notrans(alpha, a, x, y, m, n);
            }
        }
        GemvTrans::Trans => {
            if m * n > 4096 {
                gemv_add_blocked_trans(alpha, a, x, y, m, n);
            } else {
                gemv_add_unblocked_trans(alpha, a, x, y, m, n);
            }
        }
        GemvTrans::ConjTrans => {
            if m * n > 4096 {
                gemv_add_blocked_conjtrans(alpha, a, x, y, m, n);
            } else {
                gemv_add_unblocked_conjtrans(alpha, a, x, y, m, n);
            }
        }
    }
}

/// Unblocked fused GEMV+add for `NoTrans`.
#[inline]
fn gemv_add_unblocked_notrans<T: Field>(
    alpha: T,
    a: MatRef<'_, T>,
    x: &[T],
    y: &mut [T],
    m: usize,
    n: usize,
) {
    for i in 0..m {
        let mut sum = T::zero();
        // 4-way unrolling
        let chunks = n / 4;
        let remainder = n % 4;

        for j in 0..chunks {
            let base = j * 4;
            sum += a[(i, base)] * x[base];
            sum += a[(i, base + 1)] * x[base + 1];
            sum += a[(i, base + 2)] * x[base + 2];
            sum += a[(i, base + 3)] * x[base + 3];
        }

        let base = chunks * 4;
        for j in 0..remainder {
            sum += a[(i, base + j)] * x[base + j];
        }

        y[i] += alpha * sum;
    }
}

/// Blocked fused GEMV+add for `NoTrans`.
#[inline]
fn gemv_add_blocked_notrans<T: Field>(
    alpha: T,
    a: MatRef<'_, T>,
    x: &[T],
    y: &mut [T],
    m: usize,
    n: usize,
) {
    // Process in blocks for better cache utilization
    for mc in (0..m).step_by(MC_BLOCK) {
        let mc_end = (mc + MC_BLOCK).min(m);

        for kc in (0..n).step_by(KC_BLOCK) {
            let kc_end = (kc + KC_BLOCK).min(n);

            for i in mc..mc_end {
                let mut sum = T::zero();
                // 8-way unrolling within block
                let block_size = kc_end - kc;
                let chunks = block_size / 8;
                let remainder = block_size % 8;

                for j in 0..chunks {
                    let base = kc + j * 8;
                    sum += a[(i, base)] * x[base];
                    sum += a[(i, base + 1)] * x[base + 1];
                    sum += a[(i, base + 2)] * x[base + 2];
                    sum += a[(i, base + 3)] * x[base + 3];
                    sum += a[(i, base + 4)] * x[base + 4];
                    sum += a[(i, base + 5)] * x[base + 5];
                    sum += a[(i, base + 6)] * x[base + 6];
                    sum += a[(i, base + 7)] * x[base + 7];
                }

                let base = kc + chunks * 8;
                for j in 0..remainder {
                    sum += a[(i, base + j)] * x[base + j];
                }

                y[i] += alpha * sum;
            }
        }
    }
}

/// Unblocked fused GEMV+add for Trans.
#[inline]
fn gemv_add_unblocked_trans<T: Field>(
    alpha: T,
    a: MatRef<'_, T>,
    x: &[T],
    y: &mut [T],
    m: usize,
    n: usize,
) {
    for i in 0..m {
        let xi = x[i];
        if xi != T::zero() {
            let axi = alpha * xi;
            for j in 0..n {
                y[j] += a[(i, j)] * axi;
            }
        }
    }
}

/// Blocked fused GEMV+add for Trans.
#[inline]
fn gemv_add_blocked_trans<T: Field>(
    alpha: T,
    a: MatRef<'_, T>,
    x: &[T],
    y: &mut [T],
    m: usize,
    n: usize,
) {
    for kc in (0..m).step_by(KC_BLOCK) {
        let kc_end = (kc + KC_BLOCK).min(m);

        for mc in (0..n).step_by(MC_BLOCK) {
            let mc_end = (mc + MC_BLOCK).min(n);

            for i in kc..kc_end {
                let xi = x[i];
                if xi != T::zero() {
                    let axi = alpha * xi;
                    for j in mc..mc_end {
                        y[j] += a[(i, j)] * axi;
                    }
                }
            }
        }
    }
}

/// Unblocked fused GEMV+add for `ConjTrans`.
#[inline]
fn gemv_add_unblocked_conjtrans<T: Field>(
    alpha: T,
    a: MatRef<'_, T>,
    x: &[T],
    y: &mut [T],
    m: usize,
    n: usize,
) {
    for i in 0..m {
        let xi = x[i];
        if xi != T::zero() {
            let axi = alpha * xi;
            for j in 0..n {
                y[j] += a[(i, j)].conj() * axi;
            }
        }
    }
}

/// Blocked fused GEMV+add for `ConjTrans`.
#[inline]
fn gemv_add_blocked_conjtrans<T: Field>(
    alpha: T,
    a: MatRef<'_, T>,
    x: &[T],
    y: &mut [T],
    m: usize,
    n: usize,
) {
    for kc in (0..m).step_by(KC_BLOCK) {
        let kc_end = (kc + KC_BLOCK).min(m);

        for mc in (0..n).step_by(MC_BLOCK) {
            let mc_end = (mc + MC_BLOCK).min(n);

            for i in kc..kc_end {
                let xi = x[i];
                if xi != T::zero() {
                    let axi = alpha * xi;
                    for j in mc..mc_end {
                        y[j] += a[(i, j)].conj() * axi;
                    }
                }
            }
        }
    }
}

/// Fused operation: y = α·A·x + β·B·z
///
/// Computes the sum of two GEMV operations in a single function call.
/// This can be more efficient than two separate GEMV calls when the output
/// vectors would otherwise need to be combined.
///
/// # Arguments
///
/// * `alpha` - Scalar for first GEMV
/// * `a` - First matrix
/// * `x` - First input vector
/// * `beta` - Scalar for second GEMV
/// * `b` - Second matrix
/// * `z` - Second input vector
///
/// # Returns
///
/// The result vector y = α·A·x + β·B·z
pub fn gemv_sum2<T: Field>(
    alpha: T,
    a: MatRef<'_, T>,
    x: &[T],
    beta: T,
    b: MatRef<'_, T>,
    z: &[T],
) -> Vec<T> {
    let m = a.nrows();
    let n1 = a.ncols();
    let n2 = b.ncols();

    assert_eq!(
        a.nrows(),
        b.nrows(),
        "Matrices must have same number of rows"
    );
    assert_eq!(x.len(), n1, "x length must match columns of A");
    assert_eq!(z.len(), n2, "z length must match columns of B");

    let mut y = vec![T::zero(); m];

    // Compute y = α·A·x
    if alpha != T::zero() {
        gemv_add_inplace(GemvTrans::NoTrans, alpha, a, x, &mut y, m, n1);
    }

    // Add y += β·B·z
    if beta != T::zero() {
        gemv_add_inplace(GemvTrans::NoTrans, beta, b, z, &mut y, m, n2);
    }

    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxiblas_matrix::Mat;

    #[test]
    fn test_gemv_basic() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let x = [1.0, 2.0, 3.0];
        let mut y = [0.0, 0.0];

        gemv(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, 0.0, &mut y);

        // y[0] = 1*1 + 2*2 + 3*3 = 14
        // y[1] = 4*1 + 5*2 + 6*3 = 32
        assert!((y[0] - 14.0).abs() < 1e-10);
        assert!((y[1] - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemv_with_alpha_beta() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let x = [1.0, 1.0];
        let mut y = [10.0, 10.0];

        // y = 2 * A * x + 3 * y
        // A*x = [3, 7]
        // y = 2*[3,7] + 3*[10,10] = [6,14] + [30,30] = [36, 44]
        gemv(GemvTrans::NoTrans, 2.0, a.as_ref(), &x, 3.0, &mut y);

        assert!((y[0] - 36.0).abs() < 1e-10);
        assert!((y[1] - 44.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemv_transpose() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let x = [1.0, 2.0]; // 2 elements for A^T (2×3 -> 3×2)
        let mut y = [0.0, 0.0, 0.0];

        // y = A^T * x
        // A^T = [[1,4], [2,5], [3,6]]
        // y[0] = 1*1 + 4*2 = 9
        // y[1] = 2*1 + 5*2 = 12
        // y[2] = 3*1 + 6*2 = 15
        gemv(GemvTrans::Trans, 1.0, a.as_ref(), &x, 0.0, &mut y);

        assert!((y[0] - 9.0).abs() < 1e-10);
        assert!((y[1] - 12.0).abs() < 1e-10);
        assert!((y[2] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemv_alpha_zero() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let x = [1.0, 2.0];
        let mut y = [10.0, 20.0];

        // y = 0 * A * x + 2 * y = [20, 40]
        gemv(GemvTrans::NoTrans, 0.0, a.as_ref(), &x, 2.0, &mut y);

        assert!((y[0] - 20.0).abs() < 1e-10);
        assert!((y[1] - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemv_beta_zero() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let x = [1.0, 2.0];
        let mut y = [100.0, 200.0]; // Should be ignored

        // y = 1 * A * x + 0 * y
        gemv(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, 0.0, &mut y);

        // y[0] = 1*1 + 2*2 = 5
        // y[1] = 3*1 + 4*2 = 11
        assert!((y[0] - 5.0).abs() < 1e-10);
        assert!((y[1] - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemv_identity() {
        let eye = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);
        let x = [1.0, 2.0, 3.0];
        let mut y = [0.0, 0.0, 0.0];

        gemv(GemvTrans::NoTrans, 1.0, eye.as_ref(), &x, 0.0, &mut y);

        assert_eq!(y, x);
    }

    #[test]
    fn test_gemv_simple() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let x = [1.0, 2.0];

        let y = gemv_simple(a.as_ref(), &x);

        assert!((y[0] - 5.0).abs() < 1e-10);
        assert!((y[1] - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemv_f32() {
        let a = Mat::from_rows(&[&[1.0f32, 2.0], &[3.0, 4.0]]);
        let x = [1.0f32, 2.0];
        let mut y = [0.0f32, 0.0];

        gemv(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, 0.0, &mut y);

        assert!((y[0] - 5.0).abs() < 1e-5);
        assert!((y[1] - 11.0).abs() < 1e-5);
    }

    #[test]
    fn test_gemv_empty() {
        let a: Mat<f64> = Mat::zeros(0, 0);
        let x: [f64; 0] = [];
        let mut y: [f64; 0] = [];

        gemv(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, 0.0, &mut y);
        // Should not panic
    }

    #[test]
    fn test_gemv_single() {
        let a = Mat::from_rows(&[&[3.0f64]]);
        let x = [2.0];
        let mut y = [0.0];

        gemv(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, 0.0, &mut y);

        assert!((y[0] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemv_tall_matrix() {
        // 4x2 matrix
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0], &[7.0, 8.0]]);
        let x = [1.0, 1.0];
        let mut y = [0.0, 0.0, 0.0, 0.0];

        gemv(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, 0.0, &mut y);

        assert!((y[0] - 3.0).abs() < 1e-10);
        assert!((y[1] - 7.0).abs() < 1e-10);
        assert!((y[2] - 11.0).abs() < 1e-10);
        assert!((y[3] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemv_wide_matrix() {
        // 2x4 matrix
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]]);
        let x = [1.0, 1.0, 1.0, 1.0];
        let mut y = [0.0, 0.0];

        gemv(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, 0.0, &mut y);

        assert!((y[0] - 10.0).abs() < 1e-10);
        assert!((y[1] - 26.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemv_parallel() {
        // Test with a larger matrix to trigger parallel execution
        let n = 512;
        let a: Mat<f64> = Mat::filled(n, n, 1.0);
        let x: Vec<f64> = vec![1.0; n];
        let mut y: Vec<f64> = vec![0.0; n];

        #[cfg(feature = "parallel")]
        {
            gemv_with_par(
                GemvTrans::NoTrans,
                1.0,
                a.as_ref(),
                &x,
                0.0,
                &mut y,
                Par::Rayon,
            );
        }
        #[cfg(not(feature = "parallel"))]
        {
            gemv(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, 0.0, &mut y);
        }

        // Each element of y should be n (sum of n ones)
        for i in 0..n {
            assert!(
                (y[i] - n as f64).abs() < 1e-10,
                "y[{}] = {}, expected {}",
                i,
                y[i],
                n
            );
        }
    }

    #[test]
    fn test_gemv_parallel_transpose() {
        // Test parallel transpose
        let n = 512;
        let a: Mat<f64> = Mat::filled(n, n, 1.0);
        let x: Vec<f64> = vec![1.0; n];
        let mut y: Vec<f64> = vec![0.0; n];

        #[cfg(feature = "parallel")]
        {
            gemv_with_par(
                GemvTrans::Trans,
                1.0,
                a.as_ref(),
                &x,
                0.0,
                &mut y,
                Par::Rayon,
            );
        }
        #[cfg(not(feature = "parallel"))]
        {
            gemv(GemvTrans::Trans, 1.0, a.as_ref(), &x, 0.0, &mut y);
        }

        // Each element of y should be n (sum of n ones)
        for i in 0..n {
            assert!(
                (y[i] - n as f64).abs() < 1e-10,
                "y[{}] = {}, expected {}",
                i,
                y[i],
                n
            );
        }
    }

    #[test]
    fn test_gemv_blocked_notrans() {
        // Test blocked GEMV with large matrix (> 4096 elements)
        let m = 128;
        let n = 64;
        let a: Mat<f64> = Mat::filled(m, n, 2.0);
        let x: Vec<f64> = vec![1.0; n];
        let mut y: Vec<f64> = vec![0.0; m];

        gemv(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, 0.0, &mut y);

        // Each element of y should be n * 2.0 = 128.0
        for i in 0..m {
            assert!(
                (y[i] - 128.0).abs() < 1e-10,
                "y[{}] = {}, expected 128.0",
                i,
                y[i]
            );
        }
    }

    #[test]
    fn test_gemv_blocked_trans() {
        // Test blocked GEMV transpose with large matrix
        let m = 128;
        let n = 64;
        let a: Mat<f64> = Mat::filled(m, n, 2.0);
        let x: Vec<f64> = vec![1.0; m];
        let mut y: Vec<f64> = vec![0.0; n];

        gemv(GemvTrans::Trans, 1.0, a.as_ref(), &x, 0.0, &mut y);

        // Each element of y should be m * 2.0 = 256.0
        for j in 0..n {
            assert!(
                (y[j] - 256.0).abs() < 1e-10,
                "y[{}] = {}, expected 256.0",
                j,
                y[j]
            );
        }
    }

    #[test]
    fn test_gemv_blocked_with_alpha_beta() {
        // Test blocked GEMV with alpha and beta
        let n = 100;
        let a: Mat<f64> = Mat::filled(n, n, 1.0);
        let x: Vec<f64> = vec![1.0; n];
        let mut y: Vec<f64> = vec![10.0; n];

        // y = 2 * A * x + 0.5 * y
        // A * x = n * 1 = 100 for each element
        // y = 2 * 100 + 0.5 * 10 = 205
        gemv(GemvTrans::NoTrans, 2.0, a.as_ref(), &x, 0.5, &mut y);

        for i in 0..n {
            assert!(
                (y[i] - 205.0).abs() < 1e-10,
                "y[{}] = {}, expected 205.0",
                i,
                y[i]
            );
        }
    }

    // ========================================================================
    // Fused GEMV operations tests
    // ========================================================================

    #[test]
    fn test_gemv_add_basic() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let x = [1.0, 1.0];
        let z = vec![10.0, 20.0];

        // result = 1.0 * A * x + z = [3, 7] + [10, 20] = [13, 27]
        let result = gemv_add(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, z);

        assert!((result[0] - 13.0).abs() < 1e-10);
        assert!((result[1] - 27.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemv_add_with_alpha() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let x = [1.0, 1.0];
        let z = vec![10.0, 20.0];

        // result = 2.0 * A * x + z = 2*[3, 7] + [10, 20] = [6, 14] + [10, 20] = [16, 34]
        let result = gemv_add(GemvTrans::NoTrans, 2.0, a.as_ref(), &x, z);

        assert!((result[0] - 16.0).abs() < 1e-10);
        assert!((result[1] - 34.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemv_add_alpha_zero() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let x = [1.0, 1.0];
        let z = vec![10.0, 20.0];

        // result = 0 * A * x + z = z
        let result = gemv_add(GemvTrans::NoTrans, 0.0, a.as_ref(), &x, z);

        assert!((result[0] - 10.0).abs() < 1e-10);
        assert!((result[1] - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemv_add_transpose() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let x = [1.0, 1.0]; // Input for A^T (m elements)
        let z = vec![10.0, 20.0, 30.0]; // Output size n

        // A^T = [[1,4], [2,5], [3,6]]
        // A^T * x = [1+4, 2+5, 3+6] = [5, 7, 9]
        // result = 1.0 * [5, 7, 9] + [10, 20, 30] = [15, 27, 39]
        let result = gemv_add(GemvTrans::Trans, 1.0, a.as_ref(), &x, z);

        assert!((result[0] - 15.0).abs() < 1e-10);
        assert!((result[1] - 27.0).abs() < 1e-10);
        assert!((result[2] - 39.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemv_add_inplace_basic() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let x = [1.0, 1.0];
        let mut y = vec![10.0, 20.0];

        // y += 1.0 * A * x -> y = [10, 20] + [3, 7] = [13, 27]
        gemv_add_inplace(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, &mut y, 2, 2);

        assert!((y[0] - 13.0).abs() < 1e-10);
        assert!((y[1] - 27.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemv_add_inplace_twice() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let x = [1.0, 1.0];
        let mut y = vec![0.0, 0.0];

        // First: y = [0, 0] + [3, 7] = [3, 7]
        gemv_add_inplace(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, &mut y, 2, 2);
        assert!((y[0] - 3.0).abs() < 1e-10);
        assert!((y[1] - 7.0).abs() < 1e-10);

        // Second: y = [3, 7] + [3, 7] = [6, 14]
        gemv_add_inplace(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, &mut y, 2, 2);
        assert!((y[0] - 6.0).abs() < 1e-10);
        assert!((y[1] - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemv_sum2_basic() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[5.0f64, 6.0], &[7.0, 8.0]]);
        let x = [1.0, 1.0];
        let z = [1.0, 1.0];

        // A*x = [3, 7]
        // B*z = [11, 15]
        // result = 1.0 * [3, 7] + 1.0 * [11, 15] = [14, 22]
        let result = gemv_sum2(1.0, a.as_ref(), &x, 1.0, b.as_ref(), &z);

        assert!((result[0] - 14.0).abs() < 1e-10);
        assert!((result[1] - 22.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemv_sum2_with_scalars() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[5.0f64, 6.0], &[7.0, 8.0]]);
        let x = [1.0, 1.0];
        let z = [1.0, 1.0];

        // A*x = [3, 7]
        // B*z = [11, 15]
        // result = 2.0 * [3, 7] + 0.5 * [11, 15] = [6, 14] + [5.5, 7.5] = [11.5, 21.5]
        let result = gemv_sum2(2.0, a.as_ref(), &x, 0.5, b.as_ref(), &z);

        assert!((result[0] - 11.5).abs() < 1e-10);
        assert!((result[1] - 21.5).abs() < 1e-10);
    }

    #[test]
    fn test_gemv_sum2_alpha_zero() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[5.0f64, 6.0], &[7.0, 8.0]]);
        let x = [1.0, 1.0];
        let z = [1.0, 1.0];

        // result = 0 * A*x + 1.0 * B*z = [11, 15]
        let result = gemv_sum2(0.0, a.as_ref(), &x, 1.0, b.as_ref(), &z);

        assert!((result[0] - 11.0).abs() < 1e-10);
        assert!((result[1] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemv_sum2_beta_zero() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[5.0f64, 6.0], &[7.0, 8.0]]);
        let x = [1.0, 1.0];
        let z = [1.0, 1.0];

        // result = 1.0 * A*x + 0 * B*z = [3, 7]
        let result = gemv_sum2(1.0, a.as_ref(), &x, 0.0, b.as_ref(), &z);

        assert!((result[0] - 3.0).abs() < 1e-10);
        assert!((result[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_gemv_add_blocked() {
        // Test with large matrix that triggers blocked implementation
        let n = 100;
        let a: Mat<f64> = Mat::filled(n, n, 1.0);
        let x: Vec<f64> = vec![1.0; n];
        let z: Vec<f64> = vec![5.0; n];

        // result = 1.0 * A * x + z
        // A * x = [n, n, ..., n] (each element = 100)
        // result = [100, 100, ...] + [5, 5, ...] = [105, 105, ...]
        let result = gemv_add(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, z);

        for i in 0..n {
            assert!(
                (result[i] - 105.0).abs() < 1e-10,
                "result[{}] = {}, expected 105.0",
                i,
                result[i]
            );
        }
    }

    #[test]
    fn test_gemv_add_blocked_trans() {
        // Test blocked transpose
        let m = 100;
        let n = 80;
        let a: Mat<f64> = Mat::filled(m, n, 1.0);
        let x: Vec<f64> = vec![1.0; m];
        let z: Vec<f64> = vec![5.0; n];

        // A^T * x: each element = m = 100
        // result = 1.0 * [100, ...] + [5, ...] = [105, ...]
        let result = gemv_add(GemvTrans::Trans, 1.0, a.as_ref(), &x, z);

        for j in 0..n {
            assert!(
                (result[j] - 105.0).abs() < 1e-10,
                "result[{}] = {}, expected 105.0",
                j,
                result[j]
            );
        }
    }

    #[test]
    fn test_gemv_sum2_large() {
        // Test gemv_sum2 with larger matrices
        let n = 100;
        let a: Mat<f64> = Mat::filled(n, n, 1.0);
        let b: Mat<f64> = Mat::filled(n, n, 2.0);
        let x: Vec<f64> = vec![1.0; n];
        let z: Vec<f64> = vec![1.0; n];

        // A*x = [100, 100, ...]
        // B*z = [200, 200, ...]
        // result = 1.0 * [100, ...] + 0.5 * [200, ...] = [100, ...] + [100, ...] = [200, ...]
        let result = gemv_sum2(1.0, a.as_ref(), &x, 0.5, b.as_ref(), &z);

        for i in 0..n {
            assert!(
                (result[i] - 200.0).abs() < 1e-10,
                "result[{}] = {}, expected 200.0",
                i,
                result[i]
            );
        }
    }
}
