//! Strassen's Algorithm for Matrix Multiplication.
//!
//! This module implements Strassen's algorithm for very large matrix multiplications.
//! Strassen's algorithm reduces the complexity from O(n³) to O(n^2.807) by recursively
//! dividing matrices into 2×2 blocks and computing 7 intermediate products instead of 8.
//!
//! ## Algorithm
//!
//! For matrices A and B partitioned as:
//! ```text
//! A = [A11 A12]    B = [B11 B12]
//!     [A21 A22]        [B21 B22]
//! ```
//!
//! Compute 7 products:
//! - M1 = (A11 + A22)(B11 + B22)
//! - M2 = (A21 + A22)B11
//! - M3 = A11(B12 - B22)
//! - M4 = A22(B21 - B11)
//! - M5 = (A11 + A12)B22
//! - M6 = (A21 - A11)(B11 + B12)
//! - M7 = (A12 - A22)(B21 + B22)
//!
//! Then:
//! - C11 = M1 + M4 - M5 + M7
//! - C12 = M3 + M5
//! - C21 = M2 + M4
//! - C22 = M1 - M2 + M3 + M6
//!
//! ## Non-square and non-power-of-two shapes: dynamic peeling
//!
//! The classic 2×2 partition above requires every dimension to be even. Rather than
//! padding `m`, `k`, and `n` up to a single shared power of two of the *largest*
//! dimension (which can inflate the effective work by up to ~8x for rectangular
//! matrices, e.g. padding a 1000×8 by 8×1000 multiply to 1024×1024×1024), this
//! implementation uses *dynamic peeling*: whichever of `m`, `k`, or `n` is odd has a
//! single row/column/inner-slice shaved off and settled with a direct GEMM call, and
//! the remaining even-sized bulk recurses normally. Each dimension is therefore only
//! ever "padded" by at most one element per recursion level, so the overhead scales
//! with that dimension individually instead of with the cube of the largest one.
//!
//! Very rectangular shapes still are not good candidates for Strassen (the recursion
//! and allocation bookkeeping cost is not amortized by a mostly-tiny dimension), so
//! [`should_use_strassen`] additionally rejects shapes whose largest dimension is more
//! than `STRASSEN_MAX_ASPECT_RATIO` times the smallest one, falling back to the
//! standard blocked GEMM instead.
//!
//! ## Usage
//!
//! Strassen's algorithm is beneficial for very large matrices (typically > 512×512).
//! For smaller matrices, the standard blocked GEMM is faster due to lower overhead.

use crate::level3::gemm::{GemmBlocking, gemm_with_blocking};
use crate::level3::gemm_kernel::GemmKernel;
use oxiblas_core::parallel::Par;
use oxiblas_core::scalar::Field;
use oxiblas_matrix::{Mat, MatMut, MatRef};

/// Threshold for using Strassen vs standard GEMM.
/// Matrices smaller than this use standard blocked GEMM.
pub const STRASSEN_THRESHOLD: usize = 512;

/// Minimum dimension for Strassen recursion.
/// Below this, we use standard GEMM even within Strassen recursion.
const STRASSEN_LEAF_SIZE: usize = 64;

/// Maximum recursion depth to prevent stack overflow and manage memory.
const MAX_STRASSEN_DEPTH: usize = 4;

/// Maximum tolerated ratio between the largest and smallest dimension for
/// Strassen recursion to be considered worthwhile.
///
/// Beyond this ratio the matrices are rectangular enough (e.g. a GEMV-like
/// `k = 8` multiplied against `m = n = 1000`) that Strassen's `O(n^2.807)`
/// scaling -- which only kicks in once *all three* dimensions shrink
/// together -- gains little while still paying recursion and allocation
/// overhead, so the standard blocked GEMM is expected to win outright.
const STRASSEN_MAX_ASPECT_RATIO: usize = 8;

/// Performs matrix multiplication using Strassen's algorithm for large matrices.
///
/// C = alpha * A * B + beta * C
///
/// For matrices with dimension > `STRASSEN_THRESHOLD`, this uses Strassen's algorithm.
/// For smaller matrices, falls back to standard blocked GEMM.
///
/// # Arguments
///
/// * `alpha` - Scalar multiplier for A * B
/// * `a` - Left matrix (m × k)
/// * `b` - Right matrix (k × n)
/// * `beta` - Scalar multiplier for C
/// * `c` - Output matrix (m × n)
///
/// # Example
///
/// ```
/// use oxiblas_blas::level3::strassen::gemm_strassen;
/// use oxiblas_matrix::Mat;
///
/// let a: Mat<f64> = Mat::filled(100, 100, 1.0);
/// let b: Mat<f64> = Mat::filled(100, 100, 2.0);
/// let mut c: Mat<f64> = Mat::zeros(100, 100);
///
/// gemm_strassen(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());
///
/// // Each element should be 100 * 1 * 2 = 200
/// assert!((c[(0, 0)] - 200.0).abs() < 1e-10);
/// ```
pub fn gemm_strassen<T: Field + GemmKernel + bytemuck::Zeroable>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    mut c: MatMut<'_, T>,
) {
    gemm_strassen_with_par(alpha, a, b, beta, c.rb_mut(), Par::Seq);
}

/// Performs Strassen matrix multiplication with parallelization control.
pub fn gemm_strassen_with_par<T: Field + GemmKernel + bytemuck::Zeroable>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    mut c: MatMut<'_, T>,
    par: Par,
) {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

    assert_eq!(k, b.nrows(), "A.ncols must equal B.nrows");
    assert_eq!(c.nrows(), m, "C.nrows must equal A.nrows");
    assert_eq!(c.ncols(), n, "C.ncols must equal B.ncols");

    // Handle empty matrices
    if m == 0 || n == 0 || k == 0 {
        if k == 0 && beta != T::one() {
            // Scale C by beta
            if beta == T::zero() {
                c.fill_zero();
            } else {
                c.scale(beta);
            }
        }
        return;
    }

    if !should_use_strassen(m, k, n) {
        // Too small to amortize Strassen's bookkeeping, or too rectangular
        // for its recursive halving to pay off: use the standard blocked GEMM.
        strassen_direct_gemm(alpha, a, b, beta, c, par);
        return;
    }

    // Strassen's recursion with dynamic peeling: each dimension is only
    // ever shaved down by one row/column per level as needed (see the
    // module documentation), so the overhead scales with each dimension
    // individually rather than with a shared power-of-two of the largest
    // one. No upfront padded copy of the whole matrix is required.
    strassen_recursive(alpha, a, b, beta, c, 0, par);
}

/// Invokes the standard blocked GEMM kernel directly, bypassing Strassen's
/// recursion. Used both as the recursion's base case and to settle the
/// leftover row/column/inner-slice produced by dynamic peeling.
#[inline]
fn strassen_direct_gemm<T: Field + GemmKernel + bytemuck::Zeroable>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    par: Par,
) {
    let shape = T::micro_kernel_shape();
    let blocking = GemmBlocking::for_kernel::<T>(&shape);
    gemm_with_blocking(alpha, a, b, beta, c, par, &blocking);
}

/// Recursive Strassen implementation with dynamic peeling.
///
/// Handles arbitrary (non-square, odd-sized) `m x k` by `k x n` shapes directly:
/// odd dimensions are peeled one row/column at a time (see the module
/// documentation) until `m`, `k`, and `n` are all even, at which point the
/// standard 2x2 Strassen partition is applied.
fn strassen_recursive<T: Field + GemmKernel + bytemuck::Zeroable>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    mut c: MatMut<'_, T>,
    depth: usize,
    par: Par,
) {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();
    debug_assert_eq!(k, b.nrows());
    debug_assert_eq!(m, c.nrows());
    debug_assert_eq!(n, c.ncols());

    // Base case: fall back to the blocked GEMM kernel once any dimension is
    // small enough to no longer be worth splitting, or the recursion budget
    // is exhausted. `gemm_with_blocking` handles arbitrary (non-square,
    // odd-sized) shapes directly.
    if m.min(k).min(n) <= STRASSEN_LEAF_SIZE || depth >= MAX_STRASSEN_DEPTH {
        strassen_direct_gemm(alpha, a, b, beta, c, par);
        return;
    }

    // Dynamic peeling: shave a single row/column off odd dimensions instead
    // of padding every dimension up to a shared power of two. Each peeled
    // slice is settled with one direct GEMM call, so the padding overhead
    // stays proportional to that dimension alone (at most +1 per level).
    if m % 2 == 1 {
        strassen_peel_row(alpha, a, b, beta, c, depth, par);
        return;
    }
    if k % 2 == 1 {
        strassen_peel_inner(alpha, a, b, beta, c, depth, par);
        return;
    }
    if n % 2 == 1 {
        strassen_peel_col(alpha, a, b, beta, c, depth, par);
        return;
    }

    // All dimensions are even: apply the standard 2x2 Strassen partition.
    // A is m x k, B is k x n, C is m x n; the halves are independent per
    // matrix (half_m/half_k from A, half_k/half_n from B), so this works
    // directly for rectangular (non-square) shapes.
    let half_m = m / 2;
    let half_k = k / 2;
    let half_n = n / 2;

    let a11 = a.submatrix(0, 0, half_m, half_k);
    let a12 = a.submatrix(0, half_k, half_m, half_k);
    let a21 = a.submatrix(half_m, 0, half_m, half_k);
    let a22 = a.submatrix(half_m, half_k, half_m, half_k);

    let b11 = b.submatrix(0, 0, half_k, half_n);
    let b12 = b.submatrix(0, half_n, half_k, half_n);
    let b21 = b.submatrix(half_k, 0, half_k, half_n);
    let b22 = b.submatrix(half_k, half_n, half_k, half_n);

    // Allocate intermediate matrices for M1-M7 (each half_m x half_n).
    let mut m1: Mat<T> = Mat::zeros(half_m, half_n);
    let mut m2: Mat<T> = Mat::zeros(half_m, half_n);
    let mut m3: Mat<T> = Mat::zeros(half_m, half_n);
    let mut m4: Mat<T> = Mat::zeros(half_m, half_n);
    let mut m5: Mat<T> = Mat::zeros(half_m, half_n);
    let mut m6: Mat<T> = Mat::zeros(half_m, half_n);
    let mut m7: Mat<T> = Mat::zeros(half_m, half_n);

    // Temporary matrices for sums: temp_a mirrors an A quadrant
    // (half_m x half_k), temp_b mirrors a B quadrant (half_k x half_n).
    let mut temp_a: Mat<T> = Mat::zeros(half_m, half_k);
    let mut temp_b: Mat<T> = Mat::zeros(half_k, half_n);

    // M1 = (A11 + A22)(B11 + B22)
    matrix_add(&a11, &a22, &mut temp_a);
    matrix_add(&b11, &b22, &mut temp_b);
    strassen_recursive(
        T::one(),
        temp_a.as_ref(),
        temp_b.as_ref(),
        T::zero(),
        m1.as_mut(),
        depth + 1,
        par,
    );

    // M2 = (A21 + A22)B11
    matrix_add(&a21, &a22, &mut temp_a);
    strassen_recursive(
        T::one(),
        temp_a.as_ref(),
        b11,
        T::zero(),
        m2.as_mut(),
        depth + 1,
        par,
    );

    // M3 = A11(B12 - B22)
    matrix_sub(&b12, &b22, &mut temp_b);
    strassen_recursive(
        T::one(),
        a11,
        temp_b.as_ref(),
        T::zero(),
        m3.as_mut(),
        depth + 1,
        par,
    );

    // M4 = A22(B21 - B11)
    matrix_sub(&b21, &b11, &mut temp_b);
    strassen_recursive(
        T::one(),
        a22,
        temp_b.as_ref(),
        T::zero(),
        m4.as_mut(),
        depth + 1,
        par,
    );

    // M5 = (A11 + A12)B22
    matrix_add(&a11, &a12, &mut temp_a);
    strassen_recursive(
        T::one(),
        temp_a.as_ref(),
        b22,
        T::zero(),
        m5.as_mut(),
        depth + 1,
        par,
    );

    // M6 = (A21 - A11)(B11 + B12)
    matrix_sub(&a21, &a11, &mut temp_a);
    matrix_add(&b11, &b12, &mut temp_b);
    strassen_recursive(
        T::one(),
        temp_a.as_ref(),
        temp_b.as_ref(),
        T::zero(),
        m6.as_mut(),
        depth + 1,
        par,
    );

    // M7 = (A12 - A22)(B21 + B22)
    matrix_sub(&a12, &a22, &mut temp_a);
    matrix_add(&b21, &b22, &mut temp_b);
    strassen_recursive(
        T::one(),
        temp_a.as_ref(),
        temp_b.as_ref(),
        T::zero(),
        m7.as_mut(),
        depth + 1,
        par,
    );

    // Combine results into C
    // C11 = M1 + M4 - M5 + M7
    // C12 = M3 + M5
    // C21 = M2 + M4
    // C22 = M1 - M2 + M3 + M6

    // Apply beta to existing C if needed
    if beta != T::zero() && beta != T::one() {
        c.scale(beta);
    }

    for i in 0..half_m {
        for j in 0..half_n {
            let c11_contrib = m1[(i, j)] + m4[(i, j)] - m5[(i, j)] + m7[(i, j)];
            let c12_contrib = m3[(i, j)] + m5[(i, j)];
            let c21_contrib = m2[(i, j)] + m4[(i, j)];
            let c22_contrib = m1[(i, j)] - m2[(i, j)] + m3[(i, j)] + m6[(i, j)];

            if beta == T::zero() {
                c.set(i, j, alpha * c11_contrib);
                c.set(i, j + half_n, alpha * c12_contrib);
                c.set(i + half_m, j, alpha * c21_contrib);
                c.set(i + half_m, j + half_n, alpha * c22_contrib);
            } else {
                c.set(i, j, c[(i, j)] + alpha * c11_contrib);
                c.set(i, j + half_n, c[(i, j + half_n)] + alpha * c12_contrib);
                c.set(i + half_m, j, c[(i + half_m, j)] + alpha * c21_contrib);
                c.set(
                    i + half_m,
                    j + half_n,
                    c[(i + half_m, j + half_n)] + alpha * c22_contrib,
                );
            }
        }
    }
}

/// Peels the last row off `A`/`C` when `m` is odd, so the remaining
/// `m - 1` rows can be split evenly for the 2x2 Strassen recursion.
///
/// `C_top = alpha * A_top * B + beta * C_top` is computed recursively, and
/// the single leftover row `C_last = alpha * A_last * B + beta * C_last` is
/// settled directly -- an O(k * n) correction, negligible next to the
/// O(m * k * n) bulk of the multiply.
fn strassen_peel_row<T: Field + GemmKernel + bytemuck::Zeroable>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    depth: usize,
    par: Par,
) {
    let m = a.nrows();
    debug_assert!(m % 2 == 1, "strassen_peel_row requires an odd m");
    let bulk = m - 1;

    let (a_top, a_last) = a.split_rows(bulk);
    let (c_top, c_last) = c.split_rows(bulk);

    strassen_recursive(alpha, a_top, b, beta, c_top, depth, par);
    strassen_direct_gemm(alpha, a_last, b, beta, c_last, par);
}

/// Peels the last column off `A` and last row off `B` when `k` (the shared
/// inner/contraction dimension) is odd, so the remaining `k - 1` columns/rows
/// can be split evenly for the 2x2 Strassen recursion.
///
/// `A * B = A_left * B_top + A_right * B_bottom` splits exactly along the
/// contraction dimension, so both contributions are accumulated into the
/// same `C`: the bulk product is computed first (applying `beta` to the
/// existing `C`), then the leftover rank-1-style outer product
/// `A_right * B_bottom` (A_right is `m x 1`, B_bottom is `1 x k`) is added
/// on top with `beta = 1`.
fn strassen_peel_inner<T: Field + GemmKernel + bytemuck::Zeroable>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    mut c: MatMut<'_, T>,
    depth: usize,
    par: Par,
) {
    let k = a.ncols();
    debug_assert!(k % 2 == 1, "strassen_peel_inner requires an odd k");
    let bulk = k - 1;

    let (a_left, a_right) = a.split_cols(bulk);
    let (b_top, b_bottom) = b.split_rows(bulk);

    strassen_recursive(alpha, a_left, b_top, beta, c.rb_mut(), depth, par);
    strassen_direct_gemm(alpha, a_right, b_bottom, T::one(), c, par);
}

/// Peels the last column off `B`/`C` when `n` is odd, so the remaining
/// `n - 1` columns can be split evenly for the 2x2 Strassen recursion.
///
/// Mirrors [`strassen_peel_row`] along the column dimension.
fn strassen_peel_col<T: Field + GemmKernel + bytemuck::Zeroable>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    depth: usize,
    par: Par,
) {
    let n = b.ncols();
    debug_assert!(n % 2 == 1, "strassen_peel_col requires an odd n");
    let bulk = n - 1;

    let (b_left, b_right) = b.split_cols(bulk);
    let (c_left, c_right) = c.split_cols(bulk);

    strassen_recursive(alpha, a, b_left, beta, c_left, depth, par);
    strassen_direct_gemm(alpha, a, b_right, beta, c_right, par);
}

/// Matrix addition: C = A + B
#[inline]
fn matrix_add<T: Field>(a: &MatRef<'_, T>, b: &MatRef<'_, T>, c: &mut Mat<T>) {
    let m = a.nrows();
    let n = a.ncols();
    debug_assert_eq!(m, b.nrows());
    debug_assert_eq!(n, b.ncols());
    debug_assert_eq!(m, c.nrows());
    debug_assert_eq!(n, c.ncols());

    // Use 4-way unrolling for better performance
    for j in 0..n {
        let mut i = 0;
        while i + 4 <= m {
            c[(i, j)] = a[(i, j)] + b[(i, j)];
            c[(i + 1, j)] = a[(i + 1, j)] + b[(i + 1, j)];
            c[(i + 2, j)] = a[(i + 2, j)] + b[(i + 2, j)];
            c[(i + 3, j)] = a[(i + 3, j)] + b[(i + 3, j)];
            i += 4;
        }
        while i < m {
            c[(i, j)] = a[(i, j)] + b[(i, j)];
            i += 1;
        }
    }
}

/// Matrix subtraction: C = A - B
#[inline]
fn matrix_sub<T: Field>(a: &MatRef<'_, T>, b: &MatRef<'_, T>, c: &mut Mat<T>) {
    let m = a.nrows();
    let n = a.ncols();
    debug_assert_eq!(m, b.nrows());
    debug_assert_eq!(n, b.ncols());
    debug_assert_eq!(m, c.nrows());
    debug_assert_eq!(n, c.ncols());

    // Use 4-way unrolling for better performance
    for j in 0..n {
        let mut i = 0;
        while i + 4 <= m {
            c[(i, j)] = a[(i, j)] - b[(i, j)];
            c[(i + 1, j)] = a[(i + 1, j)] - b[(i + 1, j)];
            c[(i + 2, j)] = a[(i + 2, j)] - b[(i + 2, j)];
            c[(i + 3, j)] = a[(i + 3, j)] - b[(i + 3, j)];
            i += 4;
        }
        while i < m {
            c[(i, j)] = a[(i, j)] - b[(i, j)];
            i += 1;
        }
    }
}

/// Checks if Strassen's algorithm would be beneficial for the given dimensions.
///
/// Returns `true` only when every dimension clears [`STRASSEN_THRESHOLD`] *and*
/// the matrices are not too rectangular: the largest dimension must be no
/// more than `STRASSEN_MAX_ASPECT_RATIO` times the smallest one. Very
/// rectangular shapes (e.g. `1000 x 8` times `8 x 1000`) gain little from
/// Strassen's asymptotic scaling -- which only kicks in once *all three*
/// dimensions shrink together via recursion -- while still paying its
/// recursion and allocation overhead, so plain GEMM is used instead.
#[must_use]
pub fn should_use_strassen(m: usize, k: usize, n: usize) -> bool {
    let min_dim = m.min(k).min(n);
    let max_dim = m.max(k).max(n);
    min_dim >= STRASSEN_THRESHOLD && max_dim <= min_dim * STRASSEN_MAX_ASPECT_RATIO
}

/// Parallel Strassen for very large matrices.
///
/// Uses parallel computation for the 7 intermediate products.
#[cfg(feature = "parallel")]
pub fn gemm_strassen_parallel<T: Field + GemmKernel + bytemuck::Zeroable + Send + Sync>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    mut c: MatMut<'_, T>,
) where
    Mat<T>: Send + Sync,
{
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

    assert_eq!(k, b.nrows(), "A.ncols must equal B.nrows");
    assert_eq!(c.nrows(), m, "C.nrows must equal A.nrows");
    assert_eq!(c.ncols(), n, "C.ncols must equal B.ncols");

    if m == 0 || n == 0 || k == 0 {
        if k == 0 && beta != T::one() {
            if beta == T::zero() {
                c.fill_zero();
            } else {
                c.scale(beta);
            }
        }
        return;
    }

    if !should_use_strassen(m, k, n) {
        strassen_direct_gemm(alpha, a, b, beta, c, Par::Rayon);
        return;
    }

    strassen_recursive_parallel(alpha, a, b, beta, c, 0);
}

/// Peels the last row off `A`/`C` when `m` is odd (parallel variant).
///
/// See [`strassen_peel_row`] for the sequential version this mirrors.
#[cfg(feature = "parallel")]
fn strassen_peel_row_parallel<T: Field + GemmKernel + bytemuck::Zeroable + Send + Sync>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    depth: usize,
) where
    Mat<T>: Send + Sync,
{
    let m = a.nrows();
    debug_assert!(m % 2 == 1, "strassen_peel_row_parallel requires an odd m");
    let bulk = m - 1;

    let (a_top, a_last) = a.split_rows(bulk);
    let (c_top, c_last) = c.split_rows(bulk);

    strassen_recursive_parallel(alpha, a_top, b, beta, c_top, depth);
    strassen_direct_gemm(alpha, a_last, b, beta, c_last, Par::Rayon);
}

/// Peels the last column of `A` / last row of `B` when `k` is odd (parallel variant).
///
/// See [`strassen_peel_inner`] for the sequential version this mirrors.
#[cfg(feature = "parallel")]
fn strassen_peel_inner_parallel<T: Field + GemmKernel + bytemuck::Zeroable + Send + Sync>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    mut c: MatMut<'_, T>,
    depth: usize,
) where
    Mat<T>: Send + Sync,
{
    let k = a.ncols();
    debug_assert!(k % 2 == 1, "strassen_peel_inner_parallel requires an odd k");
    let bulk = k - 1;

    let (a_left, a_right) = a.split_cols(bulk);
    let (b_top, b_bottom) = b.split_rows(bulk);

    strassen_recursive_parallel(alpha, a_left, b_top, beta, c.rb_mut(), depth);
    strassen_direct_gemm(alpha, a_right, b_bottom, T::one(), c, Par::Rayon);
}

/// Peels the last column off `B`/`C` when `n` is odd (parallel variant).
///
/// See [`strassen_peel_col`] for the sequential version this mirrors.
#[cfg(feature = "parallel")]
fn strassen_peel_col_parallel<T: Field + GemmKernel + bytemuck::Zeroable + Send + Sync>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    c: MatMut<'_, T>,
    depth: usize,
) where
    Mat<T>: Send + Sync,
{
    let n = b.ncols();
    debug_assert!(n % 2 == 1, "strassen_peel_col_parallel requires an odd n");
    let bulk = n - 1;

    let (b_left, b_right) = b.split_cols(bulk);
    let (c_left, c_right) = c.split_cols(bulk);

    strassen_recursive_parallel(alpha, a, b_left, beta, c_left, depth);
    strassen_direct_gemm(alpha, a, b_right, beta, c_right, Par::Rayon);
}

/// Parallel recursive Strassen with dynamic peeling.
///
/// See [`strassen_recursive`] for the sequential version this mirrors.
#[cfg(feature = "parallel")]
fn strassen_recursive_parallel<T: Field + GemmKernel + bytemuck::Zeroable + Send + Sync>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    mut c: MatMut<'_, T>,
    depth: usize,
) where
    Mat<T>: Send + Sync,
{
    use rayon::prelude::*;

    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

    if m.min(k).min(n) <= STRASSEN_LEAF_SIZE || depth >= MAX_STRASSEN_DEPTH {
        strassen_direct_gemm(alpha, a, b, beta, c, Par::Rayon);
        return;
    }

    if m % 2 == 1 {
        strassen_peel_row_parallel(alpha, a, b, beta, c, depth);
        return;
    }
    if k % 2 == 1 {
        strassen_peel_inner_parallel(alpha, a, b, beta, c, depth);
        return;
    }
    if n % 2 == 1 {
        strassen_peel_col_parallel(alpha, a, b, beta, c, depth);
        return;
    }

    let half_m = m / 2;
    let half_k = k / 2;
    let half_n = n / 2;

    // Partition matrices
    let a11 = a.submatrix(0, 0, half_m, half_k);
    let a12 = a.submatrix(0, half_k, half_m, half_k);
    let a21 = a.submatrix(half_m, 0, half_m, half_k);
    let a22 = a.submatrix(half_m, half_k, half_m, half_k);

    let b11 = b.submatrix(0, 0, half_k, half_n);
    let b12 = b.submatrix(0, half_n, half_k, half_n);
    let b21 = b.submatrix(half_k, 0, half_k, half_n);
    let b22 = b.submatrix(half_k, half_n, half_k, half_n);

    // Create owned copies for parallel computation
    let a11_owned = copy_to_mat(&a11);
    let a12_owned = copy_to_mat(&a12);
    let a21_owned = copy_to_mat(&a21);
    let a22_owned = copy_to_mat(&a22);
    let b11_owned = copy_to_mat(&b11);
    let b12_owned = copy_to_mat(&b12);
    let b21_owned = copy_to_mat(&b21);
    let b22_owned = copy_to_mat(&b22);

    // Compute M1-M7 in parallel at the top level (depth == 0)
    let (m1, m2, m3, m4, m5, m6, m7) = if depth == 0 {
        // Parallel computation of the 7 products
        let results: Vec<Mat<T>> = (0..7)
            .into_par_iter()
            .map(|idx| {
                let mut temp_a: Mat<T> = Mat::zeros(half_m, half_k);
                let mut temp_b: Mat<T> = Mat::zeros(half_k, half_n);
                let mut result: Mat<T> = Mat::zeros(half_m, half_n);

                match idx {
                    0 => {
                        // M1 = (A11 + A22)(B11 + B22)
                        matrix_add(&a11_owned.as_ref(), &a22_owned.as_ref(), &mut temp_a);
                        matrix_add(&b11_owned.as_ref(), &b22_owned.as_ref(), &mut temp_b);
                        strassen_recursive_parallel(
                            T::one(),
                            temp_a.as_ref(),
                            temp_b.as_ref(),
                            T::zero(),
                            result.as_mut(),
                            depth + 1,
                        );
                    }
                    1 => {
                        // M2 = (A21 + A22)B11
                        matrix_add(&a21_owned.as_ref(), &a22_owned.as_ref(), &mut temp_a);
                        strassen_recursive_parallel(
                            T::one(),
                            temp_a.as_ref(),
                            b11_owned.as_ref(),
                            T::zero(),
                            result.as_mut(),
                            depth + 1,
                        );
                    }
                    2 => {
                        // M3 = A11(B12 - B22)
                        matrix_sub(&b12_owned.as_ref(), &b22_owned.as_ref(), &mut temp_b);
                        strassen_recursive_parallel(
                            T::one(),
                            a11_owned.as_ref(),
                            temp_b.as_ref(),
                            T::zero(),
                            result.as_mut(),
                            depth + 1,
                        );
                    }
                    3 => {
                        // M4 = A22(B21 - B11)
                        matrix_sub(&b21_owned.as_ref(), &b11_owned.as_ref(), &mut temp_b);
                        strassen_recursive_parallel(
                            T::one(),
                            a22_owned.as_ref(),
                            temp_b.as_ref(),
                            T::zero(),
                            result.as_mut(),
                            depth + 1,
                        );
                    }
                    4 => {
                        // M5 = (A11 + A12)B22
                        matrix_add(&a11_owned.as_ref(), &a12_owned.as_ref(), &mut temp_a);
                        strassen_recursive_parallel(
                            T::one(),
                            temp_a.as_ref(),
                            b22_owned.as_ref(),
                            T::zero(),
                            result.as_mut(),
                            depth + 1,
                        );
                    }
                    5 => {
                        // M6 = (A21 - A11)(B11 + B12)
                        matrix_sub(&a21_owned.as_ref(), &a11_owned.as_ref(), &mut temp_a);
                        matrix_add(&b11_owned.as_ref(), &b12_owned.as_ref(), &mut temp_b);
                        strassen_recursive_parallel(
                            T::one(),
                            temp_a.as_ref(),
                            temp_b.as_ref(),
                            T::zero(),
                            result.as_mut(),
                            depth + 1,
                        );
                    }
                    6 => {
                        // M7 = (A12 - A22)(B21 + B22)
                        matrix_sub(&a12_owned.as_ref(), &a22_owned.as_ref(), &mut temp_a);
                        matrix_add(&b21_owned.as_ref(), &b22_owned.as_ref(), &mut temp_b);
                        strassen_recursive_parallel(
                            T::one(),
                            temp_a.as_ref(),
                            temp_b.as_ref(),
                            T::zero(),
                            result.as_mut(),
                            depth + 1,
                        );
                    }
                    _ => unreachable!(),
                }
                result
            })
            .collect();

        (
            results[0].clone(),
            results[1].clone(),
            results[2].clone(),
            results[3].clone(),
            results[4].clone(),
            results[5].clone(),
            results[6].clone(),
        )
    } else {
        // Sequential computation for deeper recursion levels
        let mut m1: Mat<T> = Mat::zeros(half_m, half_n);
        let mut m2: Mat<T> = Mat::zeros(half_m, half_n);
        let mut m3: Mat<T> = Mat::zeros(half_m, half_n);
        let mut m4: Mat<T> = Mat::zeros(half_m, half_n);
        let mut m5: Mat<T> = Mat::zeros(half_m, half_n);
        let mut m6: Mat<T> = Mat::zeros(half_m, half_n);
        let mut m7: Mat<T> = Mat::zeros(half_m, half_n);
        let mut temp_a: Mat<T> = Mat::zeros(half_m, half_k);
        let mut temp_b: Mat<T> = Mat::zeros(half_k, half_n);

        matrix_add(&a11_owned.as_ref(), &a22_owned.as_ref(), &mut temp_a);
        matrix_add(&b11_owned.as_ref(), &b22_owned.as_ref(), &mut temp_b);
        strassen_recursive_parallel(
            T::one(),
            temp_a.as_ref(),
            temp_b.as_ref(),
            T::zero(),
            m1.as_mut(),
            depth + 1,
        );

        matrix_add(&a21_owned.as_ref(), &a22_owned.as_ref(), &mut temp_a);
        strassen_recursive_parallel(
            T::one(),
            temp_a.as_ref(),
            b11_owned.as_ref(),
            T::zero(),
            m2.as_mut(),
            depth + 1,
        );

        matrix_sub(&b12_owned.as_ref(), &b22_owned.as_ref(), &mut temp_b);
        strassen_recursive_parallel(
            T::one(),
            a11_owned.as_ref(),
            temp_b.as_ref(),
            T::zero(),
            m3.as_mut(),
            depth + 1,
        );

        matrix_sub(&b21_owned.as_ref(), &b11_owned.as_ref(), &mut temp_b);
        strassen_recursive_parallel(
            T::one(),
            a22_owned.as_ref(),
            temp_b.as_ref(),
            T::zero(),
            m4.as_mut(),
            depth + 1,
        );

        matrix_add(&a11_owned.as_ref(), &a12_owned.as_ref(), &mut temp_a);
        strassen_recursive_parallel(
            T::one(),
            temp_a.as_ref(),
            b22_owned.as_ref(),
            T::zero(),
            m5.as_mut(),
            depth + 1,
        );

        matrix_sub(&a21_owned.as_ref(), &a11_owned.as_ref(), &mut temp_a);
        matrix_add(&b11_owned.as_ref(), &b12_owned.as_ref(), &mut temp_b);
        strassen_recursive_parallel(
            T::one(),
            temp_a.as_ref(),
            temp_b.as_ref(),
            T::zero(),
            m6.as_mut(),
            depth + 1,
        );

        matrix_sub(&a12_owned.as_ref(), &a22_owned.as_ref(), &mut temp_a);
        matrix_add(&b21_owned.as_ref(), &b22_owned.as_ref(), &mut temp_b);
        strassen_recursive_parallel(
            T::one(),
            temp_a.as_ref(),
            temp_b.as_ref(),
            T::zero(),
            m7.as_mut(),
            depth + 1,
        );

        (m1, m2, m3, m4, m5, m6, m7)
    };

    // Apply beta and combine results
    if beta != T::zero() && beta != T::one() {
        c.scale(beta);
    }

    for i in 0..half_m {
        for j in 0..half_n {
            let c11_contrib = m1[(i, j)] + m4[(i, j)] - m5[(i, j)] + m7[(i, j)];
            let c12_contrib = m3[(i, j)] + m5[(i, j)];
            let c21_contrib = m2[(i, j)] + m4[(i, j)];
            let c22_contrib = m1[(i, j)] - m2[(i, j)] + m3[(i, j)] + m6[(i, j)];

            if beta == T::zero() {
                c.set(i, j, alpha * c11_contrib);
                c.set(i, j + half_n, alpha * c12_contrib);
                c.set(i + half_m, j, alpha * c21_contrib);
                c.set(i + half_m, j + half_n, alpha * c22_contrib);
            } else {
                c.set(i, j, c[(i, j)] + alpha * c11_contrib);
                c.set(i, j + half_n, c[(i, j + half_n)] + alpha * c12_contrib);
                c.set(i + half_m, j, c[(i + half_m, j)] + alpha * c21_contrib);
                c.set(
                    i + half_m,
                    j + half_n,
                    c[(i + half_m, j + half_n)] + alpha * c22_contrib,
                );
            }
        }
    }
}

/// Helper to copy a submatrix to an owned matrix.
#[cfg(feature = "parallel")]
fn copy_to_mat<T: Field + bytemuck::Zeroable>(m: &MatRef<'_, T>) -> Mat<T> {
    let rows = m.nrows();
    let cols = m.ncols();
    let mut result: Mat<T> = Mat::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            result[(i, j)] = m[(i, j)];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strassen_small() {
        // Small matrix should fall back to standard GEMM
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);
        let mut c: Mat<f64> = Mat::zeros(2, 2);

        gemm_strassen(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());

        // A * B = [1*5+2*7  1*6+2*8] = [19 22]
        //         [3*5+4*7  3*6+4*8]   [43 50]
        assert!((c[(0, 0)] - 19.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 22.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 43.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_strassen_medium() {
        // Medium-sized square matrix
        let n = 64;
        let a: Mat<f64> = Mat::filled(n, n, 1.0);
        let b: Mat<f64> = Mat::filled(n, n, 1.0);
        let mut c: Mat<f64> = Mat::zeros(n, n);

        gemm_strassen(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());

        // Each element should be n
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (c[(i, j)] - n as f64).abs() < 1e-10,
                    "c[{},{}] = {}, expected {}",
                    i,
                    j,
                    c[(i, j)],
                    n
                );
            }
        }
    }

    #[test]
    fn test_strassen_with_alpha_beta() {
        let n = 32;
        let a: Mat<f64> = Mat::filled(n, n, 1.0);
        let b: Mat<f64> = Mat::filled(n, n, 2.0);
        let mut c: Mat<f64> = Mat::filled(n, n, 10.0);

        // C = 2 * A * B + 3 * C
        // A * B = n * 2 = 64 (each element)
        // Result = 2 * 64 + 3 * 10 = 128 + 30 = 158
        gemm_strassen(2.0, a.as_ref(), b.as_ref(), 3.0, c.as_mut());

        let expected = 2.0 * (n as f64 * 2.0) + 3.0 * 10.0;
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (c[(i, j)] - expected).abs() < 1e-10,
                    "c[{},{}] = {}, expected {}",
                    i,
                    j,
                    c[(i, j)],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_strassen_non_square() {
        // Non-square, below-threshold matrices exercise the standard GEMM
        // fallback path in `gemm_strassen_with_par`.
        let m = 50;
        let k = 40;
        let n = 60;
        let a: Mat<f64> = Mat::filled(m, k, 1.0);
        let b: Mat<f64> = Mat::filled(k, n, 2.0);
        let mut c: Mat<f64> = Mat::zeros(m, n);

        gemm_strassen(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());

        // Each element should be k * 2 = 80
        let expected = k as f64 * 2.0;
        for i in 0..m {
            for j in 0..n {
                assert!(
                    (c[(i, j)] - expected).abs() < 1e-10,
                    "c[{},{}] = {}, expected {}",
                    i,
                    j,
                    c[(i, j)],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_strassen_non_square_non_power_of_two_large() {
        // Deliberately non-square, non-power-of-two, and with all three
        // dimensions distinct from one another -- while still clearing
        // STRASSEN_THRESHOLD so the real Strassen recursion with dynamic
        // peeling is exercised end to end through the public
        // `gemm_strassen` entry point (m, k, and n are all odd, so every
        // peeling branch -- row, inner, and column -- gets triggered at
        // least once). The result is checked against `gemm_with_blocking`,
        // an independent (non-recursive) GEMM implementation.
        let m = 513;
        let k = 515;
        let n = 517;
        assert!(should_use_strassen(m, k, n), "shape should use Strassen");

        let mut a: Mat<f64> = Mat::zeros(m, k);
        for i in 0..m {
            for j in 0..k {
                a[(i, j)] = ((i * 7 + j * 3 + 1) % 11) as f64;
            }
        }

        let mut b: Mat<f64> = Mat::zeros(k, n);
        for i in 0..k {
            for j in 0..n {
                b[(i, j)] = ((i * 5 + j * 2 + 3) % 13) as f64;
            }
        }

        let mut c_strassen: Mat<f64> = Mat::zeros(m, n);
        gemm_strassen(1.0, a.as_ref(), b.as_ref(), 0.0, c_strassen.as_mut());

        let mut c_reference: Mat<f64> = Mat::zeros(m, n);
        let shape = <f64 as GemmKernel>::micro_kernel_shape();
        let blocking = GemmBlocking::for_kernel::<f64>(&shape);
        gemm_with_blocking(
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c_reference.as_mut(),
            Par::Seq,
            &blocking,
        );

        for i in 0..m {
            for j in 0..n {
                let strassen_val = c_strassen[(i, j)];
                let reference_val = c_reference[(i, j)];
                assert!(
                    (strassen_val - reference_val).abs() < 1e-6,
                    "mismatch at ({i}, {j}): strassen={strassen_val}, reference={reference_val}"
                );
            }
        }
    }

    #[test]
    fn test_strassen_recursive_peeling_arbitrary_shape() {
        // Exercises the private dynamic-peeling recursion directly
        // (bypassing the STRASSEN_THRESHOLD/aspect-ratio gate that the
        // public entry points use) with a non-square, non-power-of-two,
        // wildly rectangular shape in the spirit of the audited example
        // (100x37 times 37x250), scaled just past STRASSEN_LEAF_SIZE so at
        // least one real 2x2 split (in addition to peeling) occurs.
        let m = 101;
        let k = 79;
        let n = 251;

        let mut a: Mat<f64> = Mat::zeros(m, k);
        for i in 0..m {
            for j in 0..k {
                a[(i, j)] = ((i * 3 + j * 7 + 2) % 9) as f64;
            }
        }

        let mut b: Mat<f64> = Mat::zeros(k, n);
        for i in 0..k {
            for j in 0..n {
                b[(i, j)] = ((i * 2 + j * 5 + 1) % 7) as f64;
            }
        }

        let mut c_strassen: Mat<f64> = Mat::zeros(m, n);
        strassen_recursive(
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c_strassen.as_mut(),
            0,
            Par::Seq,
        );

        let mut c_reference: Mat<f64> = Mat::zeros(m, n);
        let shape = <f64 as GemmKernel>::micro_kernel_shape();
        let blocking = GemmBlocking::for_kernel::<f64>(&shape);
        gemm_with_blocking(
            1.0,
            a.as_ref(),
            b.as_ref(),
            0.0,
            c_reference.as_mut(),
            Par::Seq,
            &blocking,
        );

        for i in 0..m {
            for j in 0..n {
                let strassen_val = c_strassen[(i, j)];
                let reference_val = c_reference[(i, j)];
                assert!(
                    (strassen_val - reference_val).abs() < 1e-6,
                    "mismatch at ({i}, {j}): strassen={strassen_val}, reference={reference_val}"
                );
            }
        }
    }

    #[test]
    fn test_strassen_f32() {
        let n = 64;
        let a: Mat<f32> = Mat::filled(n, n, 1.0);
        let b: Mat<f32> = Mat::filled(n, n, 1.0);
        let mut c: Mat<f32> = Mat::zeros(n, n);

        gemm_strassen(1.0f32, a.as_ref(), b.as_ref(), 0.0f32, c.as_mut());

        for i in 0..n {
            for j in 0..n {
                assert!(
                    (c[(i, j)] - n as f32).abs() < 1e-5,
                    "c[{},{}] = {}, expected {}",
                    i,
                    j,
                    c[(i, j)],
                    n
                );
            }
        }
    }

    #[test]
    fn test_strassen_identity() {
        let n = 64;
        let a: Mat<f64> = Mat::eye(n);
        // Create B with unique values
        let mut b: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                b[(i, j)] = (i * n + j) as f64;
            }
        }
        let mut c: Mat<f64> = Mat::zeros(n, n);

        gemm_strassen(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());

        // I * B = B
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (c[(i, j)] - b[(i, j)]).abs() < 1e-10,
                    "c[{},{}] = {}, expected {}",
                    i,
                    j,
                    c[(i, j)],
                    b[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_matrix_add_sub() {
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);
        let mut c: Mat<f64> = Mat::zeros(2, 2);

        matrix_add(&a.as_ref(), &b.as_ref(), &mut c);
        assert!((c[(0, 0)] - 6.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 12.0).abs() < 1e-10);

        matrix_sub(&a.as_ref(), &b.as_ref(), &mut c);
        assert!((c[(0, 0)] - (-4.0)).abs() < 1e-10);
        assert!((c[(1, 1)] - (-4.0)).abs() < 1e-10);
    }

    #[test]
    fn test_should_use_strassen() {
        assert!(!should_use_strassen(100, 100, 100));
        assert!(!should_use_strassen(511, 511, 511));
        assert!(should_use_strassen(512, 512, 512));
        assert!(should_use_strassen(1000, 1000, 1000));
        // The minimum dimension gates eligibility on its own...
        assert!(!should_use_strassen(1000, 100, 1000));
        // ...but a small-enough aspect ratio is also required even once
        // every dimension clears the threshold (see
        // `test_should_use_strassen_rejects_extreme_aspect_ratio`).
    }

    #[test]
    fn test_should_use_strassen_rejects_extreme_aspect_ratio() {
        // All dimensions individually clear STRASSEN_THRESHOLD, but the
        // shape is far too rectangular (ratio >> STRASSEN_MAX_ASPECT_RATIO)
        // for Strassen's recursive halving to pay off versus a single flat
        // GEMM call.
        assert!(!should_use_strassen(8192, 512, 8192));
        // Right at the aspect-ratio boundary: still allowed.
        assert!(should_use_strassen(4096, 512, 4096));
        // Just past the boundary: rejected.
        assert!(!should_use_strassen(4104, 512, 4104));
    }

    /// Minimal per-thread allocation tracker used only by
    /// [`test_strassen_padding_does_not_blow_up_allocations`] to verify
    /// that dynamic peeling keeps Strassen's scratch-memory footprint
    /// proportional to the actual matrix sizes instead of a shared
    /// power-of-two cube of the largest dimension.
    mod alloc_tracking {
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::cell::Cell;

        thread_local! {
            static CURRENT: Cell<usize> = const { Cell::new(0) };
            static PEAK: Cell<usize> = const { Cell::new(0) };
        }

        /// A `GlobalAlloc` wrapper around the system allocator that tracks
        /// the current and peak number of live bytes allocated *on the
        /// calling thread* since the last [`reset`].
        pub struct TrackingAllocator;

        // SAFETY: every call is forwarded unchanged to `System`, which is a
        // valid `GlobalAlloc`; the extra bookkeeping only touches
        // thread-local counters and never affects the returned pointers or
        // their validity.
        unsafe impl GlobalAlloc for TrackingAllocator {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                let ptr = unsafe { System.alloc(layout) };
                if !ptr.is_null() {
                    CURRENT.with(|current| {
                        let updated = current.get() + layout.size();
                        current.set(updated);
                        PEAK.with(|peak| {
                            if updated > peak.get() {
                                peak.set(updated);
                            }
                        });
                    });
                }
                ptr
            }

            unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                unsafe { System.dealloc(ptr, layout) };
                CURRENT.with(|current| {
                    current.set(current.get().saturating_sub(layout.size()));
                });
            }
        }

        /// Resets this thread's current/peak counters to zero.
        pub fn reset() {
            CURRENT.with(|c| c.set(0));
            PEAK.with(|p| p.set(0));
        }

        /// Returns the peak number of bytes concurrently allocated on this
        /// thread since the last [`reset`].
        pub fn peak_bytes() -> usize {
            PEAK.with(Cell::get)
        }
    }

    #[global_allocator]
    static TRACKING_ALLOCATOR: alloc_tracking::TrackingAllocator =
        alloc_tracking::TrackingAllocator;

    #[test]
    fn test_strassen_padding_does_not_blow_up_allocations() {
        // Shape chosen so the OLD implementation (padding every dimension
        // to `next_power_of_two(max(m, k, n))`) would have padded all
        // three dimensions from ~513-517 up to 1024, allocating a
        // 1024x1024 A/B/C triple (~24 MB of f64 scratch) for a multiply
        // whose actual data is only ~6 MB. All three dimensions clear
        // STRASSEN_THRESHOLD (so the Strassen path, not the small-matrix
        // fallback, is what gets measured), and all three are odd (so
        // dynamic peeling is exercised, not just a clean 2x2 split).
        let m = 513usize;
        let k = 515usize;
        let n = 517usize;
        assert!(should_use_strassen(m, k, n), "shape should use Strassen");

        let a: Mat<f64> = Mat::filled(m, k, 1.5);
        let b: Mat<f64> = Mat::filled(k, n, 0.5);
        let mut c: Mat<f64> = Mat::zeros(m, n);

        let elem = std::mem::size_of::<f64>();
        let raw_input_bytes = (m * k + k * n + m * n) * elem;

        // Mirrors the old `next_power_of_two(max(m, k, n))` global padding
        // scheme this fix removes, to quantify how bad the historical
        // blow-up was for this exact shape.
        let old_padded_dim = {
            let max_dim = m.max(k).max(n);
            let mut v = max_dim - 1;
            v |= v >> 1;
            v |= v >> 2;
            v |= v >> 4;
            v |= v >> 8;
            v |= v >> 16;
            v |= v >> 32;
            v + 1
        };
        let old_padded_bytes = 3 * old_padded_dim * old_padded_dim * elem;

        // Sanity check: confirm this shape really would have triggered a
        // large blow-up under the old scheme (old padded scratch is
        // several times larger than the matrices' actual raw data).
        assert!(
            old_padded_bytes > 3 * raw_input_bytes,
            "test shape does not exercise the old blow-up: old={old_padded_bytes}, raw={raw_input_bytes}"
        );

        alloc_tracking::reset();
        gemm_strassen(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());
        let peak = alloc_tracking::peak_bytes();

        // The new dynamic-peeling implementation never materializes a
        // padded copy of the whole matrix: the M1..M7/temp scratch buffers
        // it does allocate form a geometric series proportional to the
        // *actual* matrix sizes, well under half of what the old global
        // power-of-two-cube padding would have required.
        assert!(
            peak < old_padded_bytes / 2,
            "peak scratch allocation {peak} bytes is not meaningfully smaller than the old \
             padded-cube scratch of {old_padded_bytes} bytes -- padding overhead is no longer \
             supposed to scale with the max dimension cubed"
        );

        // Correctness: each output element is k * 1.5 * 0.5 = k * 0.75.
        let expected = k as f64 * 0.75;
        for i in 0..m {
            for j in 0..n {
                assert!(
                    (c[(i, j)] - expected).abs() < 1e-6,
                    "c[{i},{j}] = {}, expected {expected}",
                    c[(i, j)]
                );
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_strassen_parallel() {
        let n = 128;
        let a: Mat<f64> = Mat::filled(n, n, 1.0);
        let b: Mat<f64> = Mat::filled(n, n, 1.0);
        let mut c: Mat<f64> = Mat::zeros(n, n);

        gemm_strassen_parallel(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());

        for i in 0..n {
            for j in 0..n {
                assert!(
                    (c[(i, j)] - n as f64).abs() < 1e-10,
                    "c[{},{}] = {}, expected {}",
                    i,
                    j,
                    c[(i, j)],
                    n
                );
            }
        }
    }
}
