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

    // Check if Strassen is beneficial
    let min_dim = m.min(k).min(n);
    if min_dim < STRASSEN_THRESHOLD {
        // Use standard GEMM for smaller matrices
        let shape = T::micro_kernel_shape();
        let blocking = GemmBlocking::for_kernel::<T>(&shape);
        gemm_with_blocking(alpha, a, b, beta, c.rb_mut(), par, &blocking);
        return;
    }

    // For non-square or non-power-of-2 matrices, pad to the nearest power of 2
    let max_dim = m.max(k).max(n);
    let padded_dim = next_power_of_two(max_dim);

    // Create padded matrices if needed
    if m == padded_dim && k == padded_dim && n == padded_dim {
        // Already square and power of 2
        strassen_recursive(alpha, a, b, beta, c.rb_mut(), 0, par);
    } else {
        // Need to pad
        strassen_with_padding(alpha, a, b, beta, c.rb_mut(), padded_dim, par);
    }
}

/// Strassen with padding for non-square matrices.
fn strassen_with_padding<T: Field + GemmKernel + bytemuck::Zeroable>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    mut c: MatMut<'_, T>,
    padded_dim: usize,
    par: Par,
) {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

    // Create padded matrices
    let mut a_padded: Mat<T> = Mat::zeros(padded_dim, padded_dim);
    let mut b_padded: Mat<T> = Mat::zeros(padded_dim, padded_dim);
    let mut c_padded: Mat<T> = Mat::zeros(padded_dim, padded_dim);

    // Copy A to padded matrix
    for i in 0..m {
        for j in 0..k {
            a_padded[(i, j)] = a[(i, j)];
        }
    }

    // Copy B to padded matrix
    for i in 0..k {
        for j in 0..n {
            b_padded[(i, j)] = b[(i, j)];
        }
    }

    // Copy C to padded matrix and scale if needed
    for i in 0..m {
        for j in 0..n {
            c_padded[(i, j)] = c[(i, j)];
        }
    }

    // Perform Strassen on padded matrices
    strassen_recursive(
        alpha,
        a_padded.as_ref(),
        b_padded.as_ref(),
        beta,
        c_padded.as_mut(),
        0,
        par,
    );

    // Copy result back
    for i in 0..m {
        for j in 0..n {
            c.set(i, j, c_padded[(i, j)]);
        }
    }
}

/// Recursive Strassen implementation.
fn strassen_recursive<T: Field + GemmKernel + bytemuck::Zeroable>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    mut c: MatMut<'_, T>,
    depth: usize,
    par: Par,
) {
    let n = a.nrows();
    debug_assert_eq!(n, a.ncols());
    debug_assert_eq!(n, b.nrows());
    debug_assert_eq!(n, b.ncols());
    debug_assert_eq!(n, c.nrows());
    debug_assert_eq!(n, c.ncols());

    // Base case: use standard GEMM
    if n <= STRASSEN_LEAF_SIZE || depth >= MAX_STRASSEN_DEPTH {
        let shape = T::micro_kernel_shape();
        let blocking = GemmBlocking::for_kernel::<T>(&shape);
        gemm_with_blocking(alpha, a, b, beta, c.rb_mut(), par, &blocking);
        return;
    }

    let half = n / 2;

    // Partition matrices into quadrants
    let a11 = a.submatrix(0, 0, half, half);
    let a12 = a.submatrix(0, half, half, half);
    let a21 = a.submatrix(half, 0, half, half);
    let a22 = a.submatrix(half, half, half, half);

    let b11 = b.submatrix(0, 0, half, half);
    let b12 = b.submatrix(0, half, half, half);
    let b21 = b.submatrix(half, 0, half, half);
    let b22 = b.submatrix(half, half, half, half);

    // Allocate intermediate matrices for M1-M7
    let mut m1: Mat<T> = Mat::zeros(half, half);
    let mut m2: Mat<T> = Mat::zeros(half, half);
    let mut m3: Mat<T> = Mat::zeros(half, half);
    let mut m4: Mat<T> = Mat::zeros(half, half);
    let mut m5: Mat<T> = Mat::zeros(half, half);
    let mut m6: Mat<T> = Mat::zeros(half, half);
    let mut m7: Mat<T> = Mat::zeros(half, half);

    // Temporary matrices for sums
    let mut temp1: Mat<T> = Mat::zeros(half, half);
    let mut temp2: Mat<T> = Mat::zeros(half, half);

    // M1 = (A11 + A22)(B11 + B22)
    matrix_add(&a11, &a22, &mut temp1);
    matrix_add(&b11, &b22, &mut temp2);
    strassen_recursive(
        T::one(),
        temp1.as_ref(),
        temp2.as_ref(),
        T::zero(),
        m1.as_mut(),
        depth + 1,
        par,
    );

    // M2 = (A21 + A22)B11
    matrix_add(&a21, &a22, &mut temp1);
    strassen_recursive(
        T::one(),
        temp1.as_ref(),
        b11,
        T::zero(),
        m2.as_mut(),
        depth + 1,
        par,
    );

    // M3 = A11(B12 - B22)
    matrix_sub(&b12, &b22, &mut temp2);
    strassen_recursive(
        T::one(),
        a11,
        temp2.as_ref(),
        T::zero(),
        m3.as_mut(),
        depth + 1,
        par,
    );

    // M4 = A22(B21 - B11)
    matrix_sub(&b21, &b11, &mut temp2);
    strassen_recursive(
        T::one(),
        a22,
        temp2.as_ref(),
        T::zero(),
        m4.as_mut(),
        depth + 1,
        par,
    );

    // M5 = (A11 + A12)B22
    matrix_add(&a11, &a12, &mut temp1);
    strassen_recursive(
        T::one(),
        temp1.as_ref(),
        b22,
        T::zero(),
        m5.as_mut(),
        depth + 1,
        par,
    );

    // M6 = (A21 - A11)(B11 + B12)
    matrix_sub(&a21, &a11, &mut temp1);
    matrix_add(&b11, &b12, &mut temp2);
    strassen_recursive(
        T::one(),
        temp1.as_ref(),
        temp2.as_ref(),
        T::zero(),
        m6.as_mut(),
        depth + 1,
        par,
    );

    // M7 = (A12 - A22)(B21 + B22)
    matrix_sub(&a12, &a22, &mut temp1);
    matrix_add(&b21, &b22, &mut temp2);
    strassen_recursive(
        T::one(),
        temp1.as_ref(),
        temp2.as_ref(),
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

    // Get mutable quadrants of C
    for i in 0..half {
        for j in 0..half {
            // C11 = alpha * (M1 + M4 - M5 + M7) + beta * C11
            let c11_contrib = m1[(i, j)] + m4[(i, j)] - m5[(i, j)] + m7[(i, j)];
            if beta == T::zero() {
                c.set(i, j, alpha * c11_contrib);
            } else if beta == T::one() {
                c.set(i, j, c[(i, j)] + alpha * c11_contrib);
            } else {
                c.set(i, j, c[(i, j)] + alpha * c11_contrib);
            }

            // C12 = alpha * (M3 + M5) + beta * C12
            let c12_contrib = m3[(i, j)] + m5[(i, j)];
            if beta == T::zero() {
                c.set(i, j + half, alpha * c12_contrib);
            } else if beta == T::one() {
                c.set(i, j + half, c[(i, j + half)] + alpha * c12_contrib);
            } else {
                c.set(i, j + half, c[(i, j + half)] + alpha * c12_contrib);
            }

            // C21 = alpha * (M2 + M4) + beta * C21
            let c21_contrib = m2[(i, j)] + m4[(i, j)];
            if beta == T::zero() {
                c.set(i + half, j, alpha * c21_contrib);
            } else if beta == T::one() {
                c.set(i + half, j, c[(i + half, j)] + alpha * c21_contrib);
            } else {
                c.set(i + half, j, c[(i + half, j)] + alpha * c21_contrib);
            }

            // C22 = alpha * (M1 - M2 + M3 + M6) + beta * C22
            let c22_contrib = m1[(i, j)] - m2[(i, j)] + m3[(i, j)] + m6[(i, j)];
            if beta == T::zero() {
                c.set(i + half, j + half, alpha * c22_contrib);
            } else if beta == T::one() {
                c.set(
                    i + half,
                    j + half,
                    c[(i + half, j + half)] + alpha * c22_contrib,
                );
            } else {
                c.set(
                    i + half,
                    j + half,
                    c[(i + half, j + half)] + alpha * c22_contrib,
                );
            }
        }
    }
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

/// Returns the next power of two >= n.
#[inline]
const fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut v = n - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v + 1
}

/// Checks if Strassen's algorithm would be beneficial for the given dimensions.
///
/// Returns true if at least one dimension exceeds the Strassen threshold.
#[must_use]
pub fn should_use_strassen(m: usize, k: usize, n: usize) -> bool {
    let min_dim = m.min(k).min(n);
    min_dim >= STRASSEN_THRESHOLD
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

    let min_dim = m.min(k).min(n);
    if min_dim < STRASSEN_THRESHOLD {
        let shape = T::micro_kernel_shape();
        let blocking = GemmBlocking::for_kernel::<T>(&shape);
        gemm_with_blocking(alpha, a, b, beta, c.rb_mut(), Par::Rayon, &blocking);
        return;
    }

    let max_dim = m.max(k).max(n);
    let padded_dim = next_power_of_two(max_dim);

    if m == padded_dim && k == padded_dim && n == padded_dim {
        strassen_recursive_parallel(alpha, a, b, beta, c.rb_mut(), 0);
    } else {
        strassen_with_padding_parallel(alpha, a, b, beta, c.rb_mut(), padded_dim);
    }
}

/// Parallel version of strassen_with_padding.
#[cfg(feature = "parallel")]
fn strassen_with_padding_parallel<T: Field + GemmKernel + bytemuck::Zeroable + Send + Sync>(
    alpha: T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    beta: T,
    mut c: MatMut<'_, T>,
    padded_dim: usize,
) where
    Mat<T>: Send + Sync,
{
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

    let mut a_padded: Mat<T> = Mat::zeros(padded_dim, padded_dim);
    let mut b_padded: Mat<T> = Mat::zeros(padded_dim, padded_dim);
    let mut c_padded: Mat<T> = Mat::zeros(padded_dim, padded_dim);

    for i in 0..m {
        for j in 0..k {
            a_padded[(i, j)] = a[(i, j)];
        }
    }

    for i in 0..k {
        for j in 0..n {
            b_padded[(i, j)] = b[(i, j)];
        }
    }

    for i in 0..m {
        for j in 0..n {
            c_padded[(i, j)] = c[(i, j)];
        }
    }

    strassen_recursive_parallel(
        alpha,
        a_padded.as_ref(),
        b_padded.as_ref(),
        beta,
        c_padded.as_mut(),
        0,
    );

    for i in 0..m {
        for j in 0..n {
            c.set(i, j, c_padded[(i, j)]);
        }
    }
}

/// Parallel recursive Strassen.
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

    let n = a.nrows();

    if n <= STRASSEN_LEAF_SIZE || depth >= MAX_STRASSEN_DEPTH {
        let shape = T::micro_kernel_shape();
        let blocking = GemmBlocking::for_kernel::<T>(&shape);
        gemm_with_blocking(alpha, a, b, beta, c.rb_mut(), Par::Rayon, &blocking);
        return;
    }

    let half = n / 2;

    // Partition matrices
    let a11 = a.submatrix(0, 0, half, half);
    let a12 = a.submatrix(0, half, half, half);
    let a21 = a.submatrix(half, 0, half, half);
    let a22 = a.submatrix(half, half, half, half);

    let b11 = b.submatrix(0, 0, half, half);
    let b12 = b.submatrix(0, half, half, half);
    let b21 = b.submatrix(half, 0, half, half);
    let b22 = b.submatrix(half, half, half, half);

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
                let mut temp1: Mat<T> = Mat::zeros(half, half);
                let mut temp2: Mat<T> = Mat::zeros(half, half);
                let mut result: Mat<T> = Mat::zeros(half, half);

                match idx {
                    0 => {
                        // M1 = (A11 + A22)(B11 + B22)
                        matrix_add(&a11_owned.as_ref(), &a22_owned.as_ref(), &mut temp1);
                        matrix_add(&b11_owned.as_ref(), &b22_owned.as_ref(), &mut temp2);
                        strassen_recursive_parallel(
                            T::one(),
                            temp1.as_ref(),
                            temp2.as_ref(),
                            T::zero(),
                            result.as_mut(),
                            depth + 1,
                        );
                    }
                    1 => {
                        // M2 = (A21 + A22)B11
                        matrix_add(&a21_owned.as_ref(), &a22_owned.as_ref(), &mut temp1);
                        strassen_recursive_parallel(
                            T::one(),
                            temp1.as_ref(),
                            b11_owned.as_ref(),
                            T::zero(),
                            result.as_mut(),
                            depth + 1,
                        );
                    }
                    2 => {
                        // M3 = A11(B12 - B22)
                        matrix_sub(&b12_owned.as_ref(), &b22_owned.as_ref(), &mut temp2);
                        strassen_recursive_parallel(
                            T::one(),
                            a11_owned.as_ref(),
                            temp2.as_ref(),
                            T::zero(),
                            result.as_mut(),
                            depth + 1,
                        );
                    }
                    3 => {
                        // M4 = A22(B21 - B11)
                        matrix_sub(&b21_owned.as_ref(), &b11_owned.as_ref(), &mut temp2);
                        strassen_recursive_parallel(
                            T::one(),
                            a22_owned.as_ref(),
                            temp2.as_ref(),
                            T::zero(),
                            result.as_mut(),
                            depth + 1,
                        );
                    }
                    4 => {
                        // M5 = (A11 + A12)B22
                        matrix_add(&a11_owned.as_ref(), &a12_owned.as_ref(), &mut temp1);
                        strassen_recursive_parallel(
                            T::one(),
                            temp1.as_ref(),
                            b22_owned.as_ref(),
                            T::zero(),
                            result.as_mut(),
                            depth + 1,
                        );
                    }
                    5 => {
                        // M6 = (A21 - A11)(B11 + B12)
                        matrix_sub(&a21_owned.as_ref(), &a11_owned.as_ref(), &mut temp1);
                        matrix_add(&b11_owned.as_ref(), &b12_owned.as_ref(), &mut temp2);
                        strassen_recursive_parallel(
                            T::one(),
                            temp1.as_ref(),
                            temp2.as_ref(),
                            T::zero(),
                            result.as_mut(),
                            depth + 1,
                        );
                    }
                    6 => {
                        // M7 = (A12 - A22)(B21 + B22)
                        matrix_sub(&a12_owned.as_ref(), &a22_owned.as_ref(), &mut temp1);
                        matrix_add(&b21_owned.as_ref(), &b22_owned.as_ref(), &mut temp2);
                        strassen_recursive_parallel(
                            T::one(),
                            temp1.as_ref(),
                            temp2.as_ref(),
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
        let mut m1: Mat<T> = Mat::zeros(half, half);
        let mut m2: Mat<T> = Mat::zeros(half, half);
        let mut m3: Mat<T> = Mat::zeros(half, half);
        let mut m4: Mat<T> = Mat::zeros(half, half);
        let mut m5: Mat<T> = Mat::zeros(half, half);
        let mut m6: Mat<T> = Mat::zeros(half, half);
        let mut m7: Mat<T> = Mat::zeros(half, half);
        let mut temp1: Mat<T> = Mat::zeros(half, half);
        let mut temp2: Mat<T> = Mat::zeros(half, half);

        matrix_add(&a11_owned.as_ref(), &a22_owned.as_ref(), &mut temp1);
        matrix_add(&b11_owned.as_ref(), &b22_owned.as_ref(), &mut temp2);
        strassen_recursive_parallel(
            T::one(),
            temp1.as_ref(),
            temp2.as_ref(),
            T::zero(),
            m1.as_mut(),
            depth + 1,
        );

        matrix_add(&a21_owned.as_ref(), &a22_owned.as_ref(), &mut temp1);
        strassen_recursive_parallel(
            T::one(),
            temp1.as_ref(),
            b11_owned.as_ref(),
            T::zero(),
            m2.as_mut(),
            depth + 1,
        );

        matrix_sub(&b12_owned.as_ref(), &b22_owned.as_ref(), &mut temp2);
        strassen_recursive_parallel(
            T::one(),
            a11_owned.as_ref(),
            temp2.as_ref(),
            T::zero(),
            m3.as_mut(),
            depth + 1,
        );

        matrix_sub(&b21_owned.as_ref(), &b11_owned.as_ref(), &mut temp2);
        strassen_recursive_parallel(
            T::one(),
            a22_owned.as_ref(),
            temp2.as_ref(),
            T::zero(),
            m4.as_mut(),
            depth + 1,
        );

        matrix_add(&a11_owned.as_ref(), &a12_owned.as_ref(), &mut temp1);
        strassen_recursive_parallel(
            T::one(),
            temp1.as_ref(),
            b22_owned.as_ref(),
            T::zero(),
            m5.as_mut(),
            depth + 1,
        );

        matrix_sub(&a21_owned.as_ref(), &a11_owned.as_ref(), &mut temp1);
        matrix_add(&b11_owned.as_ref(), &b12_owned.as_ref(), &mut temp2);
        strassen_recursive_parallel(
            T::one(),
            temp1.as_ref(),
            temp2.as_ref(),
            T::zero(),
            m6.as_mut(),
            depth + 1,
        );

        matrix_sub(&a12_owned.as_ref(), &a22_owned.as_ref(), &mut temp1);
        matrix_add(&b21_owned.as_ref(), &b22_owned.as_ref(), &mut temp2);
        strassen_recursive_parallel(
            T::one(),
            temp1.as_ref(),
            temp2.as_ref(),
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

    for i in 0..half {
        for j in 0..half {
            let c11_contrib = m1[(i, j)] + m4[(i, j)] - m5[(i, j)] + m7[(i, j)];
            let c12_contrib = m3[(i, j)] + m5[(i, j)];
            let c21_contrib = m2[(i, j)] + m4[(i, j)];
            let c22_contrib = m1[(i, j)] - m2[(i, j)] + m3[(i, j)] + m6[(i, j)];

            if beta == T::zero() {
                c.set(i, j, alpha * c11_contrib);
                c.set(i, j + half, alpha * c12_contrib);
                c.set(i + half, j, alpha * c21_contrib);
                c.set(i + half, j + half, alpha * c22_contrib);
            } else {
                c.set(i, j, c[(i, j)] + alpha * c11_contrib);
                c.set(i, j + half, c[(i, j + half)] + alpha * c12_contrib);
                c.set(i + half, j, c[(i + half, j)] + alpha * c21_contrib);
                c.set(
                    i + half,
                    j + half,
                    c[(i + half, j + half)] + alpha * c22_contrib,
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
        // Non-square matrices should be handled correctly with padding
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
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(2), 2);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(4), 4);
        assert_eq!(next_power_of_two(5), 8);
        assert_eq!(next_power_of_two(100), 128);
        assert_eq!(next_power_of_two(512), 512);
        assert_eq!(next_power_of_two(513), 1024);
    }

    #[test]
    fn test_should_use_strassen() {
        assert!(!should_use_strassen(100, 100, 100));
        assert!(!should_use_strassen(511, 511, 511));
        assert!(should_use_strassen(512, 512, 512));
        assert!(should_use_strassen(1000, 1000, 1000));
        // Only the minimum dimension matters
        assert!(!should_use_strassen(1000, 100, 1000));
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
