//! Parallel eigenvalue decomposition algorithms.
//!
//! This module provides parallelized versions of eigenvalue decomposition algorithms
//! using Rayon for multi-threaded execution. These algorithms achieve significant
//! speedups on multi-core systems for large matrices.
//!
//! # Algorithms
//!
//! - **ParallelSymmetricEvd**: Parallel divide-and-conquer for symmetric matrices
//! - **parallel_bisection**: Parallel bisection for tridiagonal eigenvalues
//! - **parallel_inverse_iteration**: Parallel inverse iteration for eigenvectors
//!
//! # Example
//!
//! ```ignore
//! use oxiblas_lapack::evd::ParallelSymmetricEvd;
//! use oxiblas_matrix::Mat;
//!
//! let a = Mat::from_rows(&[
//!     &[4.0f64, 1.0, 0.0],
//!     &[1.0, 3.0, 1.0],
//!     &[0.0, 1.0, 2.0],
//! ]);
//!
//! let evd = ParallelSymmetricEvd::compute(a.as_ref()).unwrap();
//! let eigenvalues = evd.eigenvalues();
//! ```

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};
use rayon::prelude::*;

/// Minimum matrix size to use parallel algorithms.
/// Below this threshold, sequential algorithms are more efficient.
const PARALLEL_THRESHOLD: usize = 64;

/// Block size for parallel matrix operations.
const PARALLEL_BLOCK_SIZE: usize = 32;

/// Maximum iterations for secular equation solver.
const MAX_SECULAR_ITER: usize = 100;

/// Maximum iterations for inverse iteration.
const MAX_INVERSE_ITER: usize = 50;

/// Maximum bisection iterations.
const MAX_BISECTION_ITER: usize = 1000;

/// Error type for parallel eigenvalue decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelEvdError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix is not square.
    NotSquare,
    /// Algorithm did not converge.
    NotConverged,
    /// Secular equation solver failed.
    SecularEquationFailed,
    /// Invalid dimension.
    InvalidDimension,
}

impl core::fmt::Display for ParallelEvdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::NotSquare => write!(f, "Matrix is not square"),
            Self::NotConverged => write!(f, "Algorithm did not converge"),
            Self::SecularEquationFailed => write!(f, "Secular equation solver failed"),
            Self::InvalidDimension => write!(f, "Invalid dimension"),
        }
    }
}

impl std::error::Error for ParallelEvdError {}

/// Parallel symmetric eigenvalue decomposition using divide-and-conquer.
///
/// Uses multi-threaded execution for:
/// - Recursive subproblem solutions
/// - Secular equation solving
/// - Eigenvector computation
/// - Matrix multiplications
#[derive(Debug, Clone)]
pub struct ParallelSymmetricEvd<T: Scalar> {
    /// Eigenvalues (sorted in ascending order).
    eigenvalues: Vec<T>,
    /// Eigenvectors (columns of V).
    eigenvectors: Mat<T>,
    /// Matrix dimension.
    n: usize,
}

impl<T: Field + Real + bytemuck::Zeroable + Send + Sync> ParallelSymmetricEvd<T> {
    /// Computes the eigendecomposition using parallel divide-and-conquer.
    ///
    /// # Arguments
    ///
    /// * `a` - Symmetric matrix (only upper triangle is used)
    ///
    /// # Example
    ///
    /// ```ignore
    /// use oxiblas_lapack::evd::ParallelSymmetricEvd;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[2.0f64, 1.0],
    ///     &[1.0, 2.0],
    /// ]);
    ///
    /// let evd = ParallelSymmetricEvd::compute(a.as_ref()).unwrap();
    /// ```
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, ParallelEvdError> {
        let n = a.nrows();

        if n == 0 {
            return Err(ParallelEvdError::EmptyMatrix);
        }
        if n != a.ncols() {
            return Err(ParallelEvdError::NotSquare);
        }

        // Handle trivial case
        if n == 1 {
            let eigenvalues = vec![a[(0, 0)]];
            let mut eigenvectors = Mat::zeros(1, 1);
            eigenvectors[(0, 0)] = T::one();
            return Ok(Self {
                eigenvalues,
                eigenvectors,
                n,
            });
        }

        // Copy symmetric matrix (use upper triangle)
        let mut work = Mat::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let val = a[(i, j)];
                work[(i, j)] = val;
                work[(j, i)] = val;
            }
        }

        // Initialize eigenvector matrix to identity
        let mut v = Mat::zeros(n, n);
        for i in 0..n {
            v[(i, i)] = T::one();
        }

        // Tridiagonalize: A = Q * T * Q^T
        let (diag, off_diag) = tridiagonalize(&mut work, &mut v, n);

        // Apply parallel divide-and-conquer algorithm to tridiagonal matrix
        let eigenvalues = parallel_divide_and_conquer(diag, off_diag, &mut v, n)?;

        Ok(Self {
            eigenvalues,
            eigenvectors: v,
            n,
        })
    }

    /// Returns the eigenvalues (sorted in ascending order).
    pub fn eigenvalues(&self) -> &[T] {
        &self.eigenvalues
    }

    /// Returns the eigenvector matrix V.
    pub fn eigenvectors(&self) -> MatRef<'_, T> {
        self.eigenvectors.as_ref()
    }

    /// Returns the dimension of the matrix.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Reconstructs the original matrix: A = V * D * V^T
    pub fn reconstruct(&self) -> Mat<T> {
        let n = self.n;
        let mut a = Mat::zeros(n, n);

        // Parallel reconstruction for large matrices
        if n >= PARALLEL_THRESHOLD {
            // Compute A = V * D * V^T in parallel blocks
            let chunk_size = PARALLEL_BLOCK_SIZE;
            let eigenvalues = &self.eigenvalues;
            let eigenvectors = &self.eigenvectors;

            // Flatten indices for parallel iteration
            let pairs: Vec<(usize, usize)> =
                (0..n).flat_map(|i| (0..=i).map(move |j| (i, j))).collect();

            let results: Vec<(usize, usize, T)> = pairs
                .par_chunks(chunk_size)
                .flat_map(|chunk| {
                    chunk
                        .iter()
                        .map(|&(i, j)| {
                            let mut sum = T::zero();
                            for k in 0..n {
                                let lambda = eigenvalues[k];
                                sum = sum + lambda * eigenvectors[(i, k)] * eigenvectors[(j, k)];
                            }
                            (i, j, sum)
                        })
                        .collect::<Vec<_>>()
                })
                .collect();

            for (i, j, val) in results {
                a[(i, j)] = val;
                if i != j {
                    a[(j, i)] = val;
                }
            }
        } else {
            // Sequential for small matrices
            for k in 0..n {
                let lambda = self.eigenvalues[k];
                for i in 0..n {
                    for j in 0..n {
                        a[(i, j)] = a[(i, j)]
                            + lambda * self.eigenvectors[(i, k)] * self.eigenvectors[(j, k)];
                    }
                }
            }
        }

        a
    }
}

/// Tridiagonalizes a symmetric matrix using Householder reflections.
fn tridiagonalize<T: Field + Real>(a: &mut Mat<T>, v: &mut Mat<T>, n: usize) -> (Vec<T>, Vec<T>) {
    let mut diag = vec![T::zero(); n];
    let mut off_diag = vec![T::zero(); n.saturating_sub(1)];

    for k in 0..(n.saturating_sub(2)) {
        // Compute Householder vector for column k
        let mut norm_sq = T::zero();
        for i in (k + 1)..n {
            norm_sq = norm_sq + a[(i, k)] * a[(i, k)];
        }
        let norm = Real::sqrt(norm_sq);

        if norm > T::zero() {
            let x_k1 = a[(k + 1, k)];
            let beta = if x_k1 >= T::zero() { -norm } else { norm };

            let tau = (beta - x_k1) / beta;
            let scale = T::one() / (x_k1 - beta);
            for i in (k + 2)..n {
                a[(i, k)] = a[(i, k)] * scale;
            }

            // Apply Householder from left and right
            let mut p = vec![T::zero(); n];
            for i in (k + 1)..n {
                for j in (k + 1)..n {
                    let v_j = if j == k + 1 { T::one() } else { a[(j, k)] };
                    p[i] = p[i] + a[(i, j)] * v_j;
                }
                p[i] = tau * p[i];
            }

            let mut ptv = T::zero();
            for i in (k + 1)..n {
                let v_i = if i == k + 1 { T::one() } else { a[(i, k)] };
                ptv = ptv + p[i] * v_i;
            }
            let half_tau = tau / (T::one() + T::one());

            let mut w = vec![T::zero(); n];
            for i in (k + 1)..n {
                let v_i = if i == k + 1 { T::one() } else { a[(i, k)] };
                w[i] = p[i] - half_tau * ptv * v_i;
            }

            for i in (k + 1)..n {
                let v_i = if i == k + 1 { T::one() } else { a[(i, k)] };
                for j in (k + 1)..n {
                    let v_j = if j == k + 1 { T::one() } else { a[(j, k)] };
                    a[(i, j)] = a[(i, j)] - v_i * w[j] - w[i] * v_j;
                }
            }

            // Update V
            for i in 0..n {
                let mut vv = T::zero();
                for j in (k + 1)..n {
                    let v_j = if j == k + 1 { T::one() } else { a[(j, k)] };
                    vv = vv + v[(i, j)] * v_j;
                }
                let tau_vv = tau * vv;
                for j in (k + 1)..n {
                    let v_j = if j == k + 1 { T::one() } else { a[(j, k)] };
                    v[(i, j)] = v[(i, j)] - tau_vv * v_j;
                }
            }

            off_diag[k] = beta;
        }
    }

    for i in 0..n {
        diag[i] = a[(i, i)];
    }
    if n >= 2 {
        off_diag[n - 2] = a[(n - 1, n - 2)];
    }

    (diag, off_diag)
}

/// Parallel divide-and-conquer algorithm for symmetric tridiagonal eigenvalue problem.
fn parallel_divide_and_conquer<T: Field + Real + bytemuck::Zeroable + Send + Sync>(
    diag: Vec<T>,
    off_diag: Vec<T>,
    v: &mut Mat<T>,
    n: usize,
) -> Result<Vec<T>, ParallelEvdError> {
    if n <= 1 {
        return Ok(diag);
    }

    // For small matrices, use sequential QR
    if n <= PARALLEL_THRESHOLD {
        return qr_algorithm_sequential(diag, off_diag, v, n);
    }

    // Divide at the middle
    let m = n / 2;
    let beta = off_diag[m - 1];

    // Create sub-problems
    let mut diag1: Vec<T> = diag[0..m].to_vec();
    let mut diag2: Vec<T> = diag[m..n].to_vec();

    let abs_beta = Scalar::abs(beta);
    diag1[m - 1] = diag1[m - 1] - abs_beta;
    diag2[0] = diag2[0] - abs_beta;

    let off_diag1: Vec<T> = if m > 1 {
        off_diag[0..(m - 1)].to_vec()
    } else {
        vec![]
    };
    let off_diag2: Vec<T> = if n - m > 1 {
        off_diag[m..(n - 1)].to_vec()
    } else {
        vec![]
    };

    // Solve subproblems in parallel using rayon
    let (result1, result2) = rayon::join(
        || {
            let mut v1: Mat<T> = Mat::zeros(m, m);
            for i in 0..m {
                v1[(i, i)] = T::one();
            }
            parallel_divide_and_conquer(diag1.clone(), off_diag1.clone(), &mut v1, m)
                .map(|e| (e, v1))
        },
        || {
            let mut v2: Mat<T> = Mat::zeros(n - m, n - m);
            for i in 0..(n - m) {
                v2[(i, i)] = T::one();
            }
            parallel_divide_and_conquer(diag2.clone(), off_diag2.clone(), &mut v2, n - m)
                .map(|e| (e, v2))
        },
    );

    let (eig1, v1) = result1?;
    let (eig2, v2) = result2?;

    // Merge results
    let rho = abs_beta;

    // Build z vector
    let mut z = vec![T::zero(); n];
    for i in 0..m {
        z[i] = v1[(m - 1, i)];
    }
    for i in 0..(n - m) {
        z[m + i] = v2[(0, i)];
    }

    if beta < T::zero() {
        for i in 0..(n - m) {
            z[m + i] = -z[m + i];
        }
    }

    // Combine and sort eigenvalues
    let mut d: Vec<T> = Vec::with_capacity(n);
    d.extend_from_slice(&eig1);
    d.extend_from_slice(&eig2);

    let mut perm: Vec<usize> = (0..n).collect();
    perm.sort_by(|&i, &j| d[i].partial_cmp(&d[j]).unwrap_or(std::cmp::Ordering::Equal));

    let d_sorted: Vec<T> = perm.iter().map(|&i| d[i]).collect();
    let z_sorted: Vec<T> = perm.iter().map(|&i| z[i]).collect();

    // Solve secular equations in parallel
    let merged_eigenvalues = parallel_solve_secular_equations(&d_sorted, &z_sorted, rho, n)?;

    // Compute merged eigenvectors in parallel
    let merged_v =
        parallel_compute_merged_eigenvectors(&d_sorted, &z_sorted, &merged_eigenvalues, n);

    // Apply permutation
    let mut unperm_v: Mat<T> = Mat::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            unperm_v[(perm[i], j)] = merged_v[(i, j)];
        }
    }

    // Combine with subproblem eigenvectors using parallel matrix multiplication
    let combined = parallel_combine_eigenvectors(&v1, &v2, &unperm_v, m, n);

    // Apply accumulated transformation in parallel
    let v_copy = v.clone();
    parallel_matmul_into(&v_copy, &combined, v, n);

    // Sort final eigenvalues and eigenvectors
    let mut eigenvalues = merged_eigenvalues;
    sort_eigenvalues(&mut eigenvalues, v, n);

    Ok(eigenvalues)
}

/// Solve secular equations in parallel.
fn parallel_solve_secular_equations<T: Field + Real + Send + Sync>(
    d: &[T],
    z: &[T],
    rho: T,
    n: usize,
) -> Result<Vec<T>, ParallelEvdError> {
    let indices: Vec<usize> = (0..n).collect();

    let results: Vec<Result<T, ParallelEvdError>> = indices
        .par_iter()
        .map(|&k| {
            let (lower, upper) = if k < n - 1 {
                (d[k], d[k + 1])
            } else {
                let sum_z_sq: T = z.iter().fold(T::zero(), |acc, &zi| acc + zi * zi);
                (d[n - 1], d[n - 1] + rho * sum_z_sq)
            };

            solve_single_secular(d, z, rho, lower, upper, k)
        })
        .collect();

    let mut eigenvalues = Vec::with_capacity(n);
    for result in results {
        eigenvalues.push(result?);
    }

    Ok(eigenvalues)
}

/// Solve a single secular equation.
fn solve_single_secular<T: Field + Real>(
    d: &[T],
    z: &[T],
    rho: T,
    lower: T,
    upper: T,
    k: usize,
) -> Result<T, ParallelEvdError> {
    let eps = <T as Scalar>::epsilon();
    let tol = eps * T::from_f64(100.0).unwrap_or(T::one());

    if Scalar::abs(z[k]) < tol {
        return Ok(d[k]);
    }

    let two = T::one() + T::one();
    let mut lambda = (lower + upper) / two;
    let mut a = lower;
    let mut b = upper;

    for _ in 0..MAX_SECULAR_ITER {
        let (f, df) = secular_function_and_derivative(d, z, rho, lambda);

        if Scalar::abs(f) < tol * (T::one() + Scalar::abs(lambda)) {
            return Ok(lambda);
        }

        let delta = f / df;
        let lambda_new = lambda - delta;

        if lambda_new > a && lambda_new < b {
            lambda = lambda_new;
        } else {
            lambda = (a + b) / two;
        }

        let (f_new, _) = secular_function_and_derivative(d, z, rho, lambda);
        if f_new < T::zero() {
            a = lambda;
        } else {
            b = lambda;
        }

        if b - a < tol * (T::one() + Scalar::abs(lambda)) {
            return Ok((a + b) / two);
        }
    }

    Ok(lambda)
}

/// Evaluate secular function and derivative.
fn secular_function_and_derivative<T: Field + Real>(d: &[T], z: &[T], rho: T, lambda: T) -> (T, T) {
    let n = d.len();
    let mut f = T::one();
    let mut df = T::zero();

    for i in 0..n {
        let zi_sq = z[i] * z[i];
        let diff = d[i] - lambda;
        if Scalar::abs(diff) > <T as Scalar>::epsilon() {
            f = f + rho * zi_sq / diff;
            df = df + rho * zi_sq / (diff * diff);
        }
    }

    (f, df)
}

/// Compute merged eigenvectors in parallel.
fn parallel_compute_merged_eigenvectors<T: Field + Real + bytemuck::Zeroable + Send + Sync>(
    d: &[T],
    z: &[T],
    eigenvalues: &[T],
    n: usize,
) -> Mat<T> {
    let indices: Vec<usize> = (0..n).collect();

    let columns: Vec<Vec<T>> = indices
        .par_iter()
        .map(|&j| {
            let lambda = eigenvalues[j];
            let mut col = vec![T::zero(); n];
            let mut norm_sq = T::zero();

            for i in 0..n {
                let diff = d[i] - lambda;
                if Scalar::abs(diff) > <T as Scalar>::epsilon() {
                    col[i] = z[i] / diff;
                } else {
                    col[i] = T::one();
                }
                norm_sq = norm_sq + col[i] * col[i];
            }

            let norm = Real::sqrt(norm_sq);
            if norm > T::zero() {
                for i in 0..n {
                    col[i] = col[i] / norm;
                }
            }

            col
        })
        .collect();

    let mut v: Mat<T> = Mat::zeros(n, n);
    for (j, col) in columns.into_iter().enumerate() {
        for (i, val) in col.into_iter().enumerate() {
            v[(i, j)] = val;
        }
    }

    v
}

/// Combine eigenvectors from subproblems in parallel.
fn parallel_combine_eigenvectors<T: Field + Real + bytemuck::Zeroable + Send + Sync>(
    v1: &Mat<T>,
    v2: &Mat<T>,
    unperm_v: &Mat<T>,
    m: usize,
    n: usize,
) -> Mat<T> {
    let indices: Vec<usize> = (0..n).collect();

    let columns: Vec<Vec<T>> = indices
        .par_iter()
        .map(|&j| {
            let mut col = vec![T::zero(); n];

            // Upper part: v1 * unperm_v[0..m, j]
            for i in 0..m {
                let mut sum = T::zero();
                for k in 0..m {
                    sum = sum + v1[(i, k)] * unperm_v[(k, j)];
                }
                col[i] = sum;
            }

            // Lower part: v2 * unperm_v[m..n, j]
            for i in 0..(n - m) {
                let mut sum = T::zero();
                for k in 0..(n - m) {
                    sum = sum + v2[(i, k)] * unperm_v[(m + k, j)];
                }
                col[m + i] = sum;
            }

            col
        })
        .collect();

    let mut combined: Mat<T> = Mat::zeros(n, n);
    for (j, col) in columns.into_iter().enumerate() {
        for (i, val) in col.into_iter().enumerate() {
            combined[(i, j)] = val;
        }
    }

    combined
}

/// Parallel matrix multiplication into destination.
fn parallel_matmul_into<T: Field + Real + bytemuck::Zeroable + Send + Sync>(
    a: &Mat<T>,
    b: &Mat<T>,
    c: &mut Mat<T>,
    n: usize,
) {
    let indices: Vec<usize> = (0..n).collect();

    let columns: Vec<Vec<T>> = indices
        .par_iter()
        .map(|&j| {
            let mut col = vec![T::zero(); n];
            for i in 0..n {
                let mut sum = T::zero();
                for k in 0..n {
                    sum = sum + a[(i, k)] * b[(k, j)];
                }
                col[i] = sum;
            }
            col
        })
        .collect();

    for (j, col) in columns.into_iter().enumerate() {
        for (i, val) in col.into_iter().enumerate() {
            c[(i, j)] = val;
        }
    }
}

/// Sequential QR algorithm for small matrices.
fn qr_algorithm_sequential<T: Field + Real>(
    mut diag: Vec<T>,
    mut off_diag: Vec<T>,
    v: &mut Mat<T>,
    n: usize,
) -> Result<Vec<T>, ParallelEvdError> {
    const MAX_ITERATIONS: usize = 100;

    if n <= 1 {
        return Ok(diag);
    }

    let eps = <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or(T::one());
    let mut m = n - 1;
    let mut iter = 0;

    while m > 0 && iter < MAX_ITERATIONS * n {
        iter += 1;

        let mut l = m;
        while l > 0 {
            let test = Scalar::abs(diag[l - 1]) + Scalar::abs(diag[l]);
            if Scalar::abs(off_diag[l - 1]) <= eps * test {
                off_diag[l - 1] = T::zero();
                break;
            }
            l -= 1;
        }

        if l == m {
            m -= 1;
            continue;
        }

        let d = (diag[m - 1] - diag[m]) / (T::one() + T::one());
        let e = off_diag[m - 1];
        let mu = diag[m] - e * e / (d + Real::signum(d) * Real::hypot(d, e));

        let mut x = diag[l] - mu;
        let mut z = off_diag[l];

        for k in l..m {
            let (c, s) = givens_rotation(x, z);

            if k > l {
                off_diag[k - 1] = Real::hypot(x, z);
            }

            let d1 = diag[k];
            let d2 = diag[k + 1];
            let e = off_diag[k];

            diag[k] = c * c * d1 + s * s * d2 - (c + c) * s * e;
            diag[k + 1] = s * s * d1 + c * c * d2 + (c + c) * s * e;
            off_diag[k] = c * s * (d1 - d2) + (c * c - s * s) * e;

            if k < m - 1 {
                x = off_diag[k];
                z = -s * off_diag[k + 1];
                off_diag[k + 1] = c * off_diag[k + 1];
            }

            for i in 0..n {
                let t1 = v[(i, k)];
                let t2 = v[(i, k + 1)];
                v[(i, k)] = c * t1 - s * t2;
                v[(i, k + 1)] = s * t1 + c * t2;
            }
        }
    }

    if iter >= MAX_ITERATIONS * n {
        return Err(ParallelEvdError::NotConverged);
    }

    sort_eigenvalues(&mut diag, v, n);
    Ok(diag)
}

/// Givens rotation coefficients.
fn givens_rotation<T: Field + Real>(a: T, b: T) -> (T, T) {
    if b == T::zero() {
        (T::one(), T::zero())
    } else if Scalar::abs(b) > Scalar::abs(a) {
        let t = -a / b;
        let s = T::one() / Real::sqrt(T::one() + t * t);
        (s * t, s)
    } else {
        let t = -b / a;
        let c = T::one() / Real::sqrt(T::one() + t * t);
        (c, c * t)
    }
}

/// Sort eigenvalues and rearrange eigenvectors.
fn sort_eigenvalues<T: Field + Real>(eigenvalues: &mut [T], v: &mut Mat<T>, n: usize) {
    for i in 1..n {
        let key = eigenvalues[i];
        let mut j = i;
        while j > 0 && eigenvalues[j - 1] > key {
            eigenvalues[j] = eigenvalues[j - 1];
            for row in 0..n {
                let tmp = v[(row, j)];
                v[(row, j)] = v[(row, j - 1)];
                v[(row, j - 1)] = tmp;
            }
            j -= 1;
        }
        eigenvalues[j] = key;
    }
}

// ============================================================================
// Parallel bisection for tridiagonal eigenvalues
// ============================================================================

/// Compute eigenvalues of a symmetric tridiagonal matrix using parallel bisection.
///
/// Each eigenvalue is computed independently in parallel.
///
/// # Arguments
///
/// * `diagonal` - Main diagonal elements (length n)
/// * `off_diagonal` - Off-diagonal elements (length n-1)
///
/// # Returns
///
/// Vector of eigenvalues sorted in ascending order.
pub fn parallel_bisection_eigenvalues<T: Field + Real + Send + Sync>(
    diagonal: &[T],
    off_diagonal: &[T],
) -> Result<Vec<T>, ParallelEvdError> {
    let n = diagonal.len();

    if n == 0 {
        return Err(ParallelEvdError::EmptyMatrix);
    }

    if off_diagonal.len() != n.saturating_sub(1) {
        return Err(ParallelEvdError::InvalidDimension);
    }

    if n == 1 {
        return Ok(vec![diagonal[0]]);
    }

    // Compute Gershgorin bounds
    let (glow, ghigh) = gershgorin_bounds(diagonal, off_diagonal);

    // Compute all eigenvalues in parallel
    let indices: Vec<usize> = (0..n).collect();

    let eigenvalues: Vec<T> = indices
        .par_iter()
        .map(|&target_index| {
            bisect_single_eigenvalue(diagonal, off_diagonal, target_index, glow, ghigh)
        })
        .collect();

    Ok(eigenvalues)
}

/// Bisect for a single eigenvalue.
fn bisect_single_eigenvalue<T: Field + Real>(
    diagonal: &[T],
    off_diagonal: &[T],
    target_index: usize,
    value_low: T,
    value_high: T,
) -> T {
    let eps = <T as Scalar>::epsilon();
    let two = T::one() + T::one();

    let mut lo = value_low;
    let mut hi = value_high;

    for _iter in 0..MAX_BISECTION_ITER {
        let tol = eps * (Scalar::abs(lo) + Scalar::abs(hi));
        if hi - lo <= tol {
            break;
        }

        let mid = (lo + hi) / two;
        let count = sturm_count(diagonal, off_diagonal, mid);

        if count <= target_index {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    (lo + hi) / two
}

/// Sturm sequence count.
fn sturm_count<T: Field + Real>(diagonal: &[T], off_diagonal: &[T], x: T) -> usize {
    let n = diagonal.len();
    if n == 0 {
        return 0;
    }

    let mut count = 0;
    let eps = <T as Scalar>::epsilon();

    let mut d = diagonal[0] - x;
    if d < T::zero() {
        count += 1;
    } else if d == T::zero() {
        d = -eps;
        count += 1;
    }

    for i in 1..n {
        let e_sq = off_diagonal[i - 1] * off_diagonal[i - 1];

        if Scalar::abs(d) < eps {
            d = if d >= T::zero() { eps } else { -eps };
        }

        d = (diagonal[i] - x) - e_sq / d;

        if d < T::zero() {
            count += 1;
        } else if d == T::zero() {
            d = -eps;
            count += 1;
        }
    }

    count
}

/// Gershgorin bounds for tridiagonal matrix.
fn gershgorin_bounds<T: Field + Real>(diagonal: &[T], off_diagonal: &[T]) -> (T, T) {
    let n = diagonal.len();

    if n == 0 {
        return (T::zero(), T::zero());
    }

    if n == 1 {
        return (diagonal[0], diagonal[0]);
    }

    let mut min = diagonal[0] - Scalar::abs(off_diagonal[0]);
    let mut max = diagonal[0] + Scalar::abs(off_diagonal[0]);

    for i in 1..(n - 1) {
        let radius = Scalar::abs(off_diagonal[i - 1]) + Scalar::abs(off_diagonal[i]);
        let low = diagonal[i] - radius;
        let high = diagonal[i] + radius;
        if low < min {
            min = low;
        }
        if high > max {
            max = high;
        }
    }

    let last_low = diagonal[n - 1] - Scalar::abs(off_diagonal[n - 2]);
    let last_high = diagonal[n - 1] + Scalar::abs(off_diagonal[n - 2]);
    if last_low < min {
        min = last_low;
    }
    if last_high > max {
        max = last_high;
    }

    let margin = (max - min) * T::from_f64(0.01).unwrap_or(T::zero());
    (min - margin, max + margin)
}

// ============================================================================
// Parallel inverse iteration for eigenvectors
// ============================================================================

/// Compute eigenvectors using parallel inverse iteration.
///
/// Each eigenvector is computed independently in parallel.
///
/// # Arguments
///
/// * `diagonal` - Main diagonal elements
/// * `off_diagonal` - Off-diagonal elements
/// * `eigenvalues` - Pre-computed eigenvalues
///
/// # Returns
///
/// Matrix of eigenvectors (columns correspond to eigenvalues).
pub fn parallel_inverse_iteration<T: Field + Real + bytemuck::Zeroable + Send + Sync>(
    diagonal: &[T],
    off_diagonal: &[T],
    eigenvalues: &[T],
) -> Result<Mat<T>, ParallelEvdError> {
    let n = diagonal.len();
    let k = eigenvalues.len();

    if k == 0 || n == 0 {
        return Ok(Mat::zeros(n, k));
    }

    let eps = <T as Scalar>::epsilon();
    let tol = eps * T::from_f64(100.0).unwrap_or(T::one());

    // Compute eigenvectors in parallel
    let indices: Vec<usize> = (0..k).collect();

    let columns: Vec<Vec<T>> = indices
        .par_iter()
        .map(|&j| {
            let lambda = eigenvalues[j];

            // Initialize vector
            let mut v = vec![T::one(); n];
            for i in 0..n {
                v[i] = T::from_f64(1.0 + 0.1 * (i as f64 % 7.0)).unwrap_or(T::one());
            }

            let shift = eps * (T::one() + Scalar::abs(lambda));
            let diag_shifted: Vec<T> = diagonal.iter().map(|&d| d - lambda - shift).collect();

            for _iter in 0..MAX_INVERSE_ITER {
                if let Ok(y) = solve_tridiagonal(&diag_shifted, off_diagonal, &v) {
                    let norm = vector_norm(&y);
                    if norm < tol {
                        break;
                    }
                    for i in 0..n {
                        v[i] = y[i] / norm;
                    }
                } else {
                    break;
                }
            }

            v
        })
        .collect();

    // Orthogonalize eigenvectors (sequential for numerical stability)
    let mut eigenvectors = Mat::zeros(n, k);

    for (j, col) in columns.into_iter().enumerate() {
        for (i, val) in col.into_iter().enumerate() {
            eigenvectors[(i, j)] = val;
        }
    }

    // Modified Gram-Schmidt orthogonalization
    for j in 0..k {
        for jj in 0..j {
            let mut dot = T::zero();
            for i in 0..n {
                dot = dot + eigenvectors[(i, jj)] * eigenvectors[(i, j)];
            }
            for i in 0..n {
                eigenvectors[(i, j)] = eigenvectors[(i, j)] - dot * eigenvectors[(i, jj)];
            }
        }

        // Re-normalize
        let mut norm_sq = T::zero();
        for i in 0..n {
            norm_sq = norm_sq + eigenvectors[(i, j)] * eigenvectors[(i, j)];
        }
        let norm = Real::sqrt(norm_sq);
        if norm > tol {
            for i in 0..n {
                eigenvectors[(i, j)] = eigenvectors[(i, j)] / norm;
            }
        }
    }

    Ok(eigenvectors)
}

/// Solve tridiagonal system using Thomas algorithm.
fn solve_tridiagonal<T: Field + Real>(
    diagonal: &[T],
    off_diagonal: &[T],
    rhs: &[T],
) -> Result<Vec<T>, ParallelEvdError> {
    let n = diagonal.len();

    if n == 0 {
        return Ok(Vec::new());
    }

    if n == 1 {
        if Scalar::abs(diagonal[0]) < <T as Scalar>::epsilon() {
            return Err(ParallelEvdError::NotConverged);
        }
        return Ok(vec![rhs[0] / diagonal[0]]);
    }

    let mut c = vec![T::zero(); n - 1];
    let mut d = vec![T::zero(); n];

    c[0] = off_diagonal[0] / diagonal[0];
    d[0] = rhs[0] / diagonal[0];

    for i in 1..(n - 1) {
        let denom = diagonal[i] - off_diagonal[i - 1] * c[i - 1];
        if Scalar::abs(denom) < <T as Scalar>::epsilon() {
            let sign = if denom >= T::zero() {
                T::one()
            } else {
                -T::one()
            };
            let denom = sign * <T as Scalar>::epsilon();
            c[i] = off_diagonal[i] / denom;
            d[i] = (rhs[i] - off_diagonal[i - 1] * d[i - 1]) / denom;
        } else {
            c[i] = off_diagonal[i] / denom;
            d[i] = (rhs[i] - off_diagonal[i - 1] * d[i - 1]) / denom;
        }
    }

    let denom = diagonal[n - 1] - off_diagonal[n - 2] * c[n - 2];
    if Scalar::abs(denom) < <T as Scalar>::epsilon() {
        let sign = if denom >= T::zero() {
            T::one()
        } else {
            -T::one()
        };
        d[n - 1] =
            (rhs[n - 1] - off_diagonal[n - 2] * d[n - 2]) / (sign * <T as Scalar>::epsilon());
    } else {
        d[n - 1] = (rhs[n - 1] - off_diagonal[n - 2] * d[n - 2]) / denom;
    }

    let mut x = vec![T::zero(); n];
    x[n - 1] = d[n - 1];

    for i in (0..(n - 1)).rev() {
        x[i] = d[i] - c[i] * x[i + 1];
    }

    Ok(x)
}

/// Compute vector 2-norm.
fn vector_norm<T: Field + Real>(v: &[T]) -> T {
    let mut sum = T::zero();
    for &x in v {
        sum = sum + x * x;
    }
    Real::sqrt(sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_parallel_evd_2x2() {
        let a = Mat::from_rows(&[&[2.0f64, 1.0], &[1.0, 2.0]]);

        let evd = ParallelSymmetricEvd::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        assert!(approx_eq(eigs[0], 1.0, 1e-10));
        assert!(approx_eq(eigs[1], 3.0, 1e-10));
    }

    #[test]
    fn test_parallel_evd_3x3() {
        let a = Mat::from_rows(&[&[4.0f64, 1.0, 1.0], &[1.0, 3.0, 2.0], &[1.0, 2.0, 3.0]]);

        let evd = ParallelSymmetricEvd::compute(a.as_ref()).unwrap();
        let reconstructed = evd.reconstruct();

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-8),
                    "mismatch at ({},{})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_parallel_evd_diagonal() {
        let a = Mat::from_rows(&[&[3.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 2.0]]);

        let evd = ParallelSymmetricEvd::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        assert!(approx_eq(eigs[0], 1.0, 1e-10));
        assert!(approx_eq(eigs[1], 2.0, 1e-10));
        assert!(approx_eq(eigs[2], 3.0, 1e-10));
    }

    #[test]
    fn test_parallel_evd_orthogonality() {
        let a = Mat::from_rows(&[&[4.0f64, 2.0, 1.0], &[2.0, 5.0, 3.0], &[1.0, 3.0, 6.0]]);

        let evd = ParallelSymmetricEvd::compute(a.as_ref()).unwrap();
        let v = evd.eigenvectors();

        // Verify V^T * V = I
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += v[(k, i)] * v[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(sum, expected, 1e-8),
                    "V^T*V[{},{}] = {}, expected {}",
                    i,
                    j,
                    sum,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_parallel_bisection_2x2() {
        let diag = vec![2.0, 2.0];
        let off_diag = vec![1.0];

        let eigs = parallel_bisection_eigenvalues(&diag, &off_diag).unwrap();

        assert_eq!(eigs.len(), 2);
        assert!(approx_eq(eigs[0], 1.0, 1e-10));
        assert!(approx_eq(eigs[1], 3.0, 1e-10));
    }

    #[test]
    fn test_parallel_bisection_larger() {
        let n = 10;
        let diag: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let off_diag: Vec<f64> = vec![0.0; n - 1];

        let eigs = parallel_bisection_eigenvalues(&diag, &off_diag).unwrap();

        assert_eq!(eigs.len(), n);
        for (i, &e) in eigs.iter().enumerate() {
            assert!(approx_eq(e, (i + 1) as f64, 1e-10));
        }
    }

    #[test]
    fn test_parallel_inverse_iteration() {
        let diag = vec![2.0, 2.0];
        let off_diag = vec![1.0];
        let eigenvalues = vec![1.0, 3.0];

        let vecs = parallel_inverse_iteration(&diag, &off_diag, &eigenvalues).unwrap();

        // Check orthogonality
        let mut dot = 0.0;
        for i in 0..2 {
            dot += vecs[(i, 0)] * vecs[(i, 1)];
        }
        assert!(approx_eq(dot, 0.0, 1e-8));

        // Check normalization
        for j in 0..2 {
            let mut norm = 0.0;
            for i in 0..2 {
                norm += vecs[(i, j)] * vecs[(i, j)];
            }
            assert!(approx_eq(norm, 1.0, 1e-8));
        }
    }

    #[test]
    fn test_parallel_evd_negative_eigenvalues() {
        let a = Mat::from_rows(&[&[-2.0f64, 1.0], &[1.0, -2.0]]);

        let evd = ParallelSymmetricEvd::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        assert!(approx_eq(eigs[0], -3.0, 1e-10));
        assert!(approx_eq(eigs[1], -1.0, 1e-10));
    }

    #[test]
    fn test_parallel_evd_f32() {
        let a = Mat::from_rows(&[&[2.0f32, 1.0], &[1.0, 2.0]]);

        let evd = ParallelSymmetricEvd::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        assert!((eigs[0] - 1.0).abs() < 1e-5);
        assert!((eigs[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_parallel_evd_larger_matrix() {
        let n = 10;
        let mut a: Mat<f64> = Mat::zeros(n, n);

        for i in 0..n {
            a[(i, i)] = (i + 1) as f64;
        }

        let evd = ParallelSymmetricEvd::compute(a.as_ref()).unwrap();
        let eigs = evd.eigenvalues();

        for i in 0..n {
            assert!(
                approx_eq(eigs[i], (i + 1) as f64, 1e-10),
                "eigenvalue {} = {}, expected {}",
                i,
                eigs[i],
                i + 1
            );
        }
    }

    #[test]
    fn test_parallel_evd_vs_sequential() {
        use crate::evd::SymmetricEvdDc;

        let a = Mat::from_rows(&[
            &[4.0f64, 2.0, 1.0, 0.5],
            &[2.0, 5.0, 3.0, 1.0],
            &[1.0, 3.0, 6.0, 2.0],
            &[0.5, 1.0, 2.0, 4.0],
        ]);

        let evd_seq = SymmetricEvdDc::compute(a.as_ref()).unwrap();
        let evd_par = ParallelSymmetricEvd::compute(a.as_ref()).unwrap();

        let eigs_seq = evd_seq.eigenvalues();
        let eigs_par = evd_par.eigenvalues();

        for i in 0..4 {
            assert!(
                approx_eq(eigs_seq[i], eigs_par[i], 1e-8),
                "eigenvalue {} mismatch: seq={}, par={}",
                i,
                eigs_seq[i],
                eigs_par[i]
            );
        }
    }
}
