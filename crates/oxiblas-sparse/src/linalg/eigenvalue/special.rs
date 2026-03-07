//! Special eigenvalue methods: interval-based and polynomial filtering.
//!
//! This module provides specialized eigenvalue solvers for computing eigenvalues
//! within a specified interval or using polynomial filtering techniques:
//!
//! - [`IntervalEigen`]: Computes all eigenvalues within a given interval [low, high]
//!   using Lanczos iteration with Sturm sequence counting.
//!
//! - [`PolynomialFilteredLanczos`]: Uses Chebyshev polynomial filtering to compute
//!   interior eigenvalues without expensive matrix factorization.
//!
//! # Example
//!
//! ```ignore
//! use oxiblas_sparse::linalg::eigenvalue::special::{
//!     IntervalEigen, IntervalEigenConfig,
//!     PolynomialFilteredLanczos, PolynomialFilterConfig,
//! };
//!
//! // Compute eigenvalues in [1.0, 3.0] using interval method
//! let config = IntervalEigenConfig::new(1.0, 3.0);
//! let solver = IntervalEigen::new(config);
//! let result = solver.compute(&matrix, None)?;
//!
//! // Compute eigenvalues using polynomial filtering
//! let config = PolynomialFilterConfig::new(0.5, 2.0);
//! let solver = PolynomialFilteredLanczos::new(config);
//! let result = solver.compute(&matrix, None)?;
//! ```

use crate::csr::CsrMatrix;
use crate::ops::spmv;
use num_traits::FromPrimitive;
use oxiblas_core::scalar::{Field, Real, Scalar};

use super::error::EigenvalueError;

// Note: dot and norm from utils are available but this module uses inline implementations
// for better performance in tight loops
#[allow(unused_imports)]
use super::utils::{dot, norm};

// ============================================================================
// Interval Eigenvalue Solver
// ============================================================================

/// Configuration for interval eigenvalue computation.
///
/// Specifies the interval [low, high] and algorithm parameters for
/// finding all eigenvalues within the interval.
#[derive(Debug, Clone)]
pub struct IntervalEigenConfig<T> {
    /// Lower bound of the interval.
    pub low: T,
    /// Upper bound of the interval.
    pub high: T,
    /// Maximum Lanczos iterations.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: T,
    /// Whether to compute eigenvectors.
    pub compute_eigenvectors: bool,
    /// Krylov subspace dimension (larger = more accurate but more memory).
    pub krylov_dimension: usize,
    /// Use full reorthogonalization.
    pub full_reorthogonalization: bool,
}

impl<T: Real + FromPrimitive> IntervalEigenConfig<T> {
    /// Create a new configuration for the interval [low, high].
    pub fn new(low: T, high: T) -> Self {
        Self {
            low,
            high,
            max_iterations: 500,
            tolerance: T::from_f64(1e-10).unwrap_or_else(T::zero),
            compute_eigenvectors: true,
            krylov_dimension: 50,
            full_reorthogonalization: true,
        }
    }
}

impl Default for IntervalEigenConfig<f64> {
    fn default() -> Self {
        Self {
            low: 0.0,
            high: 1.0,
            max_iterations: 500,
            tolerance: 1e-10,
            compute_eigenvectors: true,
            krylov_dimension: 50,
            full_reorthogonalization: true,
        }
    }
}

impl Default for IntervalEigenConfig<f32> {
    fn default() -> Self {
        Self {
            low: 0.0,
            high: 1.0,
            max_iterations: 500,
            tolerance: 1e-6,
            compute_eigenvectors: true,
            krylov_dimension: 50,
            full_reorthogonalization: true,
        }
    }
}

/// Result of interval eigenvalue computation.
#[derive(Debug, Clone)]
pub struct IntervalEigenResult<T> {
    /// Eigenvalues in the interval, sorted in ascending order.
    pub eigenvalues: Vec<T>,
    /// Eigenvectors (if computed), stored as columns.
    pub eigenvectors: Option<Vec<Vec<T>>>,
    /// Number of Lanczos iterations performed.
    pub iterations: usize,
    /// Residual norms for each eigenpair.
    pub residual_norms: Vec<T>,
    /// Whether computation converged.
    pub converged: bool,
    /// Count of eigenvalues found in the interval.
    pub count: usize,
}

/// Interval eigenvalue solver for symmetric sparse matrices.
///
/// Computes all eigenvalues (and optionally eigenvectors) that lie within
/// a specified interval [low, high] using Lanczos iteration combined with
/// Sturm sequence counting.
///
/// # Algorithm
///
/// 1. Build Krylov subspace using Lanczos iteration: A = Q T Q^T
/// 2. Use Sturm sequence to count eigenvalues of T in [low, high]
/// 3. Apply bisection to locate each eigenvalue
/// 4. Compute Ritz vectors if eigenvectors are requested
///
/// This is particularly efficient when the interval contains a small
/// fraction of the total eigenvalues.
///
/// # Example
///
/// ```ignore
/// use oxiblas_sparse::linalg::eigenvalue::special::{IntervalEigen, IntervalEigenConfig};
///
/// // Compute eigenvalues in [1.0, 3.0]
/// let config = IntervalEigenConfig::new(1.0, 3.0);
/// let solver = IntervalEigen::new(config);
/// let result = solver.compute(&matrix, None)?;
///
/// println!("Found {} eigenvalues in interval", result.count);
/// for (i, ev) in result.eigenvalues.iter().enumerate() {
///     println!("  lambda_{} = {}", i, ev);
/// }
/// ```
pub struct IntervalEigen<T> {
    config: IntervalEigenConfig<T>,
}

impl<T: Scalar<Real = T> + Clone + Field + Real + FromPrimitive> IntervalEigen<T> {
    /// Create a new interval eigenvalue solver.
    pub fn new(config: IntervalEigenConfig<T>) -> Self {
        Self { config }
    }

    /// Compute eigenvalues in the interval [low, high].
    ///
    /// # Arguments
    ///
    /// * `a` - Symmetric sparse matrix in CSR format
    /// * `initial_vector` - Optional starting vector for Lanczos (if None, uses random)
    pub fn compute(
        &self,
        a: &CsrMatrix<T>,
        initial_vector: Option<&[T]>,
    ) -> Result<IntervalEigenResult<T>, EigenvalueError> {
        let n = a.nrows();
        if n != a.ncols() {
            return Err(EigenvalueError::NotSquare {
                nrows: n,
                ncols: a.ncols(),
            });
        }

        if n == 0 {
            return Ok(IntervalEigenResult {
                eigenvalues: vec![],
                eigenvectors: None,
                iterations: 0,
                residual_norms: vec![],
                converged: true,
                count: 0,
            });
        }

        if self.config.low > self.config.high {
            return Err(EigenvalueError::ComputationError(
                "Interval low bound must be <= high bound".to_string(),
            ));
        }

        let krylov_dim = self.config.krylov_dimension.min(n);

        // Step 1: Run Lanczos to build tridiagonal T
        let (alpha, beta, q_basis, iterations) =
            self.lanczos_iteration(a, initial_vector, krylov_dim)?;

        // Step 2: Count eigenvalues of T in [low, high] using Sturm sequence
        let count_in_interval = self
            .sturm_count(&alpha, &beta, self.config.high.clone())
            .saturating_sub(self.sturm_count(&alpha, &beta, self.config.low.clone()));

        if count_in_interval == 0 {
            return Ok(IntervalEigenResult {
                eigenvalues: vec![],
                eigenvectors: None,
                iterations,
                residual_norms: vec![],
                converged: true,
                count: 0,
            });
        }

        // Step 3: Find eigenvalues in interval using bisection
        let eigenvalues = self.find_eigenvalues_in_interval(&alpha, &beta, count_in_interval)?;

        // Step 4: Compute eigenvectors if requested
        let (eigenvectors, residual_norms) =
            if self.config.compute_eigenvectors && !eigenvalues.is_empty() {
                let (evecs, residuals) =
                    self.compute_ritz_vectors(a, &alpha, &beta, &q_basis, &eigenvalues)?;
                (Some(evecs), residuals)
            } else {
                (None, vec![T::zero(); eigenvalues.len()])
            };

        let converged = residual_norms.iter().all(|r| *r < self.config.tolerance);

        Ok(IntervalEigenResult {
            count: eigenvalues.len(),
            eigenvalues,
            eigenvectors,
            iterations,
            residual_norms,
            converged,
        })
    }

    /// Lanczos iteration to build tridiagonal matrix.
    fn lanczos_iteration(
        &self,
        a: &CsrMatrix<T>,
        initial_vector: Option<&[T]>,
        krylov_dim: usize,
    ) -> Result<(Vec<T>, Vec<T>, Vec<Vec<T>>, usize), EigenvalueError> {
        let n = a.nrows();
        let mut alpha = Vec::with_capacity(krylov_dim);
        let mut beta = Vec::with_capacity(krylov_dim);
        let mut q_basis: Vec<Vec<T>> = Vec::with_capacity(krylov_dim);

        // Initial vector
        let mut q: Vec<T> = if let Some(v0) = initial_vector {
            if v0.len() != n {
                return Err(EigenvalueError::DimensionMismatch {
                    expected: n,
                    actual: v0.len(),
                });
            }
            v0.to_vec()
        } else {
            // Use deterministic initial vector
            (0..n)
                .map(|i| T::from_f64((i + 1) as f64 / (n + 1) as f64).unwrap_or_else(T::zero))
                .collect()
        };

        // Normalize initial vector
        let mut norm_sq = T::zero();
        for qi in &q {
            norm_sq = norm_sq + qi.clone() * qi.clone();
        }
        let norm = Real::sqrt(norm_sq);
        if norm < T::from_f64(1e-15).unwrap_or_else(T::zero) {
            return Err(EigenvalueError::Breakdown {
                iteration: 0,
                description: "Initial vector is too small".to_string(),
            });
        }
        for qi in &mut q {
            *qi = qi.clone() / norm.clone();
        }

        let mut q_prev = vec![T::zero(); n];
        let mut beta_prev = T::zero();

        for _j in 0..krylov_dim {
            q_basis.push(q.clone());

            // w = A * q
            let mut w = vec![T::zero(); n];
            spmv(T::one(), a, &q, T::zero(), &mut w);

            // alpha[j] = q^T * w
            let mut alpha_j = T::zero();
            for i in 0..n {
                alpha_j = alpha_j + q[i].clone() * w[i].clone();
            }
            alpha.push(alpha_j.clone());

            // w = w - alpha[j] * q - beta[j-1] * q_prev
            for i in 0..n {
                w[i] = w[i].clone()
                    - alpha_j.clone() * q[i].clone()
                    - beta_prev.clone() * q_prev[i].clone();
            }

            // Full reorthogonalization
            if self.config.full_reorthogonalization {
                for qk in &q_basis {
                    let mut dot = T::zero();
                    for i in 0..n {
                        dot = dot + w[i].clone() * qk[i].clone();
                    }
                    for i in 0..n {
                        w[i] = w[i].clone() - dot.clone() * qk[i].clone();
                    }
                }
            }

            // beta[j] = ||w||
            let mut norm_sq = T::zero();
            for wi in &w {
                norm_sq = norm_sq + wi.clone() * wi.clone();
            }
            let beta_j = Real::sqrt(norm_sq);

            if beta_j < self.config.tolerance {
                // Invariant subspace found
                break;
            }

            beta.push(beta_j.clone());

            // q_prev = q, q = w / beta[j]
            q_prev = q;
            q = w.iter().map(|wi| wi.clone() / beta_j.clone()).collect();
            beta_prev = beta_j;
        }

        let krylov_size = alpha.len();
        Ok((alpha, beta, q_basis, krylov_size))
    }

    /// Count eigenvalues <= x using Sturm sequence.
    fn sturm_count(&self, alpha: &[T], beta: &[T], x: T) -> usize {
        let n = alpha.len();
        if n == 0 {
            return 0;
        }

        let eps = T::from_f64(1e-30).unwrap_or_else(T::zero);
        let mut count = 0;
        let mut d = alpha[0].clone() - x.clone();

        if d <= T::zero() {
            count += 1;
        }

        for i in 1..n {
            let b_sq = if i - 1 < beta.len() {
                beta[i - 1].clone() * beta[i - 1].clone()
            } else {
                T::zero()
            };

            // d_i = alpha[i] - x - beta[i-1]^2 / d_{i-1}
            let d_abs = if d >= T::zero() {
                d.clone()
            } else {
                T::zero() - d.clone()
            };
            d = if d_abs < eps {
                alpha[i].clone()
                    - x.clone()
                    - b_sq
                        / (if d >= T::zero() {
                            eps.clone()
                        } else {
                            T::zero() - eps.clone()
                        })
            } else {
                alpha[i].clone() - x.clone() - b_sq / d.clone()
            };

            if d <= T::zero() {
                count += 1;
            }
        }

        count
    }

    /// Find eigenvalues in interval using bisection.
    fn find_eigenvalues_in_interval(
        &self,
        alpha: &[T],
        beta: &[T],
        count: usize,
    ) -> Result<Vec<T>, EigenvalueError> {
        if count == 0 {
            return Ok(vec![]);
        }

        let mut eigenvalues = Vec::with_capacity(count);
        let count_below_low = self.sturm_count(alpha, beta, self.config.low.clone());
        let tol = self.config.tolerance.clone();
        let max_iter = 1000;

        for k in 0..count {
            let target_index = count_below_low + k;
            let mut lo = self.config.low.clone();
            let mut hi = self.config.high.clone();

            for _ in 0..max_iter {
                let mid = (lo.clone() + hi.clone()) / T::from_f64(2.0).unwrap_or_else(T::zero);
                let c = self.sturm_count(alpha, beta, mid.clone());

                if c <= target_index {
                    lo = mid;
                } else {
                    hi = mid;
                }

                if hi.clone() - lo.clone() < tol {
                    break;
                }
            }

            eigenvalues.push((lo + hi) / T::from_f64(2.0).unwrap_or_else(T::zero));
        }

        // Sort eigenvalues in ascending order
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        Ok(eigenvalues)
    }

    /// Compute Ritz vectors for eigenvalues.
    fn compute_ritz_vectors(
        &self,
        a: &CsrMatrix<T>,
        alpha: &[T],
        beta: &[T],
        q_basis: &[Vec<T>],
        eigenvalues: &[T],
    ) -> Result<(Vec<Vec<T>>, Vec<T>), EigenvalueError> {
        let n = a.nrows();
        let m = alpha.len();
        let k = eigenvalues.len();

        if k == 0 || m == 0 || q_basis.is_empty() {
            return Ok((vec![], vec![]));
        }

        let mut eigenvectors = Vec::with_capacity(k);
        let mut residual_norms = Vec::with_capacity(k);

        // For each eigenvalue, compute eigenvector of T using inverse iteration,
        // then transform to Ritz vector
        for &lambda in eigenvalues {
            // Inverse iteration on T for eigenvector y
            let y = self.inverse_iteration_tridiagonal(alpha, beta, lambda.clone())?;

            // Ritz vector: x = Q * y
            let mut x = vec![T::zero(); n];
            for i in 0..n {
                for j in 0..m.min(q_basis.len()) {
                    x[i] = x[i].clone() + q_basis[j][i].clone() * y[j].clone();
                }
            }

            // Normalize
            let mut norm_sq = T::zero();
            for xi in &x {
                norm_sq = norm_sq + xi.clone() * xi.clone();
            }
            let norm = Real::sqrt(norm_sq);
            if norm > T::from_f64(1e-15).unwrap_or_else(T::zero) {
                for xi in &mut x {
                    *xi = xi.clone() / norm.clone();
                }
            }

            // Compute residual: ||Ax - lambda*x||
            let mut ax = vec![T::zero(); n];
            spmv(T::one(), a, &x, T::zero(), &mut ax);
            let mut res_sq = T::zero();
            for i in 0..n {
                let diff = ax[i].clone() - lambda.clone() * x[i].clone();
                res_sq = res_sq + diff.clone() * diff;
            }
            let residual = Real::sqrt(res_sq);

            eigenvectors.push(x);
            residual_norms.push(residual);
        }

        Ok((eigenvectors, residual_norms))
    }

    /// Inverse iteration on tridiagonal matrix to get eigenvector.
    fn inverse_iteration_tridiagonal(
        &self,
        alpha: &[T],
        beta: &[T],
        lambda: T,
    ) -> Result<Vec<T>, EigenvalueError> {
        let n = alpha.len();
        if n == 0 {
            return Ok(vec![]);
        }

        let max_iter = 100;
        let tol = self.config.tolerance.clone();
        let eps = T::from_f64(1e-14).unwrap_or_else(T::zero);

        // Initial vector
        let mut y: Vec<T> = (0..n)
            .map(|i| T::from_f64((i + 1) as f64).unwrap_or_else(T::zero))
            .collect();

        // Normalize
        let mut norm_sq = T::zero();
        for yi in &y {
            norm_sq = norm_sq + yi.clone() * yi.clone();
        }
        let norm = Real::sqrt(norm_sq);
        for yi in &mut y {
            *yi = yi.clone() / norm.clone();
        }

        for _ in 0..max_iter {
            // Solve (T - lambda*I) * y_new = y using Thomas algorithm
            let y_new = self.solve_shifted_tridiagonal(alpha, beta, &y, lambda.clone())?;

            // Normalize
            let mut norm_sq = T::zero();
            for yi in &y_new {
                norm_sq = norm_sq + yi.clone() * yi.clone();
            }
            let norm = Real::sqrt(norm_sq);
            if norm < eps {
                break;
            }

            // Check convergence
            let mut diff_sq = T::zero();
            for i in 0..n {
                let new_val = y_new[i].clone() / norm.clone();
                let d = new_val.clone() - y[i].clone();
                diff_sq = diff_sq + d.clone() * d;
            }

            for i in 0..n {
                y[i] = y_new[i].clone() / norm.clone();
            }

            if Real::sqrt(diff_sq) < tol {
                break;
            }
        }

        Ok(y)
    }

    /// Solve (T - lambda*I)x = b for tridiagonal T.
    fn solve_shifted_tridiagonal(
        &self,
        alpha: &[T],
        beta: &[T],
        b: &[T],
        lambda: T,
    ) -> Result<Vec<T>, EigenvalueError> {
        let n = alpha.len();
        if n == 0 {
            return Ok(vec![]);
        }

        let eps = T::from_f64(1e-14).unwrap_or_else(T::zero);

        // Forward elimination
        let mut c_prime = vec![T::zero(); n];
        let mut d_prime = vec![T::zero(); n];

        // First row
        let diag = alpha[0].clone() - lambda.clone();
        let diag_safe = if Scalar::abs(diag.clone()) < eps {
            if diag >= T::zero() {
                eps.clone()
            } else {
                T::zero() - eps.clone()
            }
        } else {
            diag
        };

        c_prime[0] = if !beta.is_empty() {
            beta[0].clone() / diag_safe.clone()
        } else {
            T::zero()
        };
        d_prime[0] = b[0].clone() / diag_safe;

        // Forward sweep
        for i in 1..n {
            let a_i = if i - 1 < beta.len() {
                beta[i - 1].clone()
            } else {
                T::zero()
            };
            let diag = alpha[i].clone() - lambda.clone();
            let denom = diag - a_i.clone() * c_prime[i - 1].clone();

            let denom_safe = if Scalar::abs(denom.clone()) < eps {
                if denom >= T::zero() {
                    eps.clone()
                } else {
                    T::zero() - eps.clone()
                }
            } else {
                denom
            };

            c_prime[i] = if i < beta.len() {
                beta[i].clone() / denom_safe.clone()
            } else {
                T::zero()
            };
            d_prime[i] = (b[i].clone() - a_i * d_prime[i - 1].clone()) / denom_safe;
        }

        // Back substitution
        let mut x = vec![T::zero(); n];
        x[n - 1] = d_prime[n - 1].clone();
        for i in (0..n - 1).rev() {
            x[i] = d_prime[i].clone() - c_prime[i].clone() * x[i + 1].clone();
        }

        Ok(x)
    }
}

// ============================================================================
// Public convenience functions for interval eigenvalues
// ============================================================================

/// Convenience function to compute eigenvalues in an interval.
///
/// This function creates a default `IntervalEigenConfig` with the given bounds
/// and computes eigenvalues within that interval.
///
/// # Arguments
///
/// * `a` - Sparse symmetric matrix in CSR format
/// * `low` - Lower bound of the interval
/// * `high` - Upper bound of the interval
///
/// # Returns
///
/// Result containing eigenvalues found in the interval along with optional eigenvectors.
pub fn eigenvalues_in_interval<T: Scalar<Real = T> + Clone + Field + Real + FromPrimitive>(
    a: &CsrMatrix<T>,
    low: T,
    high: T,
) -> Result<IntervalEigenResult<T>, EigenvalueError> {
    let config = IntervalEigenConfig::new(low, high);
    let solver = IntervalEigen::new(config);
    solver.compute(a, None)
}

/// Count eigenvalues of a sparse symmetric matrix in an interval.
///
/// Uses Lanczos to approximate eigenvalues and Sturm sequence to count.
///
/// # Arguments
///
/// * `a` - Sparse symmetric matrix in CSR format
/// * `low` - Lower bound of the interval
/// * `high` - Upper bound of the interval
/// * `krylov_dim` - Dimension of Krylov subspace for approximation
///
/// # Returns
///
/// The number of eigenvalues in the interval [low, high].
pub fn count_eigenvalues_in_interval<T: Scalar<Real = T> + Clone + Field + Real + FromPrimitive>(
    a: &CsrMatrix<T>,
    low: T,
    high: T,
    krylov_dim: usize,
) -> Result<usize, EigenvalueError> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(EigenvalueError::NotSquare {
            nrows: n,
            ncols: a.ncols(),
        });
    }

    if n == 0 {
        return Ok(0);
    }

    let config = IntervalEigenConfig {
        low: low.clone(),
        high: high.clone(),
        krylov_dimension: krylov_dim.min(n),
        compute_eigenvectors: false,
        ..IntervalEigenConfig::new(low.clone(), high.clone())
    };

    let solver = IntervalEigen::new(config);
    let result = solver.compute(a, None)?;
    Ok(result.count)
}

// ============================================================================
// Polynomial Filtered Lanczos
// ============================================================================

/// Configuration for polynomial filtered Lanczos iteration.
///
/// Polynomial filtering uses Chebyshev polynomials to enhance eigenvalues
/// in a target interval while suppressing unwanted eigenvalues. This is
/// useful for computing interior eigenvalues without expensive matrix
/// factorization (as needed in shift-invert).
///
/// # Example
///
/// ```ignore
/// use oxiblas_sparse::linalg::eigenvalue::special::{PolynomialFilteredLanczos, PolynomialFilterConfig};
/// use oxiblas_sparse::CsrMatrix;
///
/// // Create a sparse matrix
/// let values = vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0];
/// let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
/// let row_ptrs = vec![0, 2, 5, 7];
/// let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();
///
/// // Configure to find eigenvalues in [0.5, 1.5]
/// let config = PolynomialFilterConfig {
///     num_eigenvalues: 1,
///     target_low: 0.5,
///     target_high: 1.5,
///     polynomial_degree: 20,
///     krylov_dimension: 30,
///     ..Default::default()
/// };
///
/// let solver = PolynomialFilteredLanczos::new(config);
/// let result = solver.compute(&a, None).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct PolynomialFilterConfig<T> {
    /// Number of eigenvalues to compute.
    pub num_eigenvalues: usize,
    /// Lower bound of target interval.
    pub target_low: T,
    /// Upper bound of target interval.
    pub target_high: T,
    /// Spectral lower bound (estimate of smallest eigenvalue).
    pub spectral_low: Option<T>,
    /// Spectral upper bound (estimate of largest eigenvalue).
    pub spectral_high: Option<T>,
    /// Degree of Chebyshev polynomial filter.
    pub polynomial_degree: usize,
    /// Krylov subspace dimension.
    pub krylov_dimension: usize,
    /// Maximum number of outer iterations.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: T,
    /// Whether to compute eigenvectors.
    pub compute_eigenvectors: bool,
    /// Full reorthogonalization.
    pub full_reorthogonalization: bool,
}

impl<T: Clone + Real + FromPrimitive> Default for PolynomialFilterConfig<T> {
    fn default() -> Self {
        Self {
            num_eigenvalues: 6,
            target_low: T::zero(),
            target_high: T::one(),
            spectral_low: None,
            spectral_high: None,
            polynomial_degree: 20,
            krylov_dimension: 50,
            max_iterations: 100,
            tolerance: T::from_f64(1e-8).unwrap_or_else(T::zero),
            compute_eigenvectors: true,
            full_reorthogonalization: true,
        }
    }
}

impl<T: Clone + Real + FromPrimitive> PolynomialFilterConfig<T> {
    /// Create a new configuration for the given target interval.
    pub fn new(target_low: T, target_high: T) -> Self {
        Self {
            target_low,
            target_high,
            ..Default::default()
        }
    }
}

/// Result of polynomial filtered Lanczos iteration.
#[derive(Debug, Clone)]
pub struct PolynomialFilteredResult<T> {
    /// Computed eigenvalues (sorted by magnitude within target interval).
    pub eigenvalues: Vec<T>,
    /// Computed eigenvectors (if requested).
    pub eigenvectors: Option<Vec<Vec<T>>>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Residual norms for each eigenvalue.
    pub residual_norms: Vec<T>,
    /// Whether convergence was achieved.
    pub converged: bool,
}

/// Polynomial filtered Lanczos iteration for interior eigenvalues.
///
/// Uses Chebyshev polynomial filtering to compute eigenvalues in a
/// specified interval without matrix factorization. Particularly useful
/// for large sparse matrices where shift-invert would be too expensive.
///
/// The filter is designed to:
/// 1. Amplify eigenvalues in the target interval [a, b]
/// 2. Dampen eigenvalues outside the interval
///
/// This is achieved using Chebyshev polynomials that map:
/// - Target interval [a, b] -> [-1, 1] (mild transformation)
/// - Unwanted spectrum -> large values (strong damping)
pub struct PolynomialFilteredLanczos<T> {
    config: PolynomialFilterConfig<T>,
}

impl<T: Scalar<Real = T> + Clone + Field + Real + FromPrimitive> PolynomialFilteredLanczos<T> {
    /// Create a new polynomial filtered Lanczos solver.
    pub fn new(config: PolynomialFilterConfig<T>) -> Self {
        Self { config }
    }

    /// Compute eigenvalues in the target interval.
    ///
    /// # Arguments
    /// * `a` - Sparse symmetric matrix
    /// * `initial_vectors` - Optional initial vectors for the Krylov basis
    ///
    /// # Returns
    /// * `Ok(PolynomialFilteredResult)` on success
    /// * `Err(EigenvalueError)` on failure
    pub fn compute(
        &self,
        a: &CsrMatrix<T>,
        initial_vectors: Option<&[Vec<T>]>,
    ) -> Result<PolynomialFilteredResult<T>, EigenvalueError> {
        let n = a.nrows();
        if n != a.ncols() {
            return Err(EigenvalueError::NotSquare {
                nrows: n,
                ncols: a.ncols(),
            });
        }

        if n == 0 {
            return Ok(PolynomialFilteredResult {
                eigenvalues: vec![],
                eigenvectors: if self.config.compute_eigenvectors {
                    Some(vec![])
                } else {
                    None
                },
                iterations: 0,
                residual_norms: vec![],
                converged: true,
            });
        }

        // Estimate spectral bounds if not provided
        let (lambda_min, lambda_max) = self.estimate_spectral_bounds(a)?;

        // Build Chebyshev filter coefficients
        let filter_coeffs = self.compute_chebyshev_filter(
            &lambda_min,
            &lambda_max,
            &self.config.target_low,
            &self.config.target_high,
        );

        // Initialize starting vectors
        let num_start_vectors = self.config.num_eigenvalues.max(1);
        let mut v_basis: Vec<Vec<T>> = if let Some(init) = initial_vectors {
            init.iter().take(num_start_vectors).cloned().collect()
        } else {
            self.random_orthonormal_vectors(n, num_start_vectors)
        };

        let krylov_dim = self.config.krylov_dimension.min(n);
        let mut converged_eigenvalues: Vec<T> = Vec::new();
        let mut converged_eigenvectors: Vec<Vec<T>> = Vec::new();
        let mut converged_residuals: Vec<T> = Vec::new();

        for iter in 0..self.config.max_iterations {
            // Apply polynomial filter to current vectors
            let filtered_vectors: Vec<Vec<T>> = v_basis
                .iter()
                .map(|v| self.apply_filter(a, v, &filter_coeffs, &lambda_min, &lambda_max))
                .collect();

            // Orthonormalize filtered vectors
            let ortho_vectors = self.orthonormalize(&filtered_vectors);
            if ortho_vectors.is_empty() {
                break;
            }

            // Build filtered Krylov subspace
            let (alpha, beta, q_basis) = self.filtered_lanczos(
                a,
                &ortho_vectors[0],
                krylov_dim,
                &filter_coeffs,
                &lambda_min,
                &lambda_max,
            )?;

            if alpha.is_empty() {
                break;
            }

            // Solve tridiagonal eigenvalue problem
            // Note: These eigenvalues are for p(A), not A
            // We need to sort by the filtered eigenvalues (largest values from filter are enhanced)
            let (eig_vals, eig_vecs) = self.solve_tridiagonal_evd(&alpha, &beta);

            // Sort by largest filtered eigenvalue (these correspond to eigenvalues in target interval)
            // The polynomial filter enhances eigenvalues in target interval to large values
            let mut sorted_indices: Vec<(usize, T)> = eig_vals
                .iter()
                .enumerate()
                .map(|(i, ev)| (i, ev.clone()))
                .collect();
            sorted_indices.sort_by(|(_, a_val), (_, b_val)| {
                // Sort by largest eigenvalue first (filter enhances target eigenvalues)
                b_val
                    .partial_cmp(a_val)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Compute Ritz vectors and actual eigenvalues using Rayleigh quotient with original matrix A
            // Note: The eigenvalues from tridiagonal EVD are for p(A), not A
            // We need to compute Rayleigh quotient with A to get actual eigenvalues
            for (idx, _) in sorted_indices.iter().take(self.config.num_eigenvalues * 2)
            // Check more candidates since some may be outside interval
            {
                // Compute Ritz vector from Krylov basis
                let ritz_vec = self.compute_ritz_vector(&q_basis, &eig_vecs[*idx]);
                if ritz_vec.is_empty() {
                    continue;
                }

                // Compute actual eigenvalue using Rayleigh quotient: lambda = (v^T * A * v) / (v^T * v)
                let mut ax = vec![T::zero(); ritz_vec.len()];
                spmv(T::one(), a, &ritz_vec, T::zero(), &mut ax);

                let vtav: T = ritz_vec
                    .iter()
                    .zip(ax.iter())
                    .map(|(vi, axi)| vi.clone() * axi.clone())
                    .fold(T::zero(), |acc, x| acc + x);
                let vtv: T = ritz_vec
                    .iter()
                    .map(|vi| vi.clone() * vi.clone())
                    .fold(T::zero(), |acc, x| acc + x);

                if vtv <= T::from_f64(1e-14).unwrap_or_else(T::zero) {
                    continue;
                }

                let eigenvalue = vtav / vtv.clone();

                // Check if actual eigenvalue is in target interval
                if eigenvalue.clone() < self.config.target_low.clone()
                    || eigenvalue.clone() > self.config.target_high.clone()
                {
                    continue;
                }

                // Compute residual: ||A*x - lambda*x||
                let residual: T = Real::sqrt(
                    ax.iter()
                        .zip(ritz_vec.iter())
                        .map(|(axi, xi)| {
                            let diff = axi.clone() - eigenvalue.clone() * xi.clone();
                            diff.clone() * diff
                        })
                        .fold(T::zero(), |acc, x| acc + x),
                );

                if residual <= self.config.tolerance {
                    converged_eigenvalues.push(eigenvalue.clone());
                    converged_residuals.push(residual);
                    if self.config.compute_eigenvectors {
                        converged_eigenvectors.push(ritz_vec);
                    }
                }
            }

            // Check if we have enough converged eigenvalues
            if converged_eigenvalues.len() >= self.config.num_eigenvalues {
                return Ok(PolynomialFilteredResult {
                    eigenvalues: converged_eigenvalues,
                    eigenvectors: if self.config.compute_eigenvectors {
                        Some(converged_eigenvectors)
                    } else {
                        None
                    },
                    iterations: iter + 1,
                    residual_norms: converged_residuals,
                    converged: true,
                });
            }

            // Update starting vectors with best Ritz vectors from filter
            v_basis.clear();
            for (idx, _) in sorted_indices.iter().take(num_start_vectors) {
                let ritz_vec = self.compute_ritz_vector(&q_basis, &eig_vecs[*idx]);
                if !ritz_vec.is_empty() {
                    v_basis.push(ritz_vec);
                }
            }

            if v_basis.is_empty() {
                // Add random vectors if no candidates
                v_basis = self.random_orthonormal_vectors(n, num_start_vectors);
            }
        }

        // Return what we have even if not fully converged
        Ok(PolynomialFilteredResult {
            eigenvalues: converged_eigenvalues.clone(),
            eigenvectors: if self.config.compute_eigenvectors {
                Some(converged_eigenvectors)
            } else {
                None
            },
            iterations: self.config.max_iterations,
            residual_norms: converged_residuals,
            converged: converged_eigenvalues.len() >= self.config.num_eigenvalues,
        })
    }

    /// Estimate spectral bounds using a few Lanczos iterations.
    fn estimate_spectral_bounds(&self, a: &CsrMatrix<T>) -> Result<(T, T), EigenvalueError> {
        if let (Some(low), Some(high)) = (
            self.config.spectral_low.clone(),
            self.config.spectral_high.clone(),
        ) {
            return Ok((low, high));
        }

        let n = a.nrows();
        let k = 20.min(n);

        // Run a few Lanczos iterations to estimate bounds
        let v0: Vec<T> = (0..n)
            .map(|i| T::from_f64(((i * 7 + 13) % 101) as f64 / 100.0 - 0.5).unwrap_or_else(T::zero))
            .collect();
        let norm: T = Real::sqrt(
            v0.iter()
                .map(|x| x.clone() * x.clone())
                .fold(T::zero(), |a, b| a + b),
        );
        let v0: Vec<T> = v0.iter().map(|x| x.clone() / norm.clone()).collect();

        let mut alpha: Vec<T> = Vec::with_capacity(k);
        let mut beta: Vec<T> = Vec::with_capacity(k);

        let mut v = v0;
        let mut v_prev = vec![T::zero(); n];

        for j in 0..k {
            // w = A * v
            let mut w = vec![T::zero(); n];
            spmv(T::one(), a, &v, T::zero(), &mut w);

            // alpha[j] = v' * w
            let alpha_j: T = v
                .iter()
                .zip(w.iter())
                .map(|(vi, wi)| vi.clone() * wi.clone())
                .fold(T::zero(), |acc, x| acc + x);
            alpha.push(alpha_j.clone());

            // w = w - alpha[j]*v - beta[j-1]*v_prev
            for (wi, vi) in w.iter_mut().zip(v.iter()) {
                *wi = wi.clone() - alpha_j.clone() * vi.clone();
            }

            if j > 0 {
                let beta_prev = beta[j - 1].clone();
                for (wi, vpi) in w.iter_mut().zip(v_prev.iter()) {
                    *wi = wi.clone() - beta_prev.clone() * vpi.clone();
                }
            }

            // beta[j] = ||w||
            let beta_j: T = Real::sqrt(
                w.iter()
                    .map(|x| x.clone() * x.clone())
                    .fold(T::zero(), |acc, x| acc + x),
            );

            if beta_j <= T::from_f64(1e-14).unwrap_or_else(T::zero) {
                break;
            }

            beta.push(beta_j.clone());

            // Update vectors
            v_prev = v;
            v = w.iter().map(|wi| wi.clone() / beta_j.clone()).collect();
        }

        // Solve tridiagonal eigenvalue problem
        let (eig_vals, _) = self.solve_tridiagonal_evd(&alpha, &beta);

        if eig_vals.is_empty() {
            return Ok((T::zero(), T::one()));
        }

        let mut min_val = eig_vals[0].clone();
        let mut max_val = eig_vals[0].clone();
        for ev in eig_vals.iter() {
            if ev.clone() < min_val {
                min_val = ev.clone();
            }
            if ev.clone() > max_val {
                max_val = ev.clone();
            }
        }

        // Add some margin
        let margin = (max_val.clone() - min_val.clone()) * T::from_f64(0.1).unwrap_or_else(T::zero);
        Ok((min_val - margin.clone(), max_val + margin))
    }

    /// Compute Chebyshev filter coefficients.
    fn compute_chebyshev_filter(
        &self,
        lambda_min: &T,
        lambda_max: &T,
        target_low: &T,
        target_high: &T,
    ) -> Vec<T> {
        let degree = self.config.polynomial_degree;
        let mut coeffs = vec![T::zero(); degree + 1];

        // Map target interval to [-1, 1] and unwanted to outside
        // For Chebyshev of first kind: T_n(x) = cos(n * acos(x))
        // We use Jackson damping for smoother filter

        let two = T::from_f64(2.0).unwrap_or_else(T::zero);
        let pi_f64 = std::f64::consts::PI;

        // Compute Jackson damping coefficients using f64 for trig functions
        let n_f64 = (degree + 2) as f64;
        for k in 0..=degree {
            let k_f64 = k as f64;
            // g_k = ((n - k) * cos(pi*k/n) + sin(pi*k/n) * cot(pi/n)) / n
            let ratio = k_f64 * pi_f64 / n_f64;
            let cos_val = ratio.cos();
            let sin_val = ratio.sin();
            let cot_pi_n = (pi_f64 / n_f64).cos() / (pi_f64 / n_f64).sin();

            let g_k = ((n_f64 - k_f64) * cos_val + sin_val * cot_pi_n) / n_f64;
            coeffs[k] = T::from_f64(g_k).unwrap_or_else(T::zero);
        }

        // Scale coefficients for the spectral range
        let scale = two.clone() / (lambda_max.clone() - lambda_min.clone());
        let center = (lambda_max.clone() + lambda_min.clone()) / two.clone();

        // Adjust for target interval
        let target_center = (target_high.clone() + target_low.clone()) / two.clone();
        let _target_half_width = (target_high.clone() - target_low.clone()) / two;

        // Modify coefficients to enhance target interval - use f64 for trig
        let center_f64 = center.to_f64().unwrap_or(0.0);
        let target_center_f64 = target_center.to_f64().unwrap_or(0.0);
        let scale_f64 = scale.to_f64().unwrap_or(1.0);

        for (k, coeff) in coeffs.iter_mut().enumerate() {
            let k_f64 = k as f64;
            // Enhance contribution near target center
            let arg = k_f64 * pi_f64 * ((target_center_f64 - center_f64) * scale_f64);
            let enhancement = arg.cos().abs() + 0.5;
            *coeff = coeff.clone() * T::from_f64(enhancement).unwrap_or_else(T::zero);
        }

        coeffs
    }

    /// Apply Chebyshev polynomial filter to a vector.
    fn apply_filter(
        &self,
        a: &CsrMatrix<T>,
        v: &[T],
        coeffs: &[T],
        lambda_min: &T,
        lambda_max: &T,
    ) -> Vec<T> {
        let n = v.len();
        if coeffs.is_empty() {
            return v.to_vec();
        }

        // Scale and shift parameters
        let two = T::from_f64(2.0).unwrap_or_else(T::zero);
        let e = (lambda_max.clone() - lambda_min.clone()) / two.clone();
        let c = (lambda_max.clone() + lambda_min.clone()) / two.clone();

        // T_0(x) = I, T_1(x) = x, T_{n+1}(x) = 2*x*T_n(x) - T_{n-1}(x)
        // Compute: p(A)*v using three-term recurrence

        // t_0 = v (T_0 = 1)
        let mut t_prev: Vec<T> = v.to_vec();

        // result = c_0 * T_0 * v
        let mut result: Vec<T> = t_prev
            .iter()
            .map(|x| x.clone() * coeffs[0].clone())
            .collect();

        if coeffs.len() == 1 {
            return result;
        }

        // t_1 = (A - c*I) * v / e  (maps eigenvalues to [-1, 1])
        let mut av = vec![T::zero(); n];
        spmv(T::one(), a, v, T::zero(), &mut av);
        let mut t_curr: Vec<T> = av
            .iter()
            .zip(v.iter())
            .map(|(avi, vi)| (avi.clone() - c.clone() * vi.clone()) / e.clone())
            .collect();

        // result += c_1 * T_1 * v
        for (ri, ti) in result.iter_mut().zip(t_curr.iter()) {
            *ri = ri.clone() + coeffs[1].clone() * ti.clone();
        }

        // Recurrence for higher degrees
        for k in 2..coeffs.len() {
            // t_next = 2 * ((A - c*I)/e) * t_curr - t_prev
            let mut at_curr = vec![T::zero(); n];
            spmv(T::one(), a, &t_curr, T::zero(), &mut at_curr);
            let t_next: Vec<T> = at_curr
                .iter()
                .zip(t_curr.iter())
                .zip(t_prev.iter())
                .map(|((ati, ti), tpi)| {
                    two.clone() * (ati.clone() - c.clone() * ti.clone()) / e.clone() - tpi.clone()
                })
                .collect();

            // result += c_k * T_k * v
            for (ri, tni) in result.iter_mut().zip(t_next.iter()) {
                *ri = ri.clone() + coeffs[k].clone() * tni.clone();
            }

            t_prev = t_curr;
            t_curr = t_next;
        }

        result
    }

    /// Run filtered Lanczos iteration.
    fn filtered_lanczos(
        &self,
        a: &CsrMatrix<T>,
        v0: &[T],
        max_iter: usize,
        filter_coeffs: &[T],
        lambda_min: &T,
        lambda_max: &T,
    ) -> Result<(Vec<T>, Vec<T>, Vec<Vec<T>>), EigenvalueError> {
        let n = v0.len();

        let mut alpha = Vec::with_capacity(max_iter);
        let mut beta = Vec::with_capacity(max_iter);
        let mut q_basis: Vec<Vec<T>> = Vec::with_capacity(max_iter + 1);

        // Normalize initial vector
        let norm: T = Real::sqrt(
            v0.iter()
                .map(|x| x.clone() * x.clone())
                .fold(T::zero(), |acc, x| acc + x),
        );

        if norm <= T::from_f64(1e-14).unwrap_or_else(T::zero) {
            return Ok((vec![], vec![], vec![]));
        }

        let mut q: Vec<T> = v0.iter().map(|x| x.clone() / norm.clone()).collect();
        q_basis.push(q.clone());

        let mut q_prev = vec![T::zero(); n];
        let mut beta_prev = T::zero();

        for j in 0..max_iter {
            // Apply filtered matrix: w = p(A) * q
            let w = self.apply_filter(a, &q, filter_coeffs, lambda_min, lambda_max);

            // alpha[j] = q' * w
            let alpha_j: T = q
                .iter()
                .zip(w.iter())
                .map(|(qi, wi)| qi.clone() * wi.clone())
                .fold(T::zero(), |acc, x| acc + x);
            alpha.push(alpha_j.clone());

            // w = w - alpha[j]*q - beta[j-1]*q_prev
            let mut w: Vec<T> = w
                .iter()
                .zip(q.iter())
                .map(|(wi, qi)| wi.clone() - alpha_j.clone() * qi.clone())
                .collect();

            if j > 0 {
                for (wi, qpi) in w.iter_mut().zip(q_prev.iter()) {
                    *wi = wi.clone() - beta_prev.clone() * qpi.clone();
                }
            }

            // Full reorthogonalization
            if self.config.full_reorthogonalization {
                for qk in &q_basis {
                    let dot: T = w
                        .iter()
                        .zip(qk.iter())
                        .map(|(wi, qki)| wi.clone() * qki.clone())
                        .fold(T::zero(), |acc, x| acc + x);
                    for (wi, qki) in w.iter_mut().zip(qk.iter()) {
                        *wi = wi.clone() - dot.clone() * qki.clone();
                    }
                }
            }

            // beta[j] = ||w||
            let beta_j: T = Real::sqrt(
                w.iter()
                    .map(|x| x.clone() * x.clone())
                    .fold(T::zero(), |acc, x| acc + x),
            );

            if beta_j <= T::from_f64(1e-12).unwrap_or_else(T::zero) {
                break;
            }

            beta.push(beta_j.clone());

            // Update vectors
            q_prev = q;
            q = w.iter().map(|wi| wi.clone() / beta_j.clone()).collect();
            q_basis.push(q.clone());
            beta_prev = beta_j;
        }

        Ok((alpha, beta, q_basis))
    }

    /// Solve tridiagonal eigenvalue problem.
    fn solve_tridiagonal_evd(&self, alpha: &[T], beta: &[T]) -> (Vec<T>, Vec<Vec<T>>) {
        let n = alpha.len();
        if n == 0 {
            return (vec![], vec![]);
        }

        if n == 1 {
            return (vec![alpha[0].clone()], vec![vec![T::one()]]);
        }

        // Use bisection + inverse iteration
        // First find bounds using Gershgorin
        let mut lower = alpha[0].clone() - beta.first().copied().unwrap_or(T::zero());
        let mut upper = alpha[0].clone() + beta.first().copied().unwrap_or(T::zero());

        for i in 0..n {
            let left = if i > 0 {
                beta[i - 1].clone()
            } else {
                T::zero()
            };
            let right = if i < beta.len() {
                beta[i].clone()
            } else {
                T::zero()
            };
            let row_sum = left.clone() + right.clone();
            let low_bound = alpha[i].clone() - row_sum.clone();
            let high_bound = alpha[i].clone() + row_sum;

            if low_bound < lower {
                lower = low_bound;
            }
            if high_bound > upper {
                upper = high_bound;
            }
        }

        // Find all eigenvalues using bisection
        let mut eigenvalues = Vec::with_capacity(n);
        let margin = (upper.clone() - lower.clone()) * T::from_f64(0.01).unwrap_or_else(T::zero);
        let a = lower.clone() - margin.clone();
        let b = upper.clone() + margin;

        for k in 0..n {
            let eigenvalue = self.bisection_find_eigenvalue(alpha, beta, &a, &b, k);
            eigenvalues.push(eigenvalue);
        }

        // Compute eigenvectors using inverse iteration
        let eigenvectors: Vec<Vec<T>> = eigenvalues
            .iter()
            .map(|ev| self.inverse_iteration_tridiag(alpha, beta, ev))
            .collect();

        (eigenvalues, eigenvectors)
    }

    /// Find k-th eigenvalue using bisection.
    fn bisection_find_eigenvalue(&self, alpha: &[T], beta: &[T], a: &T, b: &T, k: usize) -> T {
        let mut low = a.clone();
        let mut high = b.clone();

        let tol = T::from_f64(1e-14).unwrap_or_else(T::zero);

        for _ in 0..100 {
            let mid = (low.clone() + high.clone()) / T::from_f64(2.0).unwrap_or_else(T::zero);

            if Scalar::abs(high.clone() - low.clone()) < tol {
                return mid;
            }

            let count = self.sturm_count_at(alpha, beta, &mid);

            if count <= k {
                low = mid;
            } else {
                high = mid;
            }
        }

        (low + high) / T::from_f64(2.0).unwrap_or_else(T::zero)
    }

    /// Count eigenvalues <= x using Sturm sequence.
    fn sturm_count_at(&self, alpha: &[T], beta: &[T], x: &T) -> usize {
        let n = alpha.len();
        if n == 0 {
            return 0;
        }

        let eps = T::from_f64(1e-30).unwrap_or_else(T::zero);
        let mut count = 0;
        let mut d = alpha[0].clone() - x.clone();

        if d <= T::zero() {
            count += 1;
        }

        for i in 1..n {
            let beta_sq = if i <= beta.len() {
                beta[i - 1].clone() * beta[i - 1].clone()
            } else {
                T::zero()
            };

            if Scalar::abs(d.clone()) < eps {
                d = eps.clone();
            }

            d = alpha[i].clone() - x.clone() - beta_sq / d;

            if d <= T::zero() {
                count += 1;
            }
        }

        count
    }

    /// Inverse iteration to compute eigenvector.
    fn inverse_iteration_tridiag(&self, alpha: &[T], beta: &[T], eigenvalue: &T) -> Vec<T> {
        let n = alpha.len();
        if n == 0 {
            return vec![];
        }

        // Start with random vector
        let mut v: Vec<T> = (0..n)
            .map(|i| T::from_f64(((i * 13 + 7) % 97) as f64 / 97.0 - 0.5).unwrap_or_else(T::zero))
            .collect();

        let shift = T::from_f64(1e-10).unwrap_or_else(T::zero);

        for _ in 0..5 {
            // Solve (T - lambda*I) * w = v
            let w = self.solve_shifted_tridiag(alpha, beta, eigenvalue, &shift, &v);

            // Normalize
            let norm: T = Real::sqrt(
                w.iter()
                    .map(|x| x.clone() * x.clone())
                    .fold(T::zero(), |acc, x| acc + x),
            );

            if norm > T::from_f64(1e-14).unwrap_or_else(T::zero) {
                v = w.iter().map(|x| x.clone() / norm.clone()).collect();
            } else {
                break;
            }
        }

        v
    }

    /// Solve shifted tridiagonal system using Thomas algorithm.
    fn solve_shifted_tridiag(
        &self,
        alpha: &[T],
        beta: &[T],
        eigenvalue: &T,
        shift: &T,
        b: &[T],
    ) -> Vec<T> {
        let n = alpha.len();
        if n == 0 {
            return vec![];
        }

        // (T - (lambda + shift)*I) * x = b
        let lambda_shift = eigenvalue.clone() + shift.clone();

        // Modified Thomas algorithm
        let mut c_prime = vec![T::zero(); n];
        let mut d_prime = vec![T::zero(); n];

        // Forward sweep
        let diag_0 = alpha[0].clone() - lambda_shift.clone();
        let eps = T::from_f64(1e-14).unwrap_or_else(T::zero);
        let diag_0_safe = if Scalar::abs(diag_0.clone()) < eps {
            eps.clone()
        } else {
            diag_0
        };

        if !beta.is_empty() {
            c_prime[0] = beta[0].clone() / diag_0_safe.clone();
        }
        d_prime[0] = b[0].clone() / diag_0_safe;

        for i in 1..n {
            let sub_diag = if i > 0 && i - 1 < beta.len() {
                beta[i - 1].clone()
            } else {
                T::zero()
            };
            let super_diag = if i < beta.len() {
                beta[i].clone()
            } else {
                T::zero()
            };

            let diag_i = alpha[i].clone() - lambda_shift.clone();
            let denom = diag_i - sub_diag.clone() * c_prime[i - 1].clone();
            let denom_safe = if Scalar::abs(denom.clone()) < eps {
                eps.clone()
            } else {
                denom
            };

            if i < n - 1 {
                c_prime[i] = super_diag / denom_safe.clone();
            }
            d_prime[i] = (b[i].clone() - sub_diag * d_prime[i - 1].clone()) / denom_safe;
        }

        // Back substitution
        let mut x = vec![T::zero(); n];
        x[n - 1] = d_prime[n - 1].clone();

        for i in (0..n - 1).rev() {
            x[i] = d_prime[i].clone() - c_prime[i].clone() * x[i + 1].clone();
        }

        x
    }

    /// Generate random orthonormal vectors.
    fn random_orthonormal_vectors(&self, n: usize, k: usize) -> Vec<Vec<T>> {
        let mut vectors: Vec<Vec<T>> = Vec::with_capacity(k);

        for j in 0..k {
            // Generate pseudo-random vector
            let mut v: Vec<T> = (0..n)
                .map(|i| {
                    T::from_f64(
                        (((i + j * n) * 1103515245 + 12345) % 2147483648) as f64 / 2147483648.0
                            - 0.5,
                    )
                    .unwrap_or_else(T::zero)
                })
                .collect();

            // Orthogonalize against previous vectors
            for prev in &vectors {
                let dot: T = v
                    .iter()
                    .zip(prev.iter())
                    .map(|(vi, pi)| vi.clone() * pi.clone())
                    .fold(T::zero(), |acc, x| acc + x);
                for (vi, pi) in v.iter_mut().zip(prev.iter()) {
                    *vi = vi.clone() - dot.clone() * pi.clone();
                }
            }

            // Normalize
            let norm: T = Real::sqrt(
                v.iter()
                    .map(|x| x.clone() * x.clone())
                    .fold(T::zero(), |acc, x| acc + x),
            );

            if norm > T::from_f64(1e-14).unwrap_or_else(T::zero) {
                v = v.iter().map(|x| x.clone() / norm.clone()).collect();
                vectors.push(v);
            }
        }

        vectors
    }

    /// Orthonormalize a set of vectors using modified Gram-Schmidt.
    fn orthonormalize(&self, vectors: &[Vec<T>]) -> Vec<Vec<T>> {
        let mut result: Vec<Vec<T>> = Vec::with_capacity(vectors.len());

        for v in vectors {
            let mut u = v.clone();

            // Orthogonalize against previous vectors
            for q in &result {
                let dot: T = u
                    .iter()
                    .zip(q.iter())
                    .map(|(ui, qi)| ui.clone() * qi.clone())
                    .fold(T::zero(), |acc, x| acc + x);
                for (ui, qi) in u.iter_mut().zip(q.iter()) {
                    *ui = ui.clone() - dot.clone() * qi.clone();
                }
            }

            // Normalize
            let norm: T = Real::sqrt(
                u.iter()
                    .map(|x| x.clone() * x.clone())
                    .fold(T::zero(), |acc, x| acc + x),
            );

            if norm > T::from_f64(1e-14).unwrap_or_else(T::zero) {
                u = u.iter().map(|x| x.clone() / norm.clone()).collect();
                result.push(u);
            }
        }

        result
    }

    /// Compute Ritz vector from Krylov basis.
    fn compute_ritz_vector(&self, q_basis: &[Vec<T>], y: &[T]) -> Vec<T> {
        if q_basis.is_empty() || y.is_empty() {
            return vec![];
        }

        let n = q_basis[0].len();
        let mut result = vec![T::zero(); n];

        for (i, yi) in y.iter().enumerate() {
            if i < q_basis.len() {
                for (rj, qij) in result.iter_mut().zip(q_basis[i].iter()) {
                    *rj = rj.clone() + yi.clone() * qij.clone();
                }
            }
        }

        // Normalize
        let norm: T = Real::sqrt(
            result
                .iter()
                .map(|x| x.clone() * x.clone())
                .fold(T::zero(), |acc, x| acc + x),
        );

        if norm > T::from_f64(1e-14).unwrap_or_else(T::zero) {
            result = result.iter().map(|x| x.clone() / norm.clone()).collect();
        }

        result
    }
}

// ============================================================================
// Public convenience function for polynomial filtered eigenvalues
// ============================================================================

/// Convenience function for polynomial filtered eigenvalue computation.
///
/// Computes eigenvalues of a sparse symmetric matrix within a target interval
/// using Chebyshev polynomial filtering.
///
/// # Arguments
///
/// * `a` - Sparse symmetric matrix in CSR format
/// * `target_low` - Lower bound of target interval
/// * `target_high` - Upper bound of target interval
/// * `num_eigenvalues` - Number of eigenvalues to compute
///
/// # Returns
///
/// Result containing computed eigenvalues and optional eigenvectors.
pub fn polynomial_filtered_eigenvalues<
    T: Scalar<Real = T> + Clone + Field + Real + FromPrimitive,
>(
    a: &CsrMatrix<T>,
    target_low: T,
    target_high: T,
    num_eigenvalues: usize,
) -> Result<PolynomialFilteredResult<T>, EigenvalueError> {
    let config = PolynomialFilterConfig {
        num_eigenvalues,
        target_low,
        target_high,
        ..Default::default()
    };
    let solver = PolynomialFilteredLanczos::new(config);
    solver.compute(a, None)
}
