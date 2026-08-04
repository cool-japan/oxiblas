//! Interval eigenvalue solver: computes all eigenvalues within a given
//! [low, high] range using Lanczos iteration with Sturm sequence counting.

use crate::csr::CsrMatrix;
use crate::ops::spmv;
use num_traits::FromPrimitive;
use oxiblas_core::scalar::{Field, Real, Scalar};

use super::super::error::EigenvalueError;

// Note: dot and norm from utils are available but this module uses inline implementations
// for better performance in tight loops
#[allow(unused_imports)]
use super::super::utils::{dot, norm};

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
/// * **Exact path (symmetric tridiagonal / diagonal `A`).** The Sturm sequence
///   is applied *directly* to `A`'s diagonal/off-diagonal bands. The Sturm
///   (inertia) count of a symmetric tridiagonal is exact, so the count and
///   locations of eigenvalues in [low, high] are exact to the bisection
///   tolerance — no Ritz approximation. A diagonal matrix is the special case.
///
/// * **General path (arbitrary symmetric `A`).** A Lanczos tridiagonal
///   `T = Q^T A Q` is built with full reorthogonalization; its Sturm count in
///   [low, high] yields *candidate* eigenvalues. Because eigenvalues of `T`
///   (Ritz values) only approximate `A`'s, each candidate is verified against
///   `A`: the Ritz vector `x = Q y` is formed and the residual
///   `||A x - theta x||` measured. By the Lanczos / Bauer-Fike bound a Ritz
///   value with residual `r` lies within `r` of a true eigenvalue of `A`, so
///   only *converged* candidates (`r <= tolerance`) are reported — approximate
///   Ritz values are never treated as exact interval members. For a full
///   Krylov subspace (`krylov_dimension >= n`) `T` is orthogonally similar to
///   `A`, residuals vanish, and the count is exact; otherwise `converged` is
///   `false`, signalling that `krylov_dimension` should be increased.
///
/// # Example
///
/// ```
/// use oxiblas_sparse::csr::CsrMatrix;
/// use oxiblas_sparse::linalg::eigenvalue::{IntervalEigen, IntervalEigenConfig};
///
/// // A diagonal matrix has its diagonal entries as eigenvalues: 1..=5.
/// let matrix =
///     CsrMatrix::new(5, 5, vec![0, 1, 2, 3, 4, 5], vec![0, 1, 2, 3, 4], vec![
///         1.0, 2.0, 3.0, 4.0, 5.0,
///     ])
///     .unwrap();
///
/// // Compute eigenvalues in (0.5, 3.5) -> {1.0, 2.0, 3.0}. The bounds are
/// // kept strictly away from any eigenvalue to avoid float boundary cases.
/// let config = IntervalEigenConfig::new(0.5, 3.5);
/// let solver = IntervalEigen::new(config);
/// let result = solver.compute(&matrix, None)?;
///
/// println!("Found {} eigenvalues in interval", result.count);
/// for (i, ev) in result.eigenvalues.iter().enumerate() {
///     println!("  lambda_{} = {}", i, ev);
/// }
/// assert_eq!(result.count, 3);
/// # Ok::<(), oxiblas_sparse::linalg::EigenvalueError>(())
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

        // Exact path: when `A` is (symmetric) tridiagonal — a diagonal matrix
        // being the special case — the Sturm sequence applies directly to
        // `A`'s bands, giving an exact interval count/location (no Ritz
        // approximation).
        if let Some((diag_a, off_a)) = Self::symmetric_tridiagonal_bands(a) {
            return self.compute_from_bands(a, &diag_a, &off_a);
        }

        // General path: Sturm count of the Lanczos tridiagonal `T = Q^T A Q`
        // gives *candidate* eigenvalues; because Ritz values only approximate
        // `A`'s eigenvalues, each candidate is verified against `A` below.
        let krylov_dim = self.config.krylov_dimension.min(n);
        let (alpha, beta, q_basis, iterations) =
            self.lanczos_iteration(a, initial_vector, krylov_dim)?;

        let candidate_count = self
            .sturm_count(&alpha, &beta, self.config.high.clone())
            .saturating_sub(self.sturm_count(&alpha, &beta, self.config.low.clone()));

        if candidate_count == 0 {
            return Ok(IntervalEigenResult {
                eigenvalues: vec![],
                eigenvectors: if self.config.compute_eigenvectors {
                    Some(vec![])
                } else {
                    None
                },
                iterations,
                residual_norms: vec![],
                converged: true,
                count: 0,
            });
        }

        let candidates = self.find_eigenvalues_in_interval(&alpha, &beta, candidate_count)?;

        // Verify every candidate against `A` (Ritz vector + residual), always —
        // even when eigenvectors are not requested — so the reported count
        // reflects genuine eigenvalues of `A`, not raw Ritz values. A residual
        // below tolerance bounds the distance to a true eigenvalue of `A`
        // (Lanczos/Bauer-Fike); unconverged Ritz values are not reported.
        let (ritz_vectors, residuals) =
            self.compute_ritz_vectors(a, &alpha, &beta, &q_basis, &candidates)?;
        let verify_tol = self.config.tolerance.clone();
        let mut eigenvalues = Vec::with_capacity(candidates.len());
        let mut eigenvectors = Vec::with_capacity(candidates.len());
        let mut residual_norms = Vec::with_capacity(candidates.len());
        for ((lambda, ritz_vec), residual) in candidates.iter().zip(ritz_vectors).zip(residuals) {
            let in_interval = *lambda >= self.config.low && *lambda <= self.config.high;
            if in_interval && residual < verify_tol {
                eigenvalues.push(lambda.clone());
                eigenvectors.push(ritz_vec);
                residual_norms.push(residual);
            }
        }

        // Exact when every candidate converged; else the count is a lower bound.
        let converged = eigenvalues.len() == candidate_count;

        Ok(IntervalEigenResult {
            count: eigenvalues.len(),
            eigenvalues,
            eigenvectors: if self.config.compute_eigenvectors {
                Some(eigenvectors)
            } else {
                None
            },
            iterations,
            residual_norms,
            converged,
        })
    }

    /// Return the main diagonal and first super-diagonal of `a` when `a` is
    /// (structurally) symmetric tridiagonal (every stored nonzero on the main,
    /// sub- or super-diagonal; a diagonal matrix is the special case). Returns
    /// `None` when any nonzero lies strictly outside the band, deferring to the
    /// general Lanczos path. For symmetric `A` the super-diagonal equals the
    /// sub-diagonal and the Sturm sequence uses only the squared off-diagonal.
    fn symmetric_tridiagonal_bands(a: &CsrMatrix<T>) -> Option<(Vec<T>, Vec<T>)> {
        let n = a.nrows();
        let row_ptrs = a.row_ptrs();
        let col_indices = a.col_indices();
        let values = a.values();

        let mut diag = vec![T::zero(); n];
        let mut off = vec![T::zero(); n.saturating_sub(1)];

        for i in 0..n {
            for idx in row_ptrs[i]..row_ptrs[i + 1] {
                let j = col_indices[idx];
                let v = values[idx].clone();
                let dist = j.abs_diff(i);
                if dist > 1 {
                    // A genuine nonzero outside the band disqualifies the exact
                    // path; an explicitly stored zero is harmless.
                    if Scalar::abs(v) > T::zero() {
                        return None;
                    }
                    continue;
                }
                if i == j {
                    diag[i] = v;
                } else if j == i + 1 {
                    off[i] = v;
                }
                // Sub-diagonal (j == i - 1) is redundant for symmetric `A`.
            }
        }

        Some((diag, off))
    }

    /// Exact interval solve when `A` is symmetric tridiagonal: the Sturm
    /// sequence and bisection run directly on `A`'s bands (exact count and
    /// locations, no Ritz approximation). Eigenvectors, when requested, come
    /// from inverse iteration on the same bands (`A` is the tridiagonal), with
    /// the residual `||A x - lambda x||` reported for each.
    fn compute_from_bands(
        &self,
        a: &CsrMatrix<T>,
        diag_a: &[T],
        off_a: &[T],
    ) -> Result<IntervalEigenResult<T>, EigenvalueError> {
        let n = diag_a.len();

        let count = self
            .sturm_count(diag_a, off_a, self.config.high.clone())
            .saturating_sub(self.sturm_count(diag_a, off_a, self.config.low.clone()));

        if count == 0 {
            return Ok(IntervalEigenResult {
                eigenvalues: vec![],
                eigenvectors: if self.config.compute_eigenvectors {
                    Some(vec![])
                } else {
                    None
                },
                iterations: 0,
                residual_norms: vec![],
                converged: true,
                count: 0,
            });
        }

        let eigenvalues = self.find_eigenvalues_in_interval(diag_a, off_a, count)?;

        let (eigenvectors, residual_norms) = if self.config.compute_eigenvectors
            && !eigenvalues.is_empty()
        {
            let eps = T::from_f64(1e-15).unwrap_or_else(T::zero);
            let mut evecs = Vec::with_capacity(eigenvalues.len());
            let mut resids = Vec::with_capacity(eigenvalues.len());
            for lambda in &eigenvalues {
                // Eigenvector of `A` via inverse iteration on its bands.
                let mut x = self.inverse_iteration_tridiagonal(diag_a, off_a, lambda.clone())?;

                // Normalize.
                let mut norm_sq = T::zero();
                for xi in &x {
                    norm_sq = norm_sq + xi.clone() * xi.clone();
                }
                let norm = Real::sqrt(norm_sq);
                if norm > eps {
                    for xi in &mut x {
                        *xi = xi.clone() / norm.clone();
                    }
                }

                // Residual ||A x - lambda x||.
                let mut ax = vec![T::zero(); n];
                spmv(T::one(), a, &x, T::zero(), &mut ax);
                let mut res_sq = T::zero();
                for i in 0..n {
                    let diff = ax[i].clone() - lambda.clone() * x[i].clone();
                    res_sq = res_sq + diff.clone() * diff;
                }
                resids.push(Real::sqrt(res_sq));
                evecs.push(x);
            }
            (Some(evecs), resids)
        } else {
            (None, vec![T::zero(); eigenvalues.len()])
        };

        Ok(IntervalEigenResult {
            count: eigenvalues.len(),
            eigenvalues,
            eigenvectors,
            iterations: 0,
            residual_norms,
            // The eigenvalue count and locations are exact for tridiagonal `A`.
            converged: true,
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
/// For a symmetric tridiagonal (or diagonal) matrix the count is exact: the
/// Sturm sequence is applied directly to `A`'s bands. For a general symmetric
/// matrix the count is the number of *converged* Ritz eigenpairs of a Lanczos
/// tridiagonal that fall inside the interval and are verified against `A`
/// (see [`IntervalEigen`]); increase `krylov_dim` towards `n` for a guaranteed
/// count.
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
