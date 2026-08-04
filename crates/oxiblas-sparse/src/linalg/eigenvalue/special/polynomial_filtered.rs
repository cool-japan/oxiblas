//! Polynomial-filtered Lanczos: uses Chebyshev polynomial filtering to
//! compute interior eigenvalues without expensive matrix factorization.

use crate::csr::CsrMatrix;
use crate::ops::spmv;
use num_traits::FromPrimitive;
use oxiblas_core::scalar::{Field, Real, Scalar};

use super::super::error::EigenvalueError;

// Note: dot and norm from utils are available but this module uses inline implementations
// for better performance in tight loops
#[allow(unused_imports)]
use super::super::utils::{dot, norm};
// Dense symmetric eigensolver for the small Rayleigh-Ritz projection used by
// the polynomial-filtered solver.
use super::super::utils::dense_symmetric_jacobi_evd;
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
/// ```
/// use oxiblas_sparse::CsrMatrix;
/// use oxiblas_sparse::linalg::eigenvalue::{PolynomialFilterConfig, PolynomialFilteredLanczos};
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
/// assert!(!result.eigenvalues.is_empty());
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
/// The filter is a degree-`polynomial_degree` polynomial `p(A)` that
/// approximates the indicator (band-pass window) of the target interval
/// `[a, b]` over the spectral range `[lambda_min, lambda_max]`:
///
/// 1. `p(lambda) ~ 1` for eigenvalues in the target interval `[a, b]`.
/// 2. `p(lambda) ~ 0` for eigenvalues outside it.
///
/// It is built as `p(lambda) = sum_k g_k * mu_k * T_k(x(lambda))` where the
/// affine map `x(lambda) = (2*lambda - lambda_max - lambda_min) /
/// (lambda_max - lambda_min)` sends the spectrum onto `[-1, 1]`, `T_k` is the
/// Chebyshev polynomial of the first kind (applied to `A` through its
/// three-term recurrence — repeated sparse matrix-vector products), the `mu_k`
/// are the Chebyshev coefficients of the window indicator, and the `g_k` are
/// Jackson damping factors that suppress the Gibbs oscillations of the
/// truncated expansion. Running Lanczos on `p(A)` amplifies the components of
/// eigenvectors whose eigenvalues lie in the target interval relative to the
/// rest of the spectrum, so interior eigenvalues emerge without any matrix
/// factorization.
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

        // Eigenvalues separated by less than this are treated as the same
        // eigenvalue when de-duplicating discoveries across outer iterations.
        let dedup_tol = {
            let width = Scalar::abs(lambda_max.clone() - lambda_min.clone());
            let rel = width * Real::sqrt(<T as Scalar>::epsilon());
            let abs = self.config.tolerance.clone() * T::from_f64(8.0).unwrap_or_else(T::zero);
            if rel > abs { rel } else { abs }
        };

        for iter in 0..self.config.max_iterations {
            // Apply the polynomial filter to the current block of vectors; the
            // filter amplifies the components lying in the target interval.
            let filtered_vectors: Vec<Vec<T>> = v_basis
                .iter()
                .map(|v| self.apply_filter(a, v, &filter_coeffs, &lambda_min, &lambda_max))
                .collect();

            // Orthonormalize the filtered block.
            let ortho_vectors = self.orthonormalize(&filtered_vectors);
            if ortho_vectors.is_empty() {
                break;
            }

            // Block-enriched subspace: the union of the filtered Krylov
            // subspaces grown from each independent filtered start vector. A
            // single start vector captures only a one-dimensional slice of an
            // eigenspace whose eigenvalues the filter maps to (nearly) identical
            // values; several independent starts let the subspace span such
            // (near-)degenerate eigenspaces so their eigenvalues can be resolved.
            let mut subspace: Vec<Vec<T>> = Vec::new();
            for start in &ortho_vectors {
                let (_alpha, _beta, q_basis) = self.filtered_lanczos(
                    a,
                    start,
                    krylov_dim,
                    &filter_coeffs,
                    &lambda_min,
                    &lambda_max,
                )?;
                subspace.extend(q_basis);
            }

            // Orthonormalize the union (removing overlaps and dependencies).
            let q = self.orthonormalize(&subspace);
            if q.is_empty() {
                break;
            }

            // Rayleigh-Ritz projection onto the ORIGINAL matrix A: form the
            // small dense symmetric matrix H = Q^T A Q and diagonalize it. Its
            // eigenpairs approximate eigenpairs of A directly (the filter only
            // built a subspace rich in the target eigenvectors), recovering the
            // true eigenvalues — including any the filter mapped to equal values.
            let m = q.len();
            let mut aq: Vec<Vec<T>> = Vec::with_capacity(m);
            for qi in &q {
                let mut av = vec![T::zero(); n];
                spmv(T::one(), a, qi, T::zero(), &mut av);
                aq.push(av);
            }
            let mut h = vec![vec![T::zero(); m]; m];
            for i in 0..m {
                for j in i..m {
                    let mut hij = T::zero();
                    for l in 0..n {
                        hij = hij + q[i][l].clone() * aq[j][l].clone();
                    }
                    h[i][j] = hij.clone();
                    h[j][i] = hij;
                }
            }
            let (ritz_vals, ritz_vecs) = dense_symmetric_jacobi_evd(&h);

            // Examine Ritz values in order of proximity to the target interval.
            let target_center = (self.config.target_low.clone() + self.config.target_high.clone())
                / T::from_f64(2.0).unwrap_or_else(T::zero);
            let mut order: Vec<usize> = (0..ritz_vals.len()).collect();
            order.sort_by(|&i, &j| {
                let di = Scalar::abs(ritz_vals[i].clone() - target_center.clone());
                let dj = Scalar::abs(ritz_vals[j].clone() - target_center.clone());
                di.partial_cmp(&dj).unwrap_or(std::cmp::Ordering::Equal)
            });

            for &idx in &order {
                let theta = ritz_vals[idx].clone();

                // Only accept eigenvalues that lie inside the target interval.
                if theta < self.config.target_low || theta > self.config.target_high {
                    continue;
                }

                // Skip eigenvalues already discovered in a previous iteration.
                if converged_eigenvalues
                    .iter()
                    .any(|e| Scalar::abs(e.clone() - theta.clone()) <= dedup_tol)
                {
                    continue;
                }

                // Form the full Ritz vector x = Q y (already normalized).
                let ritz_vec = self.compute_ritz_vector(&q, &ritz_vecs[idx]);
                if ritz_vec.is_empty() {
                    continue;
                }

                // Verify against A: residual ||A x - theta x||.
                let mut ax = vec![T::zero(); n];
                spmv(T::one(), a, &ritz_vec, T::zero(), &mut ax);
                let residual: T = Real::sqrt(
                    ax.iter()
                        .zip(ritz_vec.iter())
                        .map(|(axi, xi)| {
                            let diff = axi.clone() - theta.clone() * xi.clone();
                            diff.clone() * diff
                        })
                        .fold(T::zero(), |acc, x| acc + x),
                );

                if residual <= self.config.tolerance {
                    converged_eigenvalues.push(theta.clone());
                    converged_residuals.push(residual);
                    if self.config.compute_eigenvectors {
                        converged_eigenvectors.push(ritz_vec);
                    }
                }
            }

            // Check if we have enough converged eigenvalues.
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

            // Restart: seed the next iteration with the Ritz vectors closest to
            // the target interval so the filter can refine them further.
            v_basis.clear();
            for &idx in order.iter().take(num_start_vectors) {
                let ritz_vec = self.compute_ritz_vector(&q, &ritz_vecs[idx]);
                if !ritz_vec.is_empty() {
                    v_basis.push(ritz_vec);
                }
            }

            if v_basis.is_empty() {
                // Add random vectors if no candidates were produced.
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

    /// Coefficients of the degree-`polynomial_degree` Chebyshev band-pass
    /// filter approximating the indicator of `[target_low, target_high]` over
    /// the spectral range `[lambda_min, lambda_max]`.
    ///
    /// Returns `c_k = g_k * mu_k`, consumed by [`apply_filter`] which evaluates
    /// `p(A) v = sum_k c_k * T_k((A - center*I)/half_width) v` via the
    /// Chebyshev three-term recurrence, where the affine map
    /// `x(lambda) = (2*lambda - lambda_max - lambda_min)/(lambda_max - lambda_min)`
    /// sends the spectrum onto `[-1, 1]`. The `mu_k` are the Chebyshev
    /// coefficients of the mapped target window: with `x = cos(theta)` the
    /// window is `theta in [theta_lo, theta_hi]`, so `mu_0 = (theta_hi -
    /// theta_lo)/pi` and `mu_k = (2/(k*pi))*(sin(k*theta_hi) - sin(k*theta_lo))`
    /// for `k >= 1`; the `g_k` are Jackson damping factors (`degree + 1`
    /// moments) suppressing Gibbs ringing. The result is `~ 1` inside the
    /// window and `~ 0` outside, genuinely amplifying target eigenvalues.
    fn compute_chebyshev_filter(
        &self,
        lambda_min: &T,
        lambda_max: &T,
        target_low: &T,
        target_high: &T,
    ) -> Vec<T> {
        let degree = self.config.polynomial_degree;
        let mut coeffs = vec![T::zero(); degree + 1];

        let pi_f64 = std::f64::consts::PI;
        let lmin = lambda_min.to_f64().unwrap_or(-1.0);
        let lmax = lambda_max.to_f64().unwrap_or(1.0);
        let width = lmax - lmin;

        // Degenerate spectral range: fall back to the identity filter (p == 1)
        // so that no spurious amplification or division by zero is introduced.
        if !(width.abs() > 0.0) {
            coeffs[0] = T::one();
            return coeffs;
        }

        // Affine map of the spectrum onto [-1, 1], then clamp the target
        // interval to the representable window.
        let map = |lambda: f64| (2.0 * lambda - lmax - lmin) / width;
        let a_low = target_low.to_f64().unwrap_or(lmin);
        let a_high = target_high.to_f64().unwrap_or(lmax);
        let mut y_low = map(a_low).clamp(-1.0, 1.0);
        let mut y_high = map(a_high).clamp(-1.0, 1.0);
        if y_low > y_high {
            std::mem::swap(&mut y_low, &mut y_high);
        }

        // In angle space x = cos(theta), the interval [y_low, y_high]
        // corresponds to theta in [acos(y_high), acos(y_low)] (acos is
        // decreasing, so acos(y_low) >= acos(y_high)).
        let theta_hi = y_low.acos();
        let theta_lo = y_high.acos();

        // Jackson kernel damping (number of moments = degree + 1).
        let n_f64 = (degree + 2) as f64;
        let cot_pi_n = (pi_f64 / n_f64).cos() / (pi_f64 / n_f64).sin();

        for (k, coeff) in coeffs.iter_mut().enumerate() {
            let k_f64 = k as f64;
            let ratio = k_f64 * pi_f64 / n_f64;
            let g_k = ((n_f64 - k_f64) * ratio.cos() + ratio.sin() * cot_pi_n) / n_f64;

            // Chebyshev coefficient of the window indicator on [-1, 1].
            let mu_k = if k == 0 {
                (theta_hi - theta_lo) / pi_f64
            } else {
                2.0 / (k_f64 * pi_f64) * ((k_f64 * theta_hi).sin() - (k_f64 * theta_lo).sin())
            };

            *coeff = T::from_f64(g_k * mu_k).unwrap_or_else(T::zero);
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

        // Scale and shift parameters. `e` is the spectral half-width and maps
        // [lambda_min, lambda_max] onto [-1, 1].
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

        // Only the constant term is well defined for a single coefficient or a
        // degenerate spectral range (half-width ~ 0); higher Chebyshev terms
        // would divide by `e`, so stop here to avoid producing NaNs.
        let eps = T::from_f64(1e-30).unwrap_or_else(T::zero);
        if coeffs.len() == 1 || Scalar::abs(e.clone()) <= eps {
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
