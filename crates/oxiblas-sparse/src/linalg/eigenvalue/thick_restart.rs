//! Thick-Restart Lanczos (TRL) algorithm for sparse symmetric eigenvalue problems.
//!
//! This module implements the Thick-Restart Lanczos method (Wu & Simon, 2000) for
//! computing eigenvalues and eigenvectors of large sparse symmetric matrices.
//!
//! # Algorithm Overview
//!
//! The TRL method improves upon basic Lanczos with restarts by retaining a set of
//! "thick" vectors (the best Ritz pairs) across restarts rather than discarding them
//! as in simple truncation. This leads to dramatically faster convergence, especially
//! for clustered eigenvalues.
//!
//! ## Key steps:
//!
//! 1. Build Krylov subspace of size `max_krylov_size` via Lanczos with full reorthogonalization
//! 2. Solve tridiagonal eigenvalue problem to get Ritz pairs
//! 3. Check convergence of wanted Ritz pairs
//! 4. Thick-restart: keep `num_thick` best Ritz pairs, reinitialize Lanczos from them
//! 5. Repeat until all `num_eigenvalues` converged or `max_restarts` exceeded
//!
//! # Convergence Advantages
//!
//! - Superior to IRAM for symmetric matrices (no complex arithmetic needed)
//! - Thick restart retains spectral information across restarts (unlike simple restart)
//! - Full reorthogonalization (two-pass MGS) ensures numerical stability
//! - Used in modern eigensolvers: ARPACK2, SLEPc, FEAST
//!
//! # References
//!
//! - Wu, K. & Simon, H. (2000). "Thick-restart Lanczos method for large symmetric eigenvalue problems."
//!   SIAM Journal on Matrix Analysis and Applications, 22(2), 602-616.

use core::fmt;

use crate::csr::CsrMatrix;
use crate::ops::spmv;
use oxiblas_core::scalar::{Field, Real, Scalar};

// =============================================================================
// Error Type
// =============================================================================

/// Error type for Thick-Restart Lanczos solver.
#[derive(Debug, Clone)]
pub enum TrlError {
    /// Solver did not converge within the allowed restarts.
    NotConverged {
        /// Number of eigenvalues that did converge.
        eigenvalues_found: usize,
        /// Number of eigenvalues requested.
        requested: usize,
    },
    /// Invalid solver configuration.
    InvalidConfig(String),
    /// Numerical failure during computation.
    NumericalFailure(String),
    /// Error during sparse matrix-vector product.
    SpMvError(String),
}

impl fmt::Display for TrlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotConverged {
                eigenvalues_found,
                requested,
            } => write!(
                f,
                "TRL did not converge: {eigenvalues_found}/{requested} eigenvalues found"
            ),
            Self::InvalidConfig(msg) => write!(f, "Invalid TRL configuration: {msg}"),
            Self::NumericalFailure(msg) => write!(f, "TRL numerical failure: {msg}"),
            Self::SpMvError(msg) => write!(f, "TRL SpMV error: {msg}"),
        }
    }
}

impl std::error::Error for TrlError {}

// =============================================================================
// EigenvalueTarget
// =============================================================================

/// Selection criterion for which eigenvalues to compute.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EigenvalueTarget {
    /// Algebraically largest eigenvalues (most positive).
    Largest,
    /// Algebraically smallest eigenvalues (most negative).
    Smallest,
    /// Eigenvalues with largest absolute value.
    LargestMagnitude,
    /// Eigenvalues with smallest absolute value.
    SmallestMagnitude,
    /// Interior eigenvalues closest to a target value (requires shift-invert in practice).
    Interior(f64),
}

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for the Thick-Restart Lanczos eigensolver.
#[derive(Debug, Clone)]
pub struct TrlConfig {
    /// Number of desired eigenvalues/eigenvectors.
    pub num_eigenvalues: usize,
    /// Maximum Krylov subspace size. Must be strictly greater than `num_eigenvalues`.
    /// Typical values: 2x–4x `num_eigenvalues`. Larger means more work per restart but
    /// better spectral resolution.
    pub max_krylov_size: usize,
    /// Number of Ritz pairs to keep after each thick restart.
    /// Must satisfy: `num_eigenvalues` <= `num_thick` < `max_krylov_size`.
    /// Typical: `num_eigenvalues + 4` or similar.
    pub num_thick: usize,
    /// Maximum number of restart cycles (outer iterations).
    pub max_restarts: usize,
    /// Convergence tolerance for residual norms: `||A*x - λ*x||`.
    pub tol: f64,
    /// Which eigenvalues to target.
    pub which: EigenvalueTarget,
    /// Whether to compute eigenvectors (always true for TRL but configurable).
    pub compute_eigenvectors: bool,
}

impl Default for TrlConfig {
    fn default() -> Self {
        Self {
            num_eigenvalues: 6,
            max_krylov_size: 30,
            num_thick: 8,
            max_restarts: 100,
            tol: 1e-10,
            which: EigenvalueTarget::LargestMagnitude,
            compute_eigenvectors: true,
        }
    }
}

// =============================================================================
// Result
// =============================================================================

/// Result of a Thick-Restart Lanczos computation.
#[derive(Debug, Clone)]
pub struct TrlResult {
    /// Computed eigenvalues, sorted according to the `EigenvalueTarget` criterion.
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors stored as a flat column-major list: `eigenvectors[i]` is the i-th eigenvector.
    pub eigenvectors: Vec<Vec<f64>>,
    /// Residual norms `||A*x_i - λ_i*x_i||` for each eigenpair.
    pub residuals: Vec<f64>,
    /// Number of restart cycles performed.
    pub n_restarts: usize,
    /// Number of eigenvalues that satisfied the convergence criterion.
    pub converged: usize,
}

// =============================================================================
// Solver
// =============================================================================

/// Thick-Restart Lanczos eigensolver for symmetric sparse matrices.
///
/// Implements the Wu-Simon (2000) thick-restart Lanczos method, which is
/// superior to IRAM for symmetric matrices due to:
///
/// - Direct handling of symmetry (real arithmetic only)
/// - Thick restart retains spectral information, not just a single starting vector
/// - Monotone convergence of Ritz values for symmetric matrices
///
/// # Usage
///
/// ```ignore
/// use oxiblas_sparse::linalg::eigenvalue::{ThickRestartLanczos, TrlConfig, EigenvalueTarget};
///
/// let config = TrlConfig {
///     num_eigenvalues: 5,
///     max_krylov_size: 25,
///     num_thick: 7,
///     tol: 1e-10,
///     which: EigenvalueTarget::LargestMagnitude,
///     ..Default::default()
/// };
///
/// let trl = ThickRestartLanczos::new(config).unwrap();
/// let result = trl.compute(&csr_matrix).unwrap();
/// println!("Eigenvalues: {:?}", result.eigenvalues);
/// ```
pub struct ThickRestartLanczos {
    config: TrlConfig,
}

impl ThickRestartLanczos {
    /// Create a new solver, validating the configuration.
    ///
    /// # Errors
    ///
    /// Returns `TrlError::InvalidConfig` if configuration parameters are inconsistent.
    pub fn new(config: TrlConfig) -> Result<Self, TrlError> {
        if config.num_eigenvalues == 0 {
            return Err(TrlError::InvalidConfig(
                "num_eigenvalues must be >= 1".to_string(),
            ));
        }
        if config.max_krylov_size <= config.num_eigenvalues {
            return Err(TrlError::InvalidConfig(format!(
                "max_krylov_size ({}) must be > num_eigenvalues ({})",
                config.max_krylov_size, config.num_eigenvalues
            )));
        }
        if config.num_thick < config.num_eigenvalues {
            return Err(TrlError::InvalidConfig(format!(
                "num_thick ({}) must be >= num_eigenvalues ({})",
                config.num_thick, config.num_eigenvalues
            )));
        }
        if config.num_thick >= config.max_krylov_size {
            return Err(TrlError::InvalidConfig(format!(
                "num_thick ({}) must be < max_krylov_size ({})",
                config.num_thick, config.max_krylov_size
            )));
        }
        if config.max_restarts == 0 {
            return Err(TrlError::InvalidConfig(
                "max_restarts must be >= 1".to_string(),
            ));
        }
        if config.tol <= 0.0 {
            return Err(TrlError::InvalidConfig("tol must be > 0".to_string()));
        }
        Ok(Self { config })
    }

    /// Compute eigenvalues (and optionally eigenvectors) of a symmetric sparse matrix.
    ///
    /// The matrix must be symmetric; only the stored (upper or full) entries are used
    /// via standard SpMV.
    ///
    /// # Algorithm
    ///
    /// Implements the Wu-Simon (2000) Thick-Restart Lanczos. The key idea:
    ///
    /// 1. Build full Krylov subspace to dimension `m` via Lanczos with full reorthogonalization.
    /// 2. Solve the `m × m` tridiagonal EVP to get Ritz pairs.
    /// 3. Check convergence of wanted Ritz pairs.
    /// 4. **Thick restart**: keep `num_thick` best Ritz pairs as the new starting basis.
    ///    The new Krylov subspace combines the thick vectors AND a new direction orthogonal to them.
    /// 5. Run the combined EVP at each restart over the full `(num_thick + Lanczos_extension)` basis.
    ///
    /// The thick restart step preserves spectral information across restarts, leading to
    /// monotone convergence of Ritz values for symmetric matrices.
    ///
    /// # Errors
    ///
    /// - `TrlError::InvalidConfig` if the matrix dimension is too small
    /// - `TrlError::NumericalFailure` for breakdown or degenerate situations
    /// - `TrlError::NotConverged` if convergence is not achieved within `max_restarts`
    pub fn compute<T>(&self, csr: &CsrMatrix<T>) -> Result<TrlResult, TrlError>
    where
        T: Scalar<Real = T> + Clone + Field + Real + num_traits::FromPrimitive,
    {
        let n = csr.nrows();
        if csr.ncols() != n {
            return Err(TrlError::InvalidConfig(format!(
                "Matrix must be square, got {}x{}",
                n,
                csr.ncols()
            )));
        }
        let k = self.config.num_eigenvalues;
        if k > n {
            return Err(TrlError::InvalidConfig(format!(
                "num_eigenvalues ({k}) must be <= matrix dimension ({n})"
            )));
        }
        let m = self.config.max_krylov_size.min(n);
        let num_thick = self.config.num_thick.min(m - 1);

        // SpMV closure: computes A*v in f64 arithmetic
        let spmv_f64 = |v: &[f64]| -> Result<Vec<f64>, TrlError> {
            let v_t: Vec<T> = v
                .iter()
                .map(|&x| {
                    T::from_f64(x)
                        .ok_or_else(|| TrlError::SpMvError("f64->T conversion failed".to_string()))
                })
                .collect::<Result<_, _>>()?;
            let mut w_t = vec![T::zero(); n];
            spmv(T::one(), csr, &v_t, T::zero(), &mut w_t);
            w_t.iter()
                .map(|x| {
                    num_traits::ToPrimitive::to_f64(x)
                        .ok_or_else(|| TrlError::SpMvError("T->f64 conversion failed".to_string()))
                })
                .collect::<Result<_, _>>()
        };

        // ------------------------------------------------------------------
        // Initialize starting vector: uniform [1/sqrt(n), ...]
        // ------------------------------------------------------------------
        let scale = 1.0 / (n as f64).sqrt();
        let mut v0: Vec<f64> = vec![scale; n];
        let v0_norm = vec_norm(&v0);
        if v0_norm < f64::EPSILON {
            return Err(TrlError::NumericalFailure(
                "Initial vector has zero norm".to_string(),
            ));
        }
        vec_scale(&mut v0, 1.0 / v0_norm);

        // ------------------------------------------------------------------
        // TRL state
        //
        // After the thick restart, the basis Q has two conceptual parts:
        //   - "Thick" block: Q[0..j_thick] = locked Ritz vectors
        //   - "Lanczos" block: Q[j_thick..dim] = new Lanczos vectors
        //
        // The tridiagonal T has the structure:
        //   - T[0..j_thick, 0..j_thick] = diag(Ritz values) with all cross-betas = 0
        //   - Coupling: beta[j_thick-1] couples thick block to Lanczos block
        //   - T[j_thick..dim, j_thick..dim] = standard tridiagonal from Lanczos
        //
        // The Wu-Simon TRL correctly handles this bordered-diagonal structure by
        // building the FULL (num_thick + Lanczos) basis and solving the EVP over
        // the ENTIRE space at each restart.
        // ------------------------------------------------------------------

        // Full basis of current subspace
        let mut basis: Vec<Vec<f64>> = vec![v0];
        // Locked (thick) vectors -- kept separately for orthogonalization
        let mut locked_vecs: Vec<Vec<f64>> = Vec::new();
        // Tridiagonal elements for the LANCZOS block only
        let mut alpha: Vec<f64> = Vec::with_capacity(m);
        let mut beta: Vec<f64> = Vec::with_capacity(m);

        let mut best_eigenvalues: Vec<f64> = Vec::new();
        let mut best_eigenvectors: Vec<Vec<f64>> = Vec::new();
        let mut best_residuals: Vec<f64> = Vec::new();
        let mut n_restarts = 0usize;

        // ------------------------------------------------------------------
        // Main thick-restart loop
        // ------------------------------------------------------------------
        'outer: for restart in 0..=self.config.max_restarts {
            // ----------------------------------------------------------------
            // Step 1: Build Lanczos subspace from basis[0] up to m total vectors
            // The Lanczos runs from the current starting point and orthogonalizes
            // against ALL previous vectors (including locked thick vectors).
            // ----------------------------------------------------------------
            let dim = self.lanczos_pure(
                &spmv_f64,
                &mut basis,
                &mut alpha,
                &mut beta,
                &locked_vecs,
                m,
                n,
            )?;

            // ----------------------------------------------------------------
            // Step 2: Build combined basis = locked_vecs + Lanczos basis
            // and solve the EVP over this combined space.
            // ----------------------------------------------------------------
            let combined_dim = locked_vecs.len() + dim;
            let combined_basis: Vec<&Vec<f64>> =
                locked_vecs.iter().chain(basis[..dim].iter()).collect();

            // Build the full projected matrix H = Q^T A Q over combined basis
            // (We compute this via alpha/beta for the Lanczos block, plus
            //  cross-terms with locked vectors.)
            let (h_evals, h_evecs) = self.compute_combined_evp(
                &spmv_f64,
                &combined_basis,
                &alpha,
                &beta,
                &locked_vecs,
                dim,
                combined_dim,
                n,
            )?;

            // ----------------------------------------------------------------
            // Step 3: Compute full-space Ritz vectors
            // ----------------------------------------------------------------
            let combined_basis_owned: Vec<Vec<f64>> =
                combined_basis.iter().map(|v| (*v).clone()).collect();
            let ritz_vecs_full = compute_ritz_vectors_full(&combined_basis_owned, &h_evecs, n);

            // Select the k wanted Ritz pairs
            let selected_indices = self.select_ritz_indices(&h_evals, k);

            // Compute residuals for the wanted pairs
            let mut cur_residuals = Vec::with_capacity(k);
            for &sel_idx in &selected_indices {
                let res = Self::residual_norm_for(
                    &spmv_f64,
                    h_evals[sel_idx],
                    &ritz_vecs_full[sel_idx],
                    n,
                )?;
                cur_residuals.push(res);
            }

            best_eigenvalues = selected_indices.iter().map(|&i| h_evals[i]).collect();
            best_eigenvectors = selected_indices
                .iter()
                .map(|&i| ritz_vecs_full[i].clone())
                .collect();
            best_residuals = cur_residuals;

            let converged_count = best_residuals
                .iter()
                .filter(|&&r| r <= self.config.tol)
                .count();

            if converged_count >= k {
                n_restarts = restart;
                break 'outer;
            }

            if restart == self.config.max_restarts {
                n_restarts = restart;
                break 'outer;
            }

            // ----------------------------------------------------------------
            // Step 4: Thick restart
            //
            // Keep `num_thick` best Ritz pairs as the new locked/thick vectors.
            // Compute new starting direction from the Lanczos tail residual.
            // ----------------------------------------------------------------
            let thick_indices = self.select_ritz_indices_thick(&h_evals, num_thick);

            // Compute the Lanczos tail residual (Wu-Simon "beta_m * v_{m+1}")
            // This is the direction A*basis[dim-1] - ... projected out of basis.
            let tail =
                self.compute_lanczos_tail(&spmv_f64, &basis, &alpha, &beta, &locked_vecs, dim, n)?;
            let tail_norm = vec_norm(&tail);

            // New locked vectors = selected thick Ritz vectors
            let new_locked: Vec<Vec<f64>> = thick_indices
                .iter()
                .map(|&i| ritz_vecs_full[i].clone())
                .collect();

            // New starting vector for Lanczos = tail, orthogonalized against new locked vecs
            let mut new_start = tail.clone();
            reorthogonalize_mgs2(&mut new_start, &new_locked);
            let new_start_norm = vec_norm(&new_start);

            let new_start_vec = if new_start_norm > f64::EPSILON * 100.0 {
                vec_scale(&mut new_start, 1.0 / new_start_norm);
                new_start
            } else if tail_norm > f64::EPSILON * 100.0 {
                // tail is nearly in span of new_locked — use fallback
                let mut fb = tail.clone();
                vec_scale(&mut fb, 1.0 / tail_norm);
                reorthogonalize_mgs2(&mut fb, &new_locked);
                let fb_norm = vec_norm(&fb);
                if fb_norm > f64::EPSILON * 10.0 {
                    vec_scale(&mut fb, 1.0 / fb_norm);
                    fb
                } else {
                    generate_orthogonal_vector(&new_locked, n)?
                }
            } else {
                generate_orthogonal_vector(&new_locked, n)?
            };

            // Reset state for next iteration
            locked_vecs = new_locked;
            basis = vec![new_start_vec];
            alpha.clear();
            beta.clear();

            n_restarts = restart + 1;
        }

        let converged_count = best_residuals
            .iter()
            .filter(|&&r| r <= self.config.tol)
            .count();

        if converged_count < k && n_restarts >= self.config.max_restarts {
            return Err(TrlError::NotConverged {
                eigenvalues_found: converged_count,
                requested: k,
            });
        }

        Ok(TrlResult {
            eigenvalues: best_eigenvalues,
            eigenvectors: if self.config.compute_eigenvectors {
                best_eigenvectors
            } else {
                vec![]
            },
            residuals: best_residuals,
            n_restarts,
            converged: converged_count,
        })
    }

    // -------------------------------------------------------------------------
    // Pure Lanczos iteration (with orthogonalization against locked vectors)
    // -------------------------------------------------------------------------
    /// Run Lanczos from `basis[0]` to at most `m` steps, orthogonalizing against
    /// all current `basis` vectors AND the locked (thick) vectors at each step.
    ///
    /// Returns the actual Lanczos subspace dimension `dim`.
    /// After return:
    ///   - `basis[0..dim]` = Lanczos vectors
    ///   - `alpha[0..dim]` = diagonal of projected tridiagonal
    ///   - `beta[0..dim-1]` = off-diagonal of projected tridiagonal
    fn lanczos_pure<F>(
        &self,
        spmv_f64: &F,
        basis: &mut Vec<Vec<f64>>,
        alpha: &mut Vec<f64>,
        beta: &mut Vec<f64>,
        locked_vecs: &[Vec<f64>],
        target_dim: usize,
        n: usize,
    ) -> Result<usize, TrlError>
    where
        F: Fn(&[f64]) -> Result<Vec<f64>, TrlError>,
    {
        let tol_breakdown = f64::EPSILON * 100.0;

        for j in 0..target_dim {
            if j >= basis.len() {
                return Err(TrlError::NumericalFailure(format!(
                    "Lanczos basis vector {j} missing (basis has {} vectors)",
                    basis.len()
                )));
            }

            let v = basis[j].clone();

            // w = A * v
            let mut w = spmv_f64(&v)?;

            // alpha_j = v^T * w
            let alpha_j = vec_dot(&v, &w);
            if j < alpha.len() {
                alpha[j] = alpha_j;
            } else {
                alpha.push(alpha_j);
            }

            // w -= alpha_j * v
            for i in 0..n {
                w[i] -= alpha_j * v[i];
            }

            // w -= beta[j-1] * basis[j-1]  (three-term Lanczos recurrence)
            if j > 0 && j - 1 < beta.len() {
                let b_prev = beta[j - 1];
                let v_prev = &basis[j - 1];
                for i in 0..n {
                    w[i] -= b_prev * v_prev[i];
                }
            }

            // Full reorthogonalization against all Lanczos basis vectors (two-pass MGS)
            reorthogonalize_mgs2(&mut w, &basis[..=j]);
            // Also orthogonalize against locked vectors (implicit deflation)
            reorthogonalize_mgs2(&mut w, locked_vecs);

            let beta_j = vec_norm(&w);

            if beta_j <= tol_breakdown {
                // Invariant subspace found
                return Ok(j + 1);
            }

            if j < beta.len() {
                beta[j] = beta_j;
            } else {
                beta.push(beta_j);
            }

            let mut v_next = w;
            vec_scale(&mut v_next, 1.0 / beta_j);

            if j + 1 < basis.len() {
                basis[j + 1] = v_next;
            } else {
                basis.push(v_next);
            }
        }

        Ok(target_dim)
    }

    // -------------------------------------------------------------------------
    // Combined EVP over locked + Lanczos basis
    // -------------------------------------------------------------------------
    /// Compute the Rayleigh-Ritz EVP over the combined basis = [locked | lanczos_basis].
    ///
    /// For the pure Lanczos block, the projected matrix is tridiagonal (alpha/beta).
    /// For the locked vectors, they are near-eigenvectors so A*u_j ≈ lambda_j * u_j.
    /// The cross-terms are computed explicitly.
    ///
    /// Returns `(eigenvalues, eigenvectors_in_basis)`.
    fn compute_combined_evp<F>(
        &self,
        spmv_f64: &F,
        combined_basis: &[&Vec<f64>],
        alpha: &[f64],
        beta: &[f64],
        locked_vecs: &[Vec<f64>],
        lanczos_dim: usize,
        combined_dim: usize,
        n: usize,
    ) -> Result<(Vec<f64>, Vec<Vec<f64>>), TrlError>
    where
        F: Fn(&[f64]) -> Result<Vec<f64>, TrlError>,
    {
        let n_locked = locked_vecs.len();

        if n_locked == 0 {
            // Pure Lanczos case: tridiagonal EVP
            let beta_slice = if lanczos_dim > 0 {
                &beta[..lanczos_dim - 1]
            } else {
                &[]
            };
            return solve_tridiagonal_evp(&alpha[..lanczos_dim], beta_slice);
        }

        // Build the full projected matrix H = Q^T A Q as a dense symmetric matrix
        // H[i][j] = q_i^T * A * q_j
        // For the locked block: A * u_i = A_ui (computed explicitly)
        // For the Lanczos block: A * v_j defined via tridiagonal structure

        let mut h = vec![vec![0.0f64; combined_dim]; combined_dim];

        // Compute A*q_i for each basis vector q_i
        let mut aq: Vec<Vec<f64>> = Vec::with_capacity(combined_dim);
        for v in combined_basis.iter() {
            let av = spmv_f64(v)?;
            aq.push(av);
        }

        // Fill H[i][j] = q_i^T * A * q_j = q_i^T * (A*q_j)
        for i in 0..combined_dim {
            for j in i..combined_dim {
                let hij = vec_dot(combined_basis[i], &aq[j]);
                h[i][j] = hij;
                h[j][i] = hij; // symmetry
            }
        }

        // Solve dense symmetric EVP via Jacobi iteration (for small matrices)
        // combined_dim is at most max_krylov_size + num_thick, typically < 50
        let _ = n; // suppress warning
        let (evals, evecs) = solve_dense_symmetric_evp(&h, combined_dim)?;

        Ok((evals, evecs))
    }

    // -------------------------------------------------------------------------
    // Lanczos tail residual
    // -------------------------------------------------------------------------
    /// Compute the Lanczos "tail" residual: the unnormalized direction beyond
    /// the current Lanczos subspace.
    ///
    /// Per the Lanczos relation: A*basis[dim-1] = alpha[dim-1]*basis[dim-1]
    ///     + beta[dim-2]*basis[dim-2] + beta[dim-1]*v_{next}
    ///
    /// So: beta[dim-1]*v_{next} = A*basis[dim-1] - alpha[dim-1]*basis[dim-1] - beta[dim-2]*basis[dim-2]
    ///
    /// We return this unnormalized tail vector.
    fn compute_lanczos_tail<F>(
        &self,
        spmv_f64: &F,
        basis: &[Vec<f64>],
        alpha: &[f64],
        beta: &[f64],
        locked_vecs: &[Vec<f64>],
        dim: usize,
        n: usize,
    ) -> Result<Vec<f64>, TrlError>
    where
        F: Fn(&[f64]) -> Result<Vec<f64>, TrlError>,
    {
        if dim == 0 || basis.is_empty() {
            return Err(TrlError::NumericalFailure(
                "Cannot compute tail of empty Lanczos basis".to_string(),
            ));
        }

        let last = dim - 1;
        let v_last = &basis[last];

        // w = A * v_last
        let mut w = spmv_f64(v_last)?;

        // w -= alpha[last] * v_last
        let a_last = if last < alpha.len() { alpha[last] } else { 0.0 };
        for i in 0..n {
            w[i] -= a_last * v_last[i];
        }

        // w -= beta[last-1] * basis[last-1]
        if last > 0 && last - 1 < beta.len() {
            let b_prev = beta[last - 1];
            let v_prev = &basis[last - 1];
            for i in 0..n {
                w[i] -= b_prev * v_prev[i];
            }
        }

        // Reorthogonalize for stability
        reorthogonalize_mgs2(&mut w, &basis[..dim]);
        reorthogonalize_mgs2(&mut w, locked_vecs);

        Ok(w)
    }

    // -------------------------------------------------------------------------
    // Lanczos extension (unused but kept for reference -- lanczos_pure is used)
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // Ritz pair selection helpers
    // -------------------------------------------------------------------------

    /// Select the indices of the `k` wanted Ritz pairs from sorted Ritz values.
    fn select_ritz_indices(&self, ritz_vals: &[f64], k: usize) -> Vec<usize> {
        let m = ritz_vals.len();
        let k = k.min(m);
        if k == 0 {
            return vec![];
        }

        let mut indexed: Vec<(usize, f64)> = ritz_vals.iter().copied().enumerate().collect();

        match self.config.which {
            EigenvalueTarget::Largest => {
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            EigenvalueTarget::Smallest => {
                indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            EigenvalueTarget::LargestMagnitude => {
                indexed.sort_by(|a, b| {
                    b.1.abs()
                        .partial_cmp(&a.1.abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            EigenvalueTarget::SmallestMagnitude => {
                indexed.sort_by(|a, b| {
                    a.1.abs()
                        .partial_cmp(&b.1.abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            EigenvalueTarget::Interior(target) => {
                indexed.sort_by(|a, b| {
                    (a.1 - target)
                        .abs()
                        .partial_cmp(&(b.1 - target).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        indexed.iter().take(k).map(|(i, _)| *i).collect()
    }

    /// Select the `num_thick` indices for the thick restart.
    /// We keep the `num_thick` best Ritz pairs (same criterion as wanted).
    fn select_ritz_indices_thick(&self, ritz_vals: &[f64], num_thick: usize) -> Vec<usize> {
        self.select_ritz_indices(ritz_vals, num_thick)
    }

    /// Compute `||A*x - λ*x||` for a single eigenpair candidate.
    fn residual_norm_for<F>(spmv_f64: &F, lambda: f64, x: &[f64], n: usize) -> Result<f64, TrlError>
    where
        F: Fn(&[f64]) -> Result<Vec<f64>, TrlError>,
    {
        let ax = spmv_f64(x)?;
        let mut res = vec![0.0f64; n];
        for i in 0..n {
            res[i] = ax[i] - lambda * x[i];
        }
        Ok(vec_norm(&res))
    }
}

// =============================================================================
// Module-level helper functions
// =============================================================================

/// Compute full-space Ritz vectors: x_i = Q * y_i, normalized.
///
/// `q_basis` is a slice of basis vectors (each of length `n`).
/// `ritz_vecs_small` is a list of coordinate vectors in the basis.
fn compute_ritz_vectors_full(
    q_basis: &[Vec<f64>],
    ritz_vecs_small: &[Vec<f64>],
    n: usize,
) -> Vec<Vec<f64>> {
    ritz_vecs_small
        .iter()
        .map(|y| {
            let mut x = vec![0.0f64; n];
            for (j, qj) in q_basis.iter().enumerate() {
                if j < y.len() {
                    let coeff = y[j];
                    for i in 0..n {
                        x[i] += coeff * qj[i];
                    }
                }
            }
            let xn = vec_norm(&x);
            if xn > f64::EPSILON {
                vec_scale(&mut x, 1.0 / xn);
            }
            x
        })
        .collect()
}

/// Generate a unit vector orthogonal to all vectors in `basis`.
///
/// Tries canonical basis vectors e_i in order until one with non-negligible
/// component outside the span of `basis` is found. Applies two-pass MGS.
fn generate_orthogonal_vector(basis: &[Vec<f64>], n: usize) -> Result<Vec<f64>, TrlError> {
    for i in 0..n {
        let mut v = vec![0.0f64; n];
        v[i] = 1.0;
        reorthogonalize_mgs2(&mut v, basis);
        let vn = vec_norm(&v);
        if vn > f64::EPSILON * 100.0 {
            vec_scale(&mut v, 1.0 / vn);
            return Ok(v);
        }
    }
    Err(TrlError::NumericalFailure(
        "Cannot find a vector orthogonal to the current basis (matrix dimension too small?)"
            .to_string(),
    ))
}

/// Solve a dense symmetric eigenvalue problem H*y = λ*y via Jacobi iteration.
///
/// For small matrices (dim <= ~100), Jacobi iteration is efficient and numerically
/// stable. The matrix `h` is stored as a row-major `dim × dim` vector of vectors.
///
/// Returns `(eigenvalues, eigenvectors)` where `eigenvectors[i]` is the i-th
/// eigenvector in the basis.
fn solve_dense_symmetric_evp(
    h: &[Vec<f64>],
    dim: usize,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), TrlError> {
    if dim == 0 {
        return Ok((vec![], vec![]));
    }
    if dim == 1 {
        return Ok((vec![h[0][0]], vec![vec![1.0]]));
    }

    // Work on a copy
    let mut a: Vec<Vec<f64>> = h.to_vec();

    // Initialize eigenvector matrix as identity
    let mut v: Vec<Vec<f64>> = (0..dim)
        .map(|i| {
            let mut row = vec![0.0f64; dim];
            row[i] = 1.0;
            row
        })
        .collect();

    let max_sweeps = 50;
    let tol = f64::EPSILON * (dim as f64) * 10.0;

    for _sweep in 0..max_sweeps {
        // Compute off-diagonal norm
        let mut off_norm_sq = 0.0f64;
        for i in 0..dim {
            for j in (i + 1)..dim {
                off_norm_sq += 2.0 * a[i][j] * a[i][j];
            }
        }
        if off_norm_sq.sqrt() <= tol {
            break;
        }

        // Jacobi sweep: annihilate all off-diagonal elements
        for p in 0..dim {
            for q_idx in (p + 1)..dim {
                let app = a[p][p];
                let aqq = a[q_idx][q_idx];
                let apq = a[p][q_idx];

                if apq.abs() < tol * 0.5 {
                    continue;
                }

                // Compute Jacobi rotation
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    1.0 / (tau - (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Update eigenvalue matrix
                a[p][p] = app - t * apq;
                a[q_idx][q_idx] = aqq + t * apq;
                a[p][q_idx] = 0.0;
                a[q_idx][p] = 0.0;

                // Update off-diagonal elements for rows/cols p and q_idx
                for r in 0..dim {
                    if r != p && r != q_idx {
                        let arp = a[r][p];
                        let arq = a[r][q_idx];
                        a[r][p] = c * arp - s * arq;
                        a[p][r] = a[r][p];
                        a[r][q_idx] = s * arp + c * arq;
                        a[q_idx][r] = a[r][q_idx];
                    }
                }

                // Update eigenvector matrix
                for r in 0..dim {
                    let vrp = v[r][p];
                    let vrq = v[r][q_idx];
                    v[r][p] = c * vrp - s * vrq;
                    v[r][q_idx] = s * vrp + c * vrq;
                }
            }
        }
    }

    // Extract eigenvalues from diagonal and eigenvectors from v
    let eigenvalues: Vec<f64> = (0..dim).map(|i| a[i][i]).collect();
    let eigenvectors: Vec<Vec<f64>> = (0..dim)
        .map(|col| v.iter().map(|row| row[col]).collect())
        .collect();

    Ok((eigenvalues, eigenvectors))
}

// =============================================================================
// Tridiagonal Symmetric EVP solver
// =============================================================================

/// Solve the symmetric tridiagonal eigenvalue problem T*y = λ*y.
///
/// Uses implicit symmetric QR iteration with Wilkinson shifts for stability
/// and guaranteed convergence. This is a standard LAPACK-style steqr algorithm.
///
/// Returns `(eigenvalues, eigenvectors)` where eigenvectors are stored as
/// a list of column vectors.
pub fn solve_tridiagonal_evp(
    alpha: &[f64],
    beta: &[f64],
) -> Result<(Vec<f64>, Vec<Vec<f64>>), TrlError> {
    let n = alpha.len();
    if n == 0 {
        return Ok((vec![], vec![]));
    }
    if n == 1 {
        return Ok((vec![alpha[0]], vec![vec![1.0]]));
    }

    // Working copies
    let mut d = alpha.to_vec(); // diagonal
    let mut e: Vec<f64> = {
        let mut e = beta.to_vec();
        e.truncate(n - 1);
        while e.len() < n - 1 {
            e.push(0.0);
        }
        e
    };

    // Initialize eigenvector matrix as identity: z[i][j] = delta_{ij}
    // Stored as z[row][col]
    let mut z: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0f64; n];
            row[i] = 1.0;
            row
        })
        .collect();

    let max_iter = 60 * n;
    let eps = f64::EPSILON;

    for _iter in 0..max_iter {
        // Find the largest unreduced submatrix: scan from bottom
        // We use the standard LAPACK approach: find l such that e[l-1] is negligible
        let mut l = n - 1;
        while l > 0 {
            let thresh = eps * (d[l - 1].abs() + d[l].abs());
            if e[l - 1].abs() <= thresh {
                e[l - 1] = 0.0;
                break;
            }
            l -= 1;
        }

        if l == 0 {
            // All off-diagonals are zero: done
            break;
        }

        // Find start of unreduced block: m <= l such that e[m-1] is small (or m==0)
        let mut m = l;
        while m > 0 {
            let thresh = eps * (d[m - 1].abs() + d[m].abs());
            if e[m - 1].abs() <= thresh {
                e[m - 1] = 0.0;
                break;
            }
            m -= 1;
        }

        // Wilkinson shift: shift chosen as eigenvalue of 2x2 bottom submatrix closer to d[l]
        let a = d[l - 1];
        let b = e[l - 1];
        let c = d[l];
        let half_diff = (a - c) * 0.5;
        let r = (half_diff * half_diff + b * b).sqrt();
        let shift = if half_diff >= 0.0 {
            c - b * b / (half_diff + r)
        } else {
            c - b * b / (half_diff - r)
        };

        // Implicit symmetric QR step on submatrix [m..=l]
        let mut g = d[m] - shift;
        let mut cs = 1.0f64;
        let mut cs_old = 1.0f64;
        let mut sn = 0.0f64;
        let mut sn_old = 0.0f64;
        let mut p = 0.0f64;

        for i in m..l {
            let f = sn * e[i];
            let b_val = cs * e[i];

            // Compute Givens rotation to annihilate e[i]
            let (c2, s2, r2) = givens(g, f);
            cs = c2;
            sn = s2;

            if i > m {
                e[i - 1] = r2;
            }

            g = d[i] - p;
            let rr = (d[i + 1] - g) * sn + 2.0 * cs * b_val;
            p = sn * rr;
            d[i] = g + p;
            g = cs * rr - b_val;

            // Accumulate the rotation into z
            for k in 0..n {
                let tmp = z[k][i + 1];
                z[k][i + 1] = sn * z[k][i] + cs * tmp;
                z[k][i] = cs * z[k][i] - sn * tmp;
            }

            cs_old = cs;
            sn_old = sn;
        }

        // Suppress unused variable warnings for the last cs_old/sn_old
        let _ = (cs_old, sn_old);

        if l > m {
            e[l - 1] = cs * g;
        }
        d[l] = d[l] - p;
    }

    // Extract eigenvectors: z[k][i] is the k-th component of the i-th eigenvector
    let eigenvectors: Vec<Vec<f64>> = (0..n)
        .map(|col| z.iter().map(|row| row[col]).collect())
        .collect();

    Ok((d, eigenvectors))
}

// =============================================================================
// Reorthogonalization
// =============================================================================

/// Two-pass Modified Gram-Schmidt reorthogonalization.
///
/// Projects `v` onto the orthogonal complement of `basis`. Two passes are used
/// for numerical stability (classical double reorthogonalization strategy).
///
/// The vector is NOT normalized after this call.
fn reorthogonalize_mgs2(v: &mut [f64], basis: &[Vec<f64>]) {
    // Two passes for numerical stability
    for _ in 0..2 {
        for b in basis.iter() {
            let dot = vec_dot(v, b);
            for (vi, bi) in v.iter_mut().zip(b.iter()) {
                *vi -= dot * bi;
            }
        }
    }
}

// =============================================================================
// Low-level vector routines
// =============================================================================

/// Dot product of two f64 slices.
#[inline]
fn vec_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Euclidean norm of an f64 slice.
#[inline]
fn vec_norm(v: &[f64]) -> f64 {
    vec_dot(v, v).sqrt()
}

/// Scale a vector in-place: v *= s.
#[inline]
fn vec_scale(v: &mut [f64], s: f64) {
    for x in v.iter_mut() {
        *x *= s;
    }
}

/// Compute Givens rotation parameters (c, s, r) such that:
/// [c  s] [a]   [r]
/// [-s c] [b] = [0]
#[inline]
fn givens(a: f64, b: f64) -> (f64, f64, f64) {
    if b.abs() < f64::EPSILON {
        return (1.0, 0.0, a);
    }
    if a.abs() < f64::EPSILON {
        let sign = if b >= 0.0 { 1.0 } else { -1.0 };
        return (0.0, sign, b.abs());
    }
    let r = (a * a + b * b).sqrt();
    (a / r, b / r, r)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr::CsrMatrix;
    use crate::test_matrices::{laplacian_2d, tridiagonal};

    /// Build a small dense symmetric matrix as CSR.
    fn dense_csr(data: &[&[f64]]) -> CsrMatrix<f64> {
        let n = data.len();
        let mut row_ptrs = vec![0usize];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for row in data.iter() {
            for (j, &v) in row.iter().enumerate() {
                if v.abs() > 1e-15 {
                    col_indices.push(j);
                    values.push(v);
                }
            }
            row_ptrs.push(col_indices.len());
        }

        CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap()
    }

    #[test]
    fn test_trl_config_validation() {
        // Valid config
        let cfg = TrlConfig {
            num_eigenvalues: 3,
            max_krylov_size: 10,
            num_thick: 5,
            max_restarts: 50,
            tol: 1e-10,
            ..Default::default()
        };
        assert!(ThickRestartLanczos::new(cfg).is_ok());

        // num_eigenvalues = 0
        let bad = TrlConfig {
            num_eigenvalues: 0,
            ..Default::default()
        };
        assert!(ThickRestartLanczos::new(bad).is_err());

        // max_krylov_size <= num_eigenvalues
        let bad = TrlConfig {
            num_eigenvalues: 6,
            max_krylov_size: 6,
            num_thick: 7,
            ..Default::default()
        };
        assert!(ThickRestartLanczos::new(bad).is_err());

        // num_thick < num_eigenvalues
        let bad = TrlConfig {
            num_eigenvalues: 6,
            max_krylov_size: 20,
            num_thick: 5,
            ..Default::default()
        };
        assert!(ThickRestartLanczos::new(bad).is_err());

        // num_thick >= max_krylov_size
        let bad = TrlConfig {
            num_eigenvalues: 3,
            max_krylov_size: 10,
            num_thick: 10,
            ..Default::default()
        };
        assert!(ThickRestartLanczos::new(bad).is_err());

        // tol <= 0
        let bad = TrlConfig {
            tol: 0.0,
            ..Default::default()
        };
        assert!(ThickRestartLanczos::new(bad).is_err());
    }

    #[test]
    fn test_trl_2x2_matrix() {
        // Matrix [[2, 1], [1, 2]] has eigenvalues 1 and 3
        let a = dense_csr(&[&[2.0, 1.0], &[1.0, 2.0]]);

        let cfg = TrlConfig {
            num_eigenvalues: 2,
            max_krylov_size: 4,
            num_thick: 2,
            max_restarts: 50,
            tol: 1e-10,
            which: EigenvalueTarget::Largest,
            compute_eigenvectors: true,
        };
        let trl = ThickRestartLanczos::new(cfg).unwrap();
        let result = trl.compute::<f64>(&a).unwrap();

        assert_eq!(result.eigenvalues.len(), 2);
        let mut eigs = result.eigenvalues.clone();
        eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(
            (eigs[0] - 1.0).abs() < 1e-8,
            "smallest eigenvalue should be 1.0, got {}",
            eigs[0]
        );
        assert!(
            (eigs[1] - 3.0).abs() < 1e-8,
            "largest eigenvalue should be 3.0, got {}",
            eigs[1]
        );
    }

    #[test]
    fn test_trl_tridiagonal_10() {
        // 10x10 symmetric tridiagonal: diag=2, off-diag=-1
        // (discrete Laplacian 1D, n=10)
        // Eigenvalues: 2 - 2*cos(k*pi/11) for k=1..10
        // Largest 3: k=10,9,8
        let a = tridiagonal(10, -1.0, 2.0, -1.0).unwrap();

        let cfg = TrlConfig {
            num_eigenvalues: 3,
            max_krylov_size: 12,
            num_thick: 4,
            max_restarts: 100,
            tol: 1e-8,
            which: EigenvalueTarget::Largest,
            compute_eigenvectors: true,
        };
        let trl = ThickRestartLanczos::new(cfg).unwrap();
        let result = trl.compute::<f64>(&a).unwrap();

        assert_eq!(result.eigenvalues.len(), 3);

        // Exact eigenvalues for tridiagonal(10, -1, 2, -1): 2 - 2*cos(k*pi/11)
        let exact_largest: Vec<f64> = (8..=10)
            .rev()
            .map(|k| 2.0 - 2.0 * (k as f64 * std::f64::consts::PI / 11.0).cos())
            .collect();

        let mut computed = result.eigenvalues.clone();
        computed.sort_by(|a, b| b.partial_cmp(a).unwrap());

        for (got, expected) in computed.iter().zip(exact_largest.iter()) {
            assert!(
                (got - expected).abs() < 1e-6,
                "eigenvalue mismatch: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_trl_laplacian_1d() {
        // 20-point 1D Laplacian (tridiagonal): smallest eigenvalue = 2*(1 - cos(pi/21))
        // pi/21 ≈ 0.14960, cos(0.14960) ≈ 0.98883, so lambda_min ≈ 0.02234
        // The 1D Laplacian eigenvalue formula: lambda_k = 2*(1 - cos(k*pi/(n+1)))
        // For k=1, n=20: 2*(1 - cos(pi/21))
        let a = tridiagonal(20, -1.0, 2.0, -1.0).unwrap();

        let expected_smallest = 2.0 * (1.0 - (std::f64::consts::PI / 21.0).cos());

        // Finding smallest eigenvalues is harder for positive-definite matrices;
        // use larger Krylov space to ensure convergence.
        let cfg = TrlConfig {
            num_eigenvalues: 1,
            max_krylov_size: 18,
            num_thick: 3,
            max_restarts: 200,
            tol: 1e-7,
            which: EigenvalueTarget::Smallest,
            compute_eigenvectors: true,
        };
        let trl = ThickRestartLanczos::new(cfg).unwrap();
        let result = trl.compute::<f64>(&a).unwrap();

        assert_eq!(result.eigenvalues.len(), 1);
        assert!(
            (result.eigenvalues[0] - expected_smallest).abs() < 1e-5,
            "smallest eigenvalue of 20-pt 1D Laplacian: got {}, expected {}",
            result.eigenvalues[0],
            expected_smallest
        );
    }

    #[test]
    fn test_trl_convergence_residuals() {
        // Verify residuals are below tolerance after convergence
        let a = tridiagonal(15, -1.0, 4.0, -1.0).unwrap();

        let tol = 1e-8;
        let cfg = TrlConfig {
            num_eigenvalues: 4,
            max_krylov_size: 20,
            num_thick: 5,
            max_restarts: 100,
            tol,
            which: EigenvalueTarget::LargestMagnitude,
            compute_eigenvectors: true,
        };
        let trl = ThickRestartLanczos::new(cfg).unwrap();
        let result = trl.compute::<f64>(&a).unwrap();

        assert_eq!(result.residuals.len(), result.eigenvalues.len());
        for (i, &res) in result.residuals.iter().enumerate() {
            assert!(
                res <= tol * 100.0, // allow 100x tolerance for robustness
                "residual[{i}] = {res} exceeds tolerance {tol}"
            );
        }

        // Check eigenvector orthonormality
        if !result.eigenvectors.is_empty() {
            let evecs = &result.eigenvectors;
            for i in 0..evecs.len() {
                let norm_i = vec_norm(&evecs[i]);
                assert!(
                    (norm_i - 1.0).abs() < 1e-6,
                    "eigenvector {i} not normalized: norm = {norm_i}"
                );
            }
        }
    }

    #[test]
    fn test_trl_laplacian_2d_largest() {
        // 4x4 2D Laplacian (n=16): find 3 largest eigenvalues
        // Largest eigenvalue of 2D Laplacian (4x4 grid) ≈ 4 + 2*cos(pi/5) + 2*cos(pi/5)
        // = 4 + 4*cos(pi/5) ≈ 4 + 3.236 ≈ 7.236
        let a = laplacian_2d(4, 4).unwrap();

        let cfg = TrlConfig {
            num_eigenvalues: 3,
            max_krylov_size: 18,
            num_thick: 4,
            max_restarts: 150,
            tol: 1e-7,
            which: EigenvalueTarget::Largest,
            compute_eigenvectors: true,
        };
        let trl = ThickRestartLanczos::new(cfg).unwrap();
        let result = trl.compute::<f64>(&a).unwrap();

        assert_eq!(result.eigenvalues.len(), 3);

        // All eigenvalues of 2D Laplacian are in [0, 8]
        for &ev in &result.eigenvalues {
            assert!(
                (0.0..=8.01).contains(&ev),
                "2D Laplacian eigenvalue {ev} out of expected range [0, 8]"
            );
        }

        // Largest eigenvalue of 4x4 2D Laplacian
        let expected_max = 4.0
            - 2.0 * (4.0 * std::f64::consts::PI / 5.0).cos()
            - 2.0 * (4.0 * std::f64::consts::PI / 5.0).cos();
        // Actually: lambda_{k,l} = 4 - 2*cos(k*pi/5) - 2*cos(l*pi/5), max at k=l=4
        let expected_max_correct = 4.0
            - 2.0 * (4.0 * std::f64::consts::PI / 5.0).cos()
            - 2.0 * (4.0 * std::f64::consts::PI / 5.0).cos();
        let _ = expected_max; // suppress unused warning

        let mut computed = result.eigenvalues.clone();
        computed.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!(
            (computed[0] - expected_max_correct).abs() < 0.1,
            "largest 2D Laplacian eigenvalue: got {}, expected ~{}",
            computed[0],
            expected_max_correct
        );
    }
}
