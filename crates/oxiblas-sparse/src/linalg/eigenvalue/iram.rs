//! Implicitly Restarted Arnoldi Method (IRAM) for sparse eigenvalue problems.
//!
//! IRAM is a memory-efficient eigenvalue algorithm that computes a few eigenvalues
//! of large sparse matrices. It maintains a Krylov subspace of bounded size through
//! implicit restarts using shifted QR iterations.
//!
//! # Algorithm Overview
//!
//! 1. Build initial Arnoldi factorization: A*V_m = V_m*H_m + f_m*e_m^T
//! 2. Compute Ritz values (eigenvalues of H_m)
//! 3. Select p = m - k unwanted Ritz values as shifts
//! 4. Apply p implicit QR shifts to compress factorization to dimension k
//! 5. Continue Arnoldi from dimension k back to m
//! 6. Repeat until convergence
//!
//! For symmetric matrices, IRAM reduces to Implicitly Restarted Lanczos (IRL),
//! where H is tridiagonal and the algorithm is more efficient.

use crate::csr::CsrMatrix;
use crate::ops::spmv;
use num_traits::FromPrimitive;
use oxiblas_core::scalar::{Field, Real, Scalar};

use super::error::{EigenvalueError, WhichEigenvalues};
use super::utils::{dot, givens_rotation, norm};

// =============================================================================
// IRAMConfig
// =============================================================================

/// Configuration for IRAM (Implicitly Restarted Arnoldi Method).
#[derive(Debug, Clone)]
pub struct IRAMConfig<T> {
    /// Number of eigenvalues to compute.
    pub num_eigenvalues: usize,
    /// Which eigenvalues to compute.
    pub which: WhichEigenvalues,
    /// Maximum number of outer restart iterations.
    pub max_iterations: usize,
    /// Convergence tolerance for residual norm.
    pub tolerance: T,
    /// Whether to compute eigenvectors.
    pub compute_eigenvectors: bool,
    /// Size of Krylov subspace (ncv - number of Arnoldi vectors).
    /// Must be > num_eigenvalues. Typically 2*num_eigenvalues or num_eigenvalues + 10.
    pub krylov_dimension: usize,
    /// Whether the matrix is symmetric (uses Lanczos instead of Arnoldi).
    pub symmetric: bool,
}

impl Default for IRAMConfig<f64> {
    fn default() -> Self {
        Self {
            num_eigenvalues: 6,
            which: WhichEigenvalues::LargestMagnitude,
            max_iterations: 300,
            tolerance: 1e-8,
            compute_eigenvectors: true,
            krylov_dimension: 20,
            symmetric: false,
        }
    }
}

impl Default for IRAMConfig<f32> {
    fn default() -> Self {
        Self {
            num_eigenvalues: 6,
            which: WhichEigenvalues::LargestMagnitude,
            max_iterations: 300,
            tolerance: 1e-6,
            compute_eigenvectors: true,
            krylov_dimension: 20,
            symmetric: false,
        }
    }
}

// =============================================================================
// IRAMResult
// =============================================================================

/// Result of IRAM eigenvalue computation.
#[derive(Debug, Clone)]
pub struct IRAMResult<T> {
    /// Real parts of computed eigenvalues.
    pub eigenvalues_real: Vec<T>,
    /// Imaginary parts of computed eigenvalues.
    pub eigenvalues_imag: Vec<T>,
    /// Eigenvectors (if requested), stored as column vectors.
    pub eigenvectors: Option<Vec<Vec<T>>>,
    /// Number of outer iterations (restarts) performed.
    pub iterations: usize,
    /// Residual norms for each converged eigenpair.
    pub residual_norms: Vec<T>,
    /// Whether all requested eigenvalues converged.
    pub converged: bool,
    /// Number of converged eigenvalues.
    pub num_converged: usize,
}

// =============================================================================
// IRAM Solver
// =============================================================================

/// Implicitly Restarted Arnoldi Method (IRAM).
///
/// IRAM is a memory-efficient eigenvalue algorithm that computes a few eigenvalues
/// of large sparse matrices. It maintains a Krylov subspace of bounded size through
/// implicit restarts using shifted QR iterations.
///
/// # Algorithm Overview
///
/// 1. Build initial Arnoldi factorization: A*V_m = V_m*H_m + f_m*e_m^T
/// 2. Compute Ritz values (eigenvalues of H_m)
/// 3. Select p = m - k unwanted Ritz values as shifts
/// 4. Apply p implicit QR shifts to compress factorization to dimension k
/// 5. Continue Arnoldi from dimension k back to m
/// 6. Repeat until convergence
///
/// For symmetric matrices, IRAM reduces to Implicitly Restarted Lanczos (IRL),
/// where H is tridiagonal and the algorithm is more efficient.
///
/// # Example
///
/// ```ignore
/// use oxiblas_sparse::csr::CsrMatrix;
/// use oxiblas_sparse::linalg::eigenvalue::{IRAM, IRAMConfig, WhichEigenvalues};
///
/// // Create a large sparse matrix
/// let a = CsrMatrix::<f64>::eye(1000);
///
/// let config = IRAMConfig {
///     num_eigenvalues: 10,
///     which: WhichEigenvalues::LargestMagnitude,
///     krylov_dimension: 30,  // ncv = 30 > nev = 10
///     symmetric: true,  // More efficient for symmetric matrices
///     ..Default::default()
/// };
///
/// let iram = IRAM::new(config);
/// let result = iram.compute(&a, None).unwrap();
/// println!("Converged: {}, eigenvalues: {:?}", result.converged, result.eigenvalues_real);
/// ```
pub struct IRAM<T> {
    config: IRAMConfig<T>,
}

impl<T: Scalar<Real = T> + Clone + Field + Real + FromPrimitive> IRAM<T> {
    /// Create a new IRAM solver with the given configuration.
    pub fn new(config: IRAMConfig<T>) -> Self {
        Self { config }
    }

    /// Compute eigenvalues (and optionally eigenvectors) using IRAM.
    ///
    /// # Arguments
    ///
    /// * `a` - Square sparse matrix in CSR format
    /// * `initial_vector` - Optional starting vector
    ///
    /// # Returns
    ///
    /// Computed eigenvalues and eigenvectors.
    pub fn compute(
        &self,
        a: &CsrMatrix<T>,
        initial_vector: Option<&[T]>,
    ) -> Result<IRAMResult<T>, EigenvalueError> {
        let n = a.nrows();

        if a.ncols() != n {
            return Err(EigenvalueError::NotSquare {
                nrows: n,
                ncols: a.ncols(),
            });
        }

        let nev = self.config.num_eigenvalues;
        let ncv = self.config.krylov_dimension.max(nev + 2).min(n);
        let p = ncv - nev; // Number of shifts per restart

        if nev > n {
            return Err(EigenvalueError::TooManyEigenvalues {
                requested: nev,
                max_allowed: n,
            });
        }

        if ncv <= nev {
            return Err(EigenvalueError::ComputationError(
                "Krylov dimension must be larger than number of eigenvalues".to_string(),
            ));
        }

        if self.config.symmetric {
            self.compute_symmetric(a, n, nev, ncv, p, initial_vector)
        } else {
            self.compute_general(a, n, nev, ncv, p, initial_vector)
        }
    }

    /// IRAM for symmetric matrices (Implicitly Restarted Lanczos).
    fn compute_symmetric(
        &self,
        a: &CsrMatrix<T>,
        n: usize,
        nev: usize,
        ncv: usize,
        p: usize,
        initial_vector: Option<&[T]>,
    ) -> Result<IRAMResult<T>, EigenvalueError> {
        // Initialize starting vector
        let mut v = if let Some(v0) = initial_vector {
            if v0.len() != n {
                return Err(EigenvalueError::DimensionMismatch {
                    expected: n,
                    actual: v0.len(),
                });
            }
            v0.to_vec()
        } else {
            let n_t = T::from_usize(n).unwrap_or_else(T::one);
            let scale = T::one() / Real::sqrt(n_t);
            vec![scale; n]
        };

        // Normalize initial vector
        let v_norm = norm(&v);
        if v_norm <= <T as Scalar>::epsilon() {
            return Err(EigenvalueError::Breakdown {
                iteration: 0,
                description: "Initial vector is zero".to_string(),
            });
        }
        for vi in &mut v {
            *vi = vi.clone() / v_norm.clone();
        }

        // Storage for Lanczos vectors V (n x ncv)
        let mut lanczos_vectors: Vec<Vec<T>> = Vec::with_capacity(ncv);
        lanczos_vectors.push(v.clone());

        // Tridiagonal matrix elements
        let mut alpha: Vec<T> = Vec::with_capacity(ncv); // Diagonal
        let mut beta: Vec<T> = Vec::with_capacity(ncv); // Off-diagonal

        // Residual vector
        let mut f = vec![T::zero(); n];
        let mut f_norm = T::zero();

        // Working vectors
        let mut v_prev = vec![T::zero(); n];
        let mut w = vec![T::zero(); n];

        let tol_breakdown = <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or_else(T::one);

        // Build initial Lanczos factorization to ncv vectors
        for j in 0..ncv {
            // w = A * v_j
            spmv(T::one(), a, &lanczos_vectors[j], T::zero(), &mut w);

            // alpha[j] = v_j^T * w
            let alpha_j = dot(&lanczos_vectors[j], &w);
            alpha.push(alpha_j.clone());

            // f = w - alpha[j] * v_j - beta[j-1] * v_{j-1}
            for i in 0..n {
                f[i] = w[i].clone() - alpha_j.clone() * lanczos_vectors[j][i].clone();
            }
            if j > 0 {
                let beta_prev = beta[j - 1].clone();
                for i in 0..n {
                    f[i] = f[i].clone() - beta_prev.clone() * v_prev[i].clone();
                }
            }

            // Full reorthogonalization
            for qj in &lanczos_vectors {
                let h = dot(qj, &f);
                for i in 0..n {
                    f[i] = f[i].clone() - h.clone() * qj[i].clone();
                }
            }

            // beta[j] = ||f||
            f_norm = norm(&f);

            if f_norm <= tol_breakdown {
                // Invariant subspace found - use what we have
                break;
            }

            if j + 1 < ncv {
                beta.push(f_norm.clone());
                v_prev.clone_from_slice(&lanczos_vectors[j]);
                let mut v_next = vec![T::zero(); n];
                for i in 0..n {
                    v_next[i] = f[i].clone() / f_norm.clone();
                }
                lanczos_vectors.push(v_next);
            }
        }

        let mut converged_count = 0;
        let mut residual_norms = vec![T::zero(); nev];
        let mut final_eigenvalues = vec![T::zero(); nev];
        let mut converged_flags = vec![false; nev];

        // Main IRAM iteration loop
        for iter in 0..self.config.max_iterations {
            let current_dim = alpha.len();
            if current_dim < 2 {
                break;
            }

            // Solve tridiagonal eigenvalue problem to get Ritz values
            let (ritz_values, ritz_vectors) = self.solve_symmetric_tridiagonal(&alpha, &beta)?;

            // Select wanted vs unwanted Ritz values
            let (wanted_indices, unwanted_values) =
                self.select_shifts_symmetric(&ritz_values, nev, p);

            // Check convergence
            converged_count = 0;
            for (idx, &wi) in wanted_indices.iter().enumerate() {
                if wi < ritz_vectors.len() && current_dim > 0 && !beta.is_empty() {
                    let s_last = if current_dim - 1 < ritz_vectors[wi].len() {
                        ritz_vectors[wi][current_dim - 1].clone()
                    } else {
                        T::zero()
                    };
                    let beta_last = beta.last().cloned().unwrap_or_else(T::zero);
                    let residual = Scalar::abs(f_norm.clone() * s_last);
                    residual_norms[idx] = residual.clone();

                    if residual
                        <= self.config.tolerance
                            * Scalar::abs(ritz_values[wi].clone()).max(beta_last)
                    {
                        converged_flags[idx] = true;
                        converged_count += 1;
                    }

                    final_eigenvalues[idx] = ritz_values[wi].clone();
                }
            }

            // Check if all wanted eigenvalues converged
            if converged_count >= nev {
                // Compute eigenvectors if requested
                let eigenvectors = if self.config.compute_eigenvectors {
                    Some(self.compute_eigenvectors_symmetric(
                        &lanczos_vectors,
                        &ritz_vectors,
                        &wanted_indices,
                        n,
                    ))
                } else {
                    None
                };

                return Ok(IRAMResult {
                    eigenvalues_real: final_eigenvalues,
                    eigenvalues_imag: vec![T::zero(); nev],
                    eigenvectors,
                    iterations: iter + 1,
                    residual_norms,
                    converged: true,
                    num_converged: converged_count,
                });
            }

            // Apply implicit QR shifts to compress from ncv to nev
            let shifts = unwanted_values;
            self.apply_implicit_qr_shifts_symmetric(
                &mut alpha,
                &mut beta,
                &mut lanczos_vectors,
                &mut f,
                &shifts,
                nev,
            );

            // Continue Lanczos from nev back to ncv
            f_norm = norm(&f);
            if f_norm <= tol_breakdown {
                break;
            }

            // Normalize residual and continue
            let mut v_next = vec![T::zero(); n];
            for i in 0..n {
                v_next[i] = f[i].clone() / f_norm.clone();
            }

            if lanczos_vectors.len() > nev {
                lanczos_vectors.truncate(nev);
            }
            lanczos_vectors.push(v_next.clone());
            beta.truncate(nev.saturating_sub(1));
            if alpha.len() > nev {
                alpha.truncate(nev);
            }
            beta.push(f_norm.clone());

            // Continue Lanczos iteration to refill to ncv
            v_prev = if lanczos_vectors.len() >= 2 {
                lanczos_vectors[lanczos_vectors.len() - 2].clone()
            } else {
                vec![T::zero(); n]
            };

            while lanczos_vectors.len() < ncv {
                let j = lanczos_vectors.len() - 1;
                let current_v = lanczos_vectors[j].clone();

                // w = A * v_j
                spmv(T::one(), a, &current_v, T::zero(), &mut w);

                // alpha[j] = v_j^T * w
                let alpha_j = dot(&current_v, &w);
                alpha.push(alpha_j.clone());

                // f = w - alpha[j] * v_j - beta[j-1] * v_{j-1}
                for i in 0..n {
                    f[i] = w[i].clone() - alpha_j.clone() * current_v[i].clone();
                }
                if !beta.is_empty() {
                    let beta_prev = beta.last().cloned().unwrap_or_else(T::zero);
                    for i in 0..n {
                        f[i] = f[i].clone() - beta_prev.clone() * v_prev[i].clone();
                    }
                }

                // Full reorthogonalization
                for qj in &lanczos_vectors {
                    let h = dot(qj, &f);
                    for i in 0..n {
                        f[i] = f[i].clone() - h.clone() * qj[i].clone();
                    }
                }

                f_norm = norm(&f);

                if f_norm <= tol_breakdown {
                    break;
                }

                beta.push(f_norm.clone());
                v_prev.clone_from_slice(&current_v);
                let mut v_new = vec![T::zero(); n];
                for i in 0..n {
                    v_new[i] = f[i].clone() / f_norm.clone();
                }
                lanczos_vectors.push(v_new);
            }
        }

        // Return best results found
        let eigenvectors = if self.config.compute_eigenvectors {
            let (ritz_values, ritz_vectors) = self.solve_symmetric_tridiagonal(&alpha, &beta)?;
            let (wanted_indices, _) = self.select_shifts_symmetric(&ritz_values, nev, p);
            Some(self.compute_eigenvectors_symmetric(
                &lanczos_vectors,
                &ritz_vectors,
                &wanted_indices,
                n,
            ))
        } else {
            None
        };

        Ok(IRAMResult {
            eigenvalues_real: final_eigenvalues,
            eigenvalues_imag: vec![T::zero(); nev],
            eigenvectors,
            iterations: self.config.max_iterations,
            residual_norms,
            converged: converged_count >= nev,
            num_converged: converged_count,
        })
    }

    /// IRAM for general (non-symmetric) matrices.
    fn compute_general(
        &self,
        a: &CsrMatrix<T>,
        n: usize,
        nev: usize,
        ncv: usize,
        p: usize,
        initial_vector: Option<&[T]>,
    ) -> Result<IRAMResult<T>, EigenvalueError> {
        // Initialize starting vector
        let mut v = if let Some(v0) = initial_vector {
            if v0.len() != n {
                return Err(EigenvalueError::DimensionMismatch {
                    expected: n,
                    actual: v0.len(),
                });
            }
            v0.to_vec()
        } else {
            let n_t = T::from_usize(n).unwrap_or_else(T::one);
            let scale = T::one() / Real::sqrt(n_t);
            vec![scale; n]
        };

        // Normalize initial vector
        let v_norm = norm(&v);
        if v_norm <= <T as Scalar>::epsilon() {
            return Err(EigenvalueError::Breakdown {
                iteration: 0,
                description: "Initial vector is zero".to_string(),
            });
        }
        for vi in &mut v {
            *vi = vi.clone() / v_norm.clone();
        }

        // Storage for Arnoldi vectors V (n x ncv)
        let mut arnoldi_vectors: Vec<Vec<T>> = Vec::with_capacity(ncv);
        arnoldi_vectors.push(v.clone());

        // Upper Hessenberg matrix H (ncv x ncv)
        let mut h: Vec<Vec<T>> = vec![vec![T::zero(); ncv]; ncv];

        // Residual vector
        let mut f = vec![T::zero(); n];
        let mut f_norm;

        // Working vector
        let mut w = vec![T::zero(); n];

        let tol_breakdown = <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or_else(T::one);

        // Build initial Arnoldi factorization to ncv vectors
        for j in 0..ncv {
            // w = A * v_j
            spmv(T::one(), a, &arnoldi_vectors[j], T::zero(), &mut w);

            // Modified Gram-Schmidt orthogonalization
            for i in 0..=j {
                h[i][j] = dot(&arnoldi_vectors[i], &w);
                for idx in 0..n {
                    w[idx] = w[idx].clone() - h[i][j].clone() * arnoldi_vectors[i][idx].clone();
                }
            }

            f_norm = norm(&w);

            if f_norm <= tol_breakdown {
                break;
            }

            if j + 1 < ncv {
                h[j + 1][j] = f_norm.clone();
                let mut v_next = vec![T::zero(); n];
                for idx in 0..n {
                    v_next[idx] = w[idx].clone() / f_norm.clone();
                }
                arnoldi_vectors.push(v_next);
            } else {
                // Save residual for last iteration
                f.clone_from_slice(&w);
            }
        }

        let mut converged_count = 0;
        let mut residual_norms = vec![T::zero(); nev];
        let mut final_eigenvalues_real = vec![T::zero(); nev];
        let mut final_eigenvalues_imag = vec![T::zero(); nev];

        // Main IRAM iteration loop
        for iter in 0..self.config.max_iterations {
            let current_dim = arnoldi_vectors.len().min(ncv);
            if current_dim < 2 {
                break;
            }

            // Solve Hessenberg eigenvalue problem to get Ritz values
            let (ritz_real, ritz_imag) = self.solve_hessenberg_eigenvalues(&h, current_dim)?;

            // Select wanted vs unwanted Ritz values (get shifts)
            let (wanted_indices, unwanted_real, unwanted_imag) =
                self.select_shifts_general(&ritz_real, &ritz_imag, nev, p);

            // Check convergence
            converged_count = 0;
            for (idx, &wi) in wanted_indices.iter().enumerate() {
                if idx >= nev || wi >= current_dim {
                    continue;
                }

                // Residual estimate: |H[m,m-1] * y_m| where y is Ritz vector
                // For simplicity, use a conservative estimate
                let h_sub = if current_dim > 0 && current_dim <= ncv {
                    h[current_dim.min(ncv - 1)][current_dim.saturating_sub(1)].clone()
                } else {
                    T::zero()
                };
                let residual = Scalar::abs(h_sub);
                residual_norms[idx] = residual.clone();

                let ritz_mag = Real::sqrt(
                    ritz_real[wi].clone() * ritz_real[wi].clone()
                        + ritz_imag[wi].clone() * ritz_imag[wi].clone(),
                );

                if residual <= self.config.tolerance * ritz_mag.max(T::one()) {
                    converged_count += 1;
                }

                final_eigenvalues_real[idx] = ritz_real[wi].clone();
                final_eigenvalues_imag[idx] = ritz_imag[wi].clone();
            }

            // Check if all wanted eigenvalues converged
            if converged_count >= nev {
                let eigenvectors = if self.config.compute_eigenvectors {
                    Some(self.compute_eigenvectors_general(
                        &arnoldi_vectors,
                        &h,
                        &wanted_indices,
                        n,
                        current_dim,
                    ))
                } else {
                    None
                };

                return Ok(IRAMResult {
                    eigenvalues_real: final_eigenvalues_real,
                    eigenvalues_imag: final_eigenvalues_imag,
                    eigenvectors,
                    iterations: iter + 1,
                    residual_norms,
                    converged: true,
                    num_converged: converged_count,
                });
            }

            // Apply implicit QR shifts (real shifts only for simplicity)
            self.apply_implicit_qr_shifts_general(
                &mut h,
                &mut arnoldi_vectors,
                &mut f,
                &unwanted_real,
                &unwanted_imag,
                nev,
                ncv,
            );

            // Continue Arnoldi from nev back to ncv
            f_norm = norm(&f);
            if f_norm <= tol_breakdown {
                break;
            }

            // Truncate and continue
            arnoldi_vectors.truncate(nev);

            // Normalize residual
            let mut v_next = vec![T::zero(); n];
            for i in 0..n {
                v_next[i] = f[i].clone() / f_norm.clone();
            }
            arnoldi_vectors.push(v_next);
            h[nev][nev - 1] = f_norm.clone();

            // Continue Arnoldi to refill
            while arnoldi_vectors.len() < ncv {
                let j = arnoldi_vectors.len() - 1;

                // w = A * v_j
                spmv(T::one(), a, &arnoldi_vectors[j], T::zero(), &mut w);

                // Modified Gram-Schmidt
                for i in 0..=j {
                    if i < arnoldi_vectors.len() {
                        h[i][j] = dot(&arnoldi_vectors[i], &w);
                        for idx in 0..n {
                            w[idx] =
                                w[idx].clone() - h[i][j].clone() * arnoldi_vectors[i][idx].clone();
                        }
                    }
                }

                f_norm = norm(&w);

                if f_norm <= tol_breakdown {
                    break;
                }

                if j + 1 < ncv {
                    h[j + 1][j] = f_norm.clone();
                    let mut v_new = vec![T::zero(); n];
                    for idx in 0..n {
                        v_new[idx] = w[idx].clone() / f_norm.clone();
                    }
                    arnoldi_vectors.push(v_new);
                } else {
                    f.clone_from_slice(&w);
                }
            }
        }

        // Return best results found
        let eigenvectors = if self.config.compute_eigenvectors {
            let current_dim = arnoldi_vectors.len().min(ncv);
            let (ritz_real, ritz_imag) = self.solve_hessenberg_eigenvalues(&h, current_dim)?;
            let (wanted_indices, _, _) = self.select_shifts_general(&ritz_real, &ritz_imag, nev, p);
            Some(self.compute_eigenvectors_general(
                &arnoldi_vectors,
                &h,
                &wanted_indices,
                n,
                current_dim,
            ))
        } else {
            None
        };

        Ok(IRAMResult {
            eigenvalues_real: final_eigenvalues_real,
            eigenvalues_imag: final_eigenvalues_imag,
            eigenvectors,
            iterations: self.config.max_iterations,
            residual_norms,
            converged: converged_count >= nev,
            num_converged: converged_count,
        })
    }

    /// Solve symmetric tridiagonal eigenvalue problem.
    fn solve_symmetric_tridiagonal(
        &self,
        alpha: &[T],
        beta: &[T],
    ) -> Result<(Vec<T>, Vec<Vec<T>>), EigenvalueError> {
        let n = alpha.len();
        if n == 0 {
            return Ok((vec![], vec![]));
        }

        let mut d = alpha.to_vec();
        let mut e = if beta.is_empty() {
            vec![T::zero(); n.saturating_sub(1)]
        } else {
            let mut e = beta.to_vec();
            e.truncate(n.saturating_sub(1));
            while e.len() < n.saturating_sub(1) {
                e.push(T::zero());
            }
            e
        };

        // Initialize eigenvector matrix as identity
        let mut z: Vec<Vec<T>> = (0..n)
            .map(|i| {
                let mut row = vec![T::zero(); n];
                row[i] = T::one();
                row
            })
            .collect();

        let max_qr_iter = 30 * n;
        let tol = <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or_else(T::one);
        let two = T::from_f64(2.0).unwrap_or_else(T::one);

        for _iter in 0..max_qr_iter {
            let mut l = 0;
            for i in (0..n.saturating_sub(1)).rev() {
                if Scalar::abs(e[i].clone())
                    <= tol.clone() * (Scalar::abs(d[i].clone()) + Scalar::abs(d[i + 1].clone()))
                {
                    e[i] = T::zero();
                } else {
                    l = i + 1;
                    break;
                }
            }

            if l == 0 {
                break;
            }

            let mut m = l;
            for i in (0..l).rev() {
                if Scalar::abs(e[i].clone())
                    <= tol.clone() * (Scalar::abs(d[i].clone()) + Scalar::abs(d[i + 1].clone()))
                {
                    e[i] = T::zero();
                    m = i + 1;
                    break;
                }
                if i == 0 {
                    m = 0;
                }
            }

            // Wilkinson shift
            let p = (d[l - 1].clone() - d[l].clone()) / two.clone();
            let r = Real::sqrt(p.clone() * p.clone() + e[l - 1].clone() * e[l - 1].clone());
            let shift = if p >= T::zero() {
                d[l].clone() - e[l - 1].clone() * e[l - 1].clone() / (p.clone() + r)
            } else {
                d[l].clone() - e[l - 1].clone() * e[l - 1].clone() / (p.clone() - r)
            };

            // Implicit QR step
            let mut g = d[m].clone() - shift.clone();
            let mut s = T::one();
            let mut c = T::one();
            let mut p_val = T::zero();

            for i in m..l {
                let f_val = s.clone() * e[i].clone();
                let b = c.clone() * e[i].clone();

                if Scalar::abs(f_val.clone()) >= Scalar::abs(g.clone()) {
                    c = g.clone() / f_val.clone();
                    let r = Real::sqrt(c.clone() * c.clone() + T::one());
                    if i > m {
                        e[i - 1] = f_val.clone() * r.clone();
                    }
                    s = T::one() / r.clone();
                    c = c.clone() * s.clone();
                } else {
                    s = f_val.clone() / g.clone();
                    let r = Real::sqrt(s.clone() * s.clone() + T::one());
                    if i > m {
                        e[i - 1] = g.clone() * r.clone();
                    }
                    c = T::one() / r.clone();
                    s = s.clone() * c.clone();
                }

                g = d[i].clone() - p_val.clone();
                let r = (d[i + 1].clone() - g.clone()) * s.clone()
                    + two.clone() * c.clone() * b.clone();
                p_val = s.clone() * r.clone();
                d[i] = g.clone() + p_val.clone();
                g = c.clone() * r.clone() - b.clone();

                // Update eigenvectors
                for k in 0..n {
                    let f_val = z[k][i + 1].clone();
                    z[k][i + 1] = s.clone() * z[k][i].clone() + c.clone() * f_val.clone();
                    z[k][i] = c.clone() * z[k][i].clone() - s.clone() * f_val;
                }
            }

            d[l] = d[l].clone() - p_val.clone();
            e[l - 1] = g;
        }

        let eigenvalues = d;
        let eigenvectors: Vec<Vec<T>> = (0..n)
            .map(|i| z.iter().map(|row| row[i].clone()).collect())
            .collect();

        Ok((eigenvalues, eigenvectors))
    }

    /// Solve upper Hessenberg eigenvalue problem.
    fn solve_hessenberg_eigenvalues(
        &self,
        h: &[Vec<T>],
        n: usize,
    ) -> Result<(Vec<T>, Vec<T>), EigenvalueError> {
        if n == 0 {
            return Ok((vec![], vec![]));
        }

        // Copy Hessenberg matrix
        let mut a: Vec<Vec<T>> = (0..n)
            .map(|i| {
                if i < h.len() {
                    h[i][..n.min(h[i].len())].to_vec()
                } else {
                    vec![T::zero(); n]
                }
            })
            .collect();

        // Pad if needed
        for row in &mut a {
            while row.len() < n {
                row.push(T::zero());
            }
        }

        let mut eigenvalues_real = vec![T::zero(); n];
        let mut eigenvalues_imag = vec![T::zero(); n];

        let tol = <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or_else(T::one);
        let two = T::from_f64(2.0).unwrap_or_else(T::one);
        let four = T::from_f64(4.0).unwrap_or_else(T::one);
        let max_iter = 30 * n;

        let mut p = n;

        for _iter in 0..max_iter {
            if p <= 1 {
                if p == 1 {
                    eigenvalues_real[0] = a[0][0].clone();
                }
                break;
            }

            let l = p - 1;
            if Scalar::abs(a[l][l - 1].clone())
                <= tol.clone()
                    * (Scalar::abs(a[l - 1][l - 1].clone()) + Scalar::abs(a[l][l].clone()))
            {
                eigenvalues_real[l] = a[l][l].clone();
                p = l;
                continue;
            }

            if p >= 2
                && l >= 2
                && Scalar::abs(a[l - 1][l - 2].clone())
                    <= tol.clone()
                        * (Scalar::abs(a[l - 2][l - 2].clone())
                            + Scalar::abs(a[l - 1][l - 1].clone()))
            {
                let a11 = a[l - 1][l - 1].clone();
                let a12 = a[l - 1][l].clone();
                let a21 = a[l][l - 1].clone();
                let a22 = a[l][l].clone();

                let trace = a11.clone() + a22.clone();
                let det = a11 * a22 - a12 * a21;
                let disc = trace.clone() * trace.clone() / four.clone() - det;

                if disc >= T::zero() {
                    let sqrt_disc = Real::sqrt(disc);
                    eigenvalues_real[l - 1] = trace.clone() / two.clone() + sqrt_disc.clone();
                    eigenvalues_real[l] = trace / two.clone() - sqrt_disc;
                } else {
                    let sqrt_disc = Real::sqrt(T::zero() - disc);
                    eigenvalues_real[l - 1] = trace.clone() / two.clone();
                    eigenvalues_real[l] = trace / two.clone();
                    eigenvalues_imag[l - 1] = sqrt_disc.clone();
                    eigenvalues_imag[l] = T::zero() - sqrt_disc;
                }

                p = l - 1;
                continue;
            }

            // Wilkinson shift
            let shift = self.compute_wilkinson_shift_general(&a, p);

            // Apply shifted QR step
            self.qr_step_hessenberg(&mut a, p, shift);
        }

        Ok((eigenvalues_real, eigenvalues_imag))
    }

    /// Compute Wilkinson shift for Hessenberg QR iteration.
    fn compute_wilkinson_shift_general(&self, a: &[Vec<T>], p: usize) -> T {
        if p < 2 {
            return a[p - 1][p - 1].clone();
        }

        let two = T::from_f64(2.0).unwrap_or_else(T::one);
        let four = T::from_f64(4.0).unwrap_or_else(T::one);

        let n = p;
        let a11 = a[n - 2][n - 2].clone();
        let a12 = a[n - 2][n - 1].clone();
        let a21 = a[n - 1][n - 2].clone();
        let a22 = a[n - 1][n - 1].clone();

        let trace = a11.clone() + a22.clone();
        let det = a11 * a22 - a12 * a21;
        let disc = trace.clone() * trace.clone() / four.clone() - det;

        if disc >= T::zero() {
            let sqrt_disc = Real::sqrt(disc);
            let lambda1 = trace.clone() / two.clone() + sqrt_disc.clone();
            let lambda2 = trace / two.clone() - sqrt_disc;

            let corner = a[n - 1][n - 1].clone();
            if Scalar::abs(lambda1.clone() - corner.clone()) < Scalar::abs(lambda2.clone() - corner)
            {
                lambda1
            } else {
                lambda2
            }
        } else {
            trace / two
        }
    }

    /// Apply QR step with shift to upper Hessenberg matrix.
    fn qr_step_hessenberg(&self, a: &mut [Vec<T>], p: usize, shift: T) {
        // Apply shift
        for i in 0..p {
            a[i][i] = a[i][i].clone() - shift.clone();
        }

        // QR factorization using Givens rotations
        for i in 0..p - 1 {
            if Scalar::abs(a[i + 1][i].clone()) <= <T as Scalar>::epsilon() {
                continue;
            }

            let (c, s, r) = givens_rotation(a[i][i].clone(), a[i + 1][i].clone());

            a[i][i] = r;
            a[i + 1][i] = T::zero();

            for j in i + 1..p {
                let temp = c.clone() * a[i][j].clone() + s.clone() * a[i + 1][j].clone();
                a[i + 1][j] =
                    T::zero() - s.clone() * a[i][j].clone() + c.clone() * a[i + 1][j].clone();
                a[i][j] = temp;
            }

            let col_end = (i + 3).min(p);
            for j in 0..col_end {
                let temp = c.clone() * a[j][i].clone() + s.clone() * a[j][i + 1].clone();
                a[j][i + 1] =
                    T::zero() - s.clone() * a[j][i].clone() + c.clone() * a[j][i + 1].clone();
                a[j][i] = temp;
            }
        }

        // Remove shift
        for i in 0..p {
            a[i][i] = a[i][i].clone() + shift.clone();
        }
    }

    /// Select wanted eigenvalues and unwanted shifts for symmetric case.
    fn select_shifts_symmetric(
        &self,
        eigenvalues: &[T],
        nev: usize,
        p: usize,
    ) -> (Vec<usize>, Vec<T>) {
        if eigenvalues.is_empty() {
            return (vec![], vec![]);
        }

        let n = eigenvalues.len();
        let nev = nev.min(n);
        let p = p.min(n.saturating_sub(nev));

        let mut indexed: Vec<(usize, T)> = eigenvalues
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.clone()))
            .collect();

        // Sort based on which eigenvalues are wanted
        match self.config.which {
            WhichEigenvalues::LargestMagnitude => {
                indexed.sort_by(|a, b| {
                    Scalar::abs(b.1.clone())
                        .partial_cmp(&Scalar::abs(a.1.clone()))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            WhichEigenvalues::SmallestMagnitude => {
                indexed.sort_by(|a, b| {
                    Scalar::abs(a.1.clone())
                        .partial_cmp(&Scalar::abs(b.1.clone()))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            WhichEigenvalues::LargestAlgebraic => {
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            WhichEigenvalues::SmallestAlgebraic => {
                indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            WhichEigenvalues::NearTarget => {
                indexed.sort_by(|a, b| {
                    Scalar::abs(a.1.clone())
                        .partial_cmp(&Scalar::abs(b.1.clone()))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        let wanted_indices: Vec<usize> = indexed.iter().take(nev).map(|(i, _)| *i).collect();
        let unwanted_values: Vec<T> = indexed
            .iter()
            .skip(nev)
            .take(p)
            .map(|(_, v)| v.clone())
            .collect();

        (wanted_indices, unwanted_values)
    }

    /// Select wanted eigenvalues and unwanted shifts for general case.
    fn select_shifts_general(
        &self,
        real: &[T],
        imag: &[T],
        nev: usize,
        p: usize,
    ) -> (Vec<usize>, Vec<T>, Vec<T>) {
        let n = real.len();
        if n == 0 {
            return (vec![], vec![], vec![]);
        }

        let nev = nev.min(n);
        let p = p.min(n.saturating_sub(nev));

        // Compute magnitudes
        let mut indexed: Vec<(usize, T)> = real
            .iter()
            .zip(imag.iter())
            .enumerate()
            .map(|(i, (r, im))| {
                let mag = Real::sqrt(r.clone() * r.clone() + im.clone() * im.clone());
                (i, mag)
            })
            .collect();

        // Sort by magnitude based on selection criterion
        match self.config.which {
            WhichEigenvalues::LargestMagnitude => {
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            WhichEigenvalues::SmallestMagnitude => {
                indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            _ => {
                // For algebraic, sort by real part
                let mut real_indexed: Vec<(usize, T)> = real
                    .iter()
                    .enumerate()
                    .map(|(i, r)| (i, r.clone()))
                    .collect();
                match self.config.which {
                    WhichEigenvalues::LargestAlgebraic => {
                        real_indexed.sort_by(|a, b| {
                            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                    _ => {
                        real_indexed.sort_by(|a, b| {
                            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                }
                indexed = real_indexed;
            }
        }

        let wanted_indices: Vec<usize> = indexed.iter().take(nev).map(|(i, _)| *i).collect();
        let unwanted_real: Vec<T> = indexed
            .iter()
            .skip(nev)
            .take(p)
            .map(|(i, _)| real[*i].clone())
            .collect();
        let unwanted_imag: Vec<T> = indexed
            .iter()
            .skip(nev)
            .take(p)
            .map(|(i, _)| imag[*i].clone())
            .collect();

        (wanted_indices, unwanted_real, unwanted_imag)
    }

    /// Apply implicit QR shifts for symmetric tridiagonal case.
    fn apply_implicit_qr_shifts_symmetric(
        &self,
        alpha: &mut Vec<T>,
        beta: &mut Vec<T>,
        v: &mut Vec<Vec<T>>,
        f: &mut [T],
        shifts: &[T],
        nev: usize,
    ) {
        let n = v[0].len();
        let two = T::from_f64(2.0).unwrap_or_else(T::one);

        for shift in shifts {
            let m = alpha.len();
            if m < 2 {
                break;
            }

            // Apply bulge-chasing QR step with shift
            // For tridiagonal, this is more efficient than full Hessenberg QR

            // First, compute the initial Givens rotation
            let mut g = alpha[0].clone() - shift.clone();
            let mut s = T::one();
            let mut c = T::one();

            for i in 0..m - 1 {
                let i_beta = if i < beta.len() {
                    beta[i].clone()
                } else {
                    T::zero()
                };
                let f_val = s.clone() * i_beta.clone();
                let b = c.clone() * i_beta;

                // Givens rotation to chase bulge
                if Scalar::abs(f_val.clone()) >= Scalar::abs(g.clone()) {
                    c = g.clone() / f_val.clone();
                    let r = Real::sqrt(c.clone() * c.clone() + T::one());
                    if i > 0 && i - 1 < beta.len() {
                        beta[i - 1] = f_val.clone() * r.clone();
                    }
                    s = T::one() / r.clone();
                    c = c.clone() * s.clone();
                } else {
                    s = f_val.clone() / g.clone();
                    let r = Real::sqrt(s.clone() * s.clone() + T::one());
                    if i > 0 && i - 1 < beta.len() {
                        beta[i - 1] = g.clone() * r.clone();
                    }
                    c = T::one() / r.clone();
                    s = s.clone() * c.clone();
                }

                // Update tridiagonal elements
                let p_val = s.clone() * alpha[i].clone() - c.clone() * b.clone();
                let next_alpha = if i + 1 < alpha.len() {
                    alpha[i + 1].clone()
                } else {
                    T::zero()
                };
                let r = (next_alpha.clone() - alpha[i].clone()) * s.clone()
                    + two.clone() * c.clone() * b.clone();

                alpha[i] = alpha[i].clone() + p_val.clone();
                g = c.clone() * r.clone() - b.clone();

                if i + 1 < alpha.len() {
                    alpha[i + 1] = next_alpha - s.clone() * r.clone();
                }

                // Apply Givens rotation to Lanczos vectors
                for k in 0..n {
                    if i + 1 < v.len() {
                        let tmp = c.clone() * v[i][k].clone() + s.clone() * v[i + 1][k].clone();
                        v[i + 1][k] = T::zero() - s.clone() * v[i][k].clone()
                            + c.clone() * v[i + 1][k].clone();
                        v[i][k] = tmp;
                    }
                }
            }

            // Update beta[m-2] if it exists
            if m >= 2 && m - 2 < beta.len() {
                beta[m - 2] = g;
            }
        }

        // Apply rotation to residual vector
        if !shifts.is_empty() {
            let m = alpha.len();
            if m > 0 && m <= v.len() {
                // The residual is updated implicitly through the Lanczos vectors
                let beta_last = if !beta.is_empty() {
                    beta.last().cloned().unwrap_or_else(T::zero)
                } else {
                    T::zero()
                };
                let last_v = v.last().cloned().unwrap_or_else(|| vec![T::zero(); n]);
                for i in 0..n {
                    f[i] = beta_last.clone() * last_v[i].clone();
                }
            }
        }

        // Truncate to nev
        alpha.truncate(nev);
        if !beta.is_empty() {
            beta.truncate(nev.saturating_sub(1));
        }
    }

    /// Apply implicit QR shifts for general Hessenberg case.
    fn apply_implicit_qr_shifts_general(
        &self,
        h: &mut Vec<Vec<T>>,
        v: &mut Vec<Vec<T>>,
        f: &mut [T],
        shifts_real: &[T],
        _shifts_imag: &[T],
        nev: usize,
        ncv: usize,
    ) {
        let n = v[0].len();

        // Apply real shifts using bulge-chasing
        for shift in shifts_real {
            let m = v.len().min(ncv);
            if m < 2 {
                break;
            }

            // Apply shifted QR step
            // First, apply shift
            for i in 0..m {
                if i < h.len() && i < h[i].len() {
                    h[i][i] = h[i][i].clone() - shift.clone();
                }
            }

            // QR factorization with Givens rotations
            for i in 0..m - 1 {
                if i + 1 >= h.len() || i >= h[i + 1].len() {
                    continue;
                }

                if Scalar::abs(h[i + 1][i].clone()) <= <T as Scalar>::epsilon() {
                    continue;
                }

                let (c, s, r) = givens_rotation(h[i][i].clone(), h[i + 1][i].clone());

                h[i][i] = r;
                h[i + 1][i] = T::zero();

                // Apply to remaining columns
                for j in i + 1..m {
                    if j < h[i].len() && j < h[i + 1].len() {
                        let temp = c.clone() * h[i][j].clone() + s.clone() * h[i + 1][j].clone();
                        h[i + 1][j] = T::zero() - s.clone() * h[i][j].clone()
                            + c.clone() * h[i + 1][j].clone();
                        h[i][j] = temp;
                    }
                }

                // Apply to columns (RQ product)
                let col_end = (i + 3).min(m);
                for j in 0..col_end {
                    if j < h.len() && i < h[j].len() && i + 1 < h[j].len() {
                        let temp = c.clone() * h[j][i].clone() + s.clone() * h[j][i + 1].clone();
                        h[j][i + 1] = T::zero() - s.clone() * h[j][i].clone()
                            + c.clone() * h[j][i + 1].clone();
                        h[j][i] = temp;
                    }
                }

                // Apply to Arnoldi vectors
                for k in 0..n {
                    if i < v.len() && i + 1 < v.len() {
                        let temp = c.clone() * v[i][k].clone() + s.clone() * v[i + 1][k].clone();
                        v[i + 1][k] = T::zero() - s.clone() * v[i][k].clone()
                            + c.clone() * v[i + 1][k].clone();
                        v[i][k] = temp;
                    }
                }
            }

            // Remove shift
            for i in 0..m {
                if i < h.len() && i < h[i].len() {
                    h[i][i] = h[i][i].clone() + shift.clone();
                }
            }
        }

        // Update residual
        if nev < v.len() {
            let h_sub = if nev < h.len() && nev > 0 && nev - 1 < h[nev].len() {
                h[nev][nev - 1].clone()
            } else {
                T::zero()
            };
            let v_nev = if nev < v.len() {
                v[nev].clone()
            } else {
                vec![T::zero(); n]
            };
            for i in 0..n {
                f[i] = h_sub.clone() * v_nev[i].clone();
            }
        }
    }

    /// Compute eigenvectors for symmetric case.
    fn compute_eigenvectors_symmetric(
        &self,
        lanczos_vectors: &[Vec<T>],
        ritz_vectors: &[Vec<T>],
        wanted_indices: &[usize],
        n: usize,
    ) -> Vec<Vec<T>> {
        let mut eigenvectors = Vec::with_capacity(wanted_indices.len());

        for &idx in wanted_indices {
            if idx >= ritz_vectors.len() {
                continue;
            }

            let y = &ritz_vectors[idx];
            let mut x = vec![T::zero(); n];

            // Transform from Lanczos basis to original basis: x = V * y
            for (j, vj) in lanczos_vectors.iter().enumerate() {
                if j < y.len() {
                    for i in 0..n {
                        x[i] = x[i].clone() + y[j].clone() * vj[i].clone();
                    }
                }
            }

            // Normalize
            let x_norm = norm(&x);
            if x_norm > <T as Scalar>::epsilon() {
                for xi in &mut x {
                    *xi = xi.clone() / x_norm.clone();
                }
            }

            eigenvectors.push(x);
        }

        eigenvectors
    }

    /// Compute eigenvectors for general case.
    fn compute_eigenvectors_general(
        &self,
        arnoldi_vectors: &[Vec<T>],
        h: &[Vec<T>],
        wanted_indices: &[usize],
        n: usize,
        m: usize,
    ) -> Vec<Vec<T>> {
        // Compute Schur vectors from Hessenberg matrix
        // For simplicity, use power iteration to refine vectors
        let mut eigenvectors = Vec::with_capacity(wanted_indices.len());

        for &idx in wanted_indices {
            if idx >= m {
                continue;
            }

            // Start with unit vector in Hessenberg basis
            let mut y = vec![T::zero(); m];
            if idx < m {
                y[idx] = T::one();
            }

            // Power iteration in Hessenberg space to refine
            for _ in 0..5 {
                let mut y_new = vec![T::zero(); m];
                for i in 0..m {
                    for j in 0..m {
                        if i < h.len() && j < h[i].len() {
                            y_new[i] = y_new[i].clone() + h[i][j].clone() * y[j].clone();
                        }
                    }
                }
                let y_norm = norm(&y_new);
                if y_norm > <T as Scalar>::epsilon() {
                    for yi in &mut y_new {
                        *yi = yi.clone() / y_norm.clone();
                    }
                }
                y = y_new;
            }

            // Transform to original basis: x = V * y
            let mut x = vec![T::zero(); n];
            for (j, vj) in arnoldi_vectors.iter().enumerate() {
                if j < y.len() {
                    for i in 0..n {
                        x[i] = x[i].clone() + y[j].clone() * vj[i].clone();
                    }
                }
            }

            // Normalize
            let x_norm = norm(&x);
            if x_norm > <T as Scalar>::epsilon() {
                for xi in &mut x {
                    *xi = xi.clone() / x_norm.clone();
                }
            }

            eigenvectors.push(x);
        }

        eigenvectors
    }
}
