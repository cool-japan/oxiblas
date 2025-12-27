//! Lanczos Iteration for Sparse Symmetric Eigenvalue Problems.
//!
//! This module implements the Lanczos algorithm for computing eigenvalues and
//! eigenvectors of large sparse symmetric matrices. The algorithm builds an
//! orthonormal basis for the Krylov subspace and reduces the matrix to
//! tridiagonal form, making eigenvalue computation efficient.
//!
//! # Algorithm Overview
//!
//! The Lanczos algorithm computes the k largest or smallest eigenvalues (and
//! optionally eigenvectors) of a symmetric matrix A using the following approach:
//!
//! 1. Build an orthonormal basis Q for the Krylov subspace K_m(A, v) = span{v, Av, A^2v, ..., A^(m-1)v}
//! 2. Reduce A to tridiagonal form T = Q^T A Q
//! 3. Compute eigenvalues of T (Ritz values) which approximate eigenvalues of A
//! 4. Transform eigenvectors of T back to the original basis
//!
//! # Features
//!
//! - **Full reorthogonalization**: Optional full reorthogonalization for numerical stability
//! - **Configurable convergence**: Adjustable tolerance and iteration limits
//! - **Multiple selection criteria**: Largest/smallest magnitude or algebraic eigenvalues
//!
//! # Example
//!
//! ```ignore
//! use oxiblas_sparse::csr::CsrMatrix;
//! use oxiblas_sparse::linalg::eigenvalue::{Lanczos, LanczosConfig, WhichEigenvalues};
//!
//! // Create a sparse symmetric matrix
//! let a = CsrMatrix::<f64>::eye(100);
//!
//! let config = LanczosConfig {
//!     num_eigenvalues: 5,
//!     which: WhichEigenvalues::LargestMagnitude,
//!     ..Default::default()
//! };
//!
//! let lanczos = Lanczos::new(config);
//! let result = lanczos.compute(&a, None).unwrap();
//! println!("Eigenvalues: {:?}", result.eigenvalues);
//! ```

use crate::csr::CsrMatrix;
use crate::ops::spmv;
use num_traits::FromPrimitive;
use oxiblas_core::scalar::{Field, Real, Scalar};

use super::error::{EigenvalueError, WhichEigenvalues};
use super::utils::{dot, norm};

// =============================================================================
// Result and Configuration Types
// =============================================================================

/// Result of Lanczos eigenvalue computation.
#[derive(Debug, Clone)]
pub struct LanczosResult<T> {
    /// Computed eigenvalues (Ritz values).
    pub eigenvalues: Vec<T>,
    /// Eigenvectors (Ritz vectors), stored column-major.
    /// Only populated if eigenvectors were requested.
    pub eigenvectors: Option<Vec<Vec<T>>>,
    /// Number of Lanczos iterations performed.
    pub iterations: usize,
    /// Residual norms for each converged eigenpair.
    pub residual_norms: Vec<T>,
    /// Whether all requested eigenvalues converged.
    pub converged: bool,
}

/// Configuration for Lanczos iteration.
#[derive(Debug, Clone)]
pub struct LanczosConfig<T> {
    /// Number of eigenvalues to compute.
    pub num_eigenvalues: usize,
    /// Which eigenvalues to compute.
    pub which: WhichEigenvalues,
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence tolerance for residual norm.
    pub tolerance: T,
    /// Whether to compute eigenvectors.
    pub compute_eigenvectors: bool,
    /// Size of Krylov subspace (should be > num_eigenvalues).
    pub krylov_dimension: usize,
    /// Whether to use full reorthogonalization (more stable, more expensive).
    pub full_reorthogonalization: bool,
}

impl Default for LanczosConfig<f64> {
    fn default() -> Self {
        Self {
            num_eigenvalues: 6,
            which: WhichEigenvalues::LargestMagnitude,
            max_iterations: 300,
            tolerance: 1e-8,
            compute_eigenvectors: true,
            krylov_dimension: 20,
            full_reorthogonalization: true,
        }
    }
}

impl Default for LanczosConfig<f32> {
    fn default() -> Self {
        Self {
            num_eigenvalues: 6,
            which: WhichEigenvalues::LargestMagnitude,
            max_iterations: 300,
            tolerance: 1e-6,
            compute_eigenvectors: true,
            krylov_dimension: 20,
            full_reorthogonalization: true,
        }
    }
}

// =============================================================================
// Lanczos Solver
// =============================================================================

/// Lanczos iteration for sparse symmetric matrices.
///
/// Computes the k largest or smallest eigenvalues (and optionally eigenvectors)
/// of a symmetric matrix A using the Lanczos algorithm.
///
/// # Algorithm
///
/// The Lanczos algorithm builds an orthonormal basis Q for the Krylov subspace
/// K_m(A, v) = span{v, Av, A^2v, ..., A^(m-1)v} and reduces A to tridiagonal form
/// T = Q^T A Q. The eigenvalues of T (Ritz values) approximate eigenvalues of A.
///
/// # Example
///
/// ```ignore
/// use oxiblas_sparse::csr::CsrMatrix;
/// use oxiblas_sparse::linalg::eigenvalue::{Lanczos, LanczosConfig, WhichEigenvalues};
///
/// // Create a sparse symmetric matrix
/// let a = CsrMatrix::<f64>::eye(100);
///
/// let config = LanczosConfig {
///     num_eigenvalues: 5,
///     which: WhichEigenvalues::LargestMagnitude,
///     ..Default::default()
/// };
///
/// let lanczos = Lanczos::new(config);
/// let result = lanczos.compute(&a, None).unwrap();
/// println!("Eigenvalues: {:?}", result.eigenvalues);
/// ```
pub struct Lanczos<T> {
    config: LanczosConfig<T>,
}

impl<T: Scalar<Real = T> + Clone + Field + Real + FromPrimitive> Lanczos<T> {
    /// Create a new Lanczos solver with the given configuration.
    pub fn new(config: LanczosConfig<T>) -> Self {
        Self { config }
    }

    /// Compute eigenvalues (and optionally eigenvectors) of a symmetric matrix.
    ///
    /// # Arguments
    ///
    /// * `a` - Symmetric sparse matrix in CSR format
    /// * `initial_vector` - Optional starting vector (random if None)
    ///
    /// # Returns
    ///
    /// Computed eigenvalues and eigenvectors.
    pub fn compute(
        &self,
        a: &CsrMatrix<T>,
        initial_vector: Option<&[T]>,
    ) -> Result<LanczosResult<T>, EigenvalueError> {
        let n = a.nrows();

        if a.ncols() != n {
            return Err(EigenvalueError::NotSquare {
                nrows: n,
                ncols: a.ncols(),
            });
        }

        let k = self.config.num_eigenvalues;
        let m = self.config.krylov_dimension.max(k + 1).min(n);

        if k > n {
            return Err(EigenvalueError::TooManyEigenvalues {
                requested: k,
                max_allowed: n,
            });
        }

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
            // Start with vector [1, 1, ..., 1] / sqrt(n)
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

        // Storage for Lanczos vectors (Q matrix)
        let mut lanczos_vectors: Vec<Vec<T>> = Vec::with_capacity(m);
        lanczos_vectors.push(v.clone());

        // Tridiagonal matrix elements
        let mut alpha: Vec<T> = Vec::with_capacity(m); // Diagonal
        let mut beta: Vec<T> = Vec::with_capacity(m); // Off-diagonal

        // Working vectors
        let mut w = vec![T::zero(); n];
        let mut v_prev = vec![T::zero(); n];

        let tol_breakdown = <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or_else(T::one);

        // Lanczos iteration
        for j in 0..m {
            // w = A * v
            spmv(T::one(), a, &v, T::zero(), &mut w);

            // alpha[j] = v^T * w
            let alpha_j = dot(&v, &w);
            alpha.push(alpha_j.clone());

            // w = w - alpha[j] * v - beta[j-1] * v_prev
            for i in 0..n {
                w[i] = w[i].clone() - alpha_j.clone() * v[i].clone();
            }
            if j > 0 {
                let beta_prev = beta[j - 1].clone();
                for i in 0..n {
                    w[i] = w[i].clone() - beta_prev.clone() * v_prev[i].clone();
                }
            }

            // Full reorthogonalization against all previous vectors
            if self.config.full_reorthogonalization {
                for qj in &lanczos_vectors {
                    let h = dot(qj, &w);
                    for i in 0..n {
                        w[i] = w[i].clone() - h.clone() * qj[i].clone();
                    }
                }
            }

            // beta[j] = ||w||
            let beta_j = norm(&w);

            // Check for breakdown (invariant subspace found)
            if beta_j <= tol_breakdown {
                // Early termination - we found an invariant subspace
                // Proceed with what we have
                break;
            }

            beta.push(beta_j.clone());

            if j + 1 < m {
                // v_prev = v
                v_prev.clone_from_slice(&v);

                // v = w / beta[j]
                for i in 0..n {
                    v[i] = w[i].clone() / beta_j.clone();
                }

                lanczos_vectors.push(v.clone());
            }
        }

        let actual_dim = alpha.len();

        // Compute eigenvalues of the tridiagonal matrix using QR iteration
        let (ritz_values, ritz_vectors) = self.solve_tridiagonal(&alpha, &beta)?;

        // Select which eigenvalues to return based on configuration
        let (selected_indices, selected_eigenvalues) = self.select_eigenvalues(&ritz_values, k);

        // Compute residual norms and check convergence
        let mut residual_norms = Vec::with_capacity(k);
        let mut converged_count = 0;

        for &idx in &selected_indices {
            // Residual norm for Ritz pair: ||A*x - lambda*x|| ~ |beta_m * s_m|
            // where s_m is the last component of the Ritz vector in the Lanczos basis
            let residual = if idx < ritz_vectors.len() && actual_dim > 0 && !beta.is_empty() {
                let s_last = ritz_vectors[idx][actual_dim - 1].clone();
                Scalar::abs(beta.last().unwrap().clone() * s_last)
            } else {
                T::zero()
            };

            residual_norms.push(residual.clone());
            if residual <= self.config.tolerance {
                converged_count += 1;
            }
        }

        // Compute eigenvectors if requested
        let eigenvectors = if self.config.compute_eigenvectors {
            let mut evecs = Vec::with_capacity(selected_indices.len());

            for &idx in &selected_indices {
                if idx < ritz_vectors.len() {
                    // Transform Ritz vector from Lanczos basis to original basis
                    // x = Q * y where y is the Ritz vector
                    let y = &ritz_vectors[idx];
                    let mut x = vec![T::zero(); n];

                    for (j, qj) in lanczos_vectors.iter().enumerate() {
                        if j < y.len() {
                            for i in 0..n {
                                x[i] = x[i].clone() + y[j].clone() * qj[i].clone();
                            }
                        }
                    }

                    // Normalize eigenvector
                    let x_norm = norm(&x);
                    if x_norm > <T as Scalar>::epsilon() {
                        for xi in &mut x {
                            *xi = xi.clone() / x_norm.clone();
                        }
                    }

                    evecs.push(x);
                }
            }

            Some(evecs)
        } else {
            None
        };

        let converged = converged_count >= k;

        Ok(LanczosResult {
            eigenvalues: selected_eigenvalues,
            eigenvectors,
            iterations: actual_dim,
            residual_norms,
            converged,
        })
    }

    /// Solve eigenvalue problem for symmetric tridiagonal matrix.
    ///
    /// Uses implicit QR iteration with Wilkinson shifts for stability.
    fn solve_tridiagonal(
        &self,
        alpha: &[T],
        beta: &[T],
    ) -> Result<(Vec<T>, Vec<Vec<T>>), EigenvalueError> {
        let n = alpha.len();
        if n == 0 {
            return Ok((vec![], vec![]));
        }

        // Copy diagonal and off-diagonal
        let mut d = alpha.to_vec();
        let mut e = if beta.is_empty() {
            vec![T::zero(); n - 1]
        } else {
            let mut e = beta.to_vec();
            // Ensure e has n-1 elements
            e.truncate(n - 1);
            while e.len() < n - 1 {
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

        // QR iteration for symmetric tridiagonal eigenvalue problem
        let max_qr_iter = 30 * n;
        let tol = <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or_else(T::one);
        let two = T::from_f64(2.0).unwrap_or_else(T::one);

        for _iter in 0..max_qr_iter {
            // Find the largest unreduced submatrix
            let mut l = 0;
            for i in (0..n - 1).rev() {
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
                // All off-diagonal elements are zero - eigenvalues found
                break;
            }

            // Find the start of the unreduced block
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

            // Apply implicit QR step
            let mut g = d[m].clone() - shift.clone();
            let mut s = T::one();
            let mut c = T::one();
            let mut p_val = T::zero();

            for i in m..l {
                let f = s.clone() * e[i].clone();
                let b = c.clone() * e[i].clone();

                // Givens rotation
                if Scalar::abs(f.clone()) >= Scalar::abs(g.clone()) {
                    c = g.clone() / f.clone();
                    let r = Real::sqrt(c.clone() * c.clone() + T::one());
                    if i > m {
                        e[i - 1] = f.clone() * r.clone();
                    }
                    s = T::one() / r.clone();
                    c = c.clone() * s.clone();
                } else {
                    s = f.clone() / g.clone();
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
                    let f = z[k][i + 1].clone();
                    z[k][i + 1] = s.clone() * z[k][i].clone() + c.clone() * f.clone();
                    z[k][i] = c.clone() * z[k][i].clone() - s.clone() * f;
                }
            }

            d[l] = d[l].clone() - p_val.clone();
            e[l - 1] = g;
        }

        // Extract eigenvalues and corresponding eigenvectors
        let eigenvalues = d;
        let eigenvectors: Vec<Vec<T>> = (0..n)
            .map(|i| z.iter().map(|row| row[i].clone()).collect())
            .collect();

        Ok((eigenvalues, eigenvectors))
    }

    /// Select eigenvalues based on the configuration.
    fn select_eigenvalues(&self, eigenvalues: &[T], k: usize) -> (Vec<usize>, Vec<T>) {
        if eigenvalues.is_empty() {
            return (vec![], vec![]);
        }

        let n = eigenvalues.len();
        let k = k.min(n);

        // Create index-value pairs
        let mut indexed: Vec<(usize, T)> = eigenvalues
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.clone()))
            .collect();

        // Sort based on which eigenvalues are requested
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
                // For NearTarget, would need shift-and-invert
                // For now, same as SmallestMagnitude
                indexed.sort_by(|a, b| {
                    Scalar::abs(a.1.clone())
                        .partial_cmp(&Scalar::abs(b.1.clone()))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        // Take top k
        let indices: Vec<usize> = indexed.iter().take(k).map(|(i, _)| *i).collect();
        let values: Vec<T> = indexed.iter().take(k).map(|(_, v)| v.clone()).collect();

        (indices, values)
    }
}
