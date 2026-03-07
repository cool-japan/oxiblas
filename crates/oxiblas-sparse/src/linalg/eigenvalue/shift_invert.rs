//! Shift-and-invert Lanczos method for interior eigenvalue computation.
//!
//! This module provides the shift-and-invert Lanczos algorithm for computing
//! eigenvalues of a sparse symmetric matrix near a specified shift value.
//! The method works by applying the standard Lanczos iteration to the
//! operator (A - σI)^{-1}, where σ is the shift.
//!
//! # Algorithm Overview
//!
//! For a symmetric matrix A and shift σ:
//! 1. Factorize (A - σI) using sparse Cholesky (for SPD) or LU decomposition
//! 2. Run Lanczos iteration on (A - σI)^{-1}
//! 3. Transform eigenvalues: λ = σ + 1/θ
//!
//! This approach finds eigenvalues closest to the shift σ, making it
//! ideal for computing interior eigenvalues.
//!
//! # Example
//!
//! ```ignore
//! use oxiblas_sparse::csr::CsrMatrix;
//! use oxiblas_sparse::linalg::eigenvalue::{ShiftInvertLanczos, ShiftInvertConfig};
//!
//! // Create a sparse symmetric matrix
//! let a = CsrMatrix::<f64>::eye(100);
//!
//! let config = ShiftInvertConfig {
//!     num_eigenvalues: 5,
//!     shift: 0.5,  // Find eigenvalues near 0.5
//!     ..Default::default()
//! };
//!
//! let solver = ShiftInvertLanczos::new(config);
//! let result = solver.compute(&a, None).unwrap();
//! println!("Eigenvalues near 0.5: {:?}", result.eigenvalues);
//! ```

use crate::csr::CsrMatrix;
use crate::linalg::cholesky::SparseCholesky;
use crate::linalg::lu::SparseLU;
use num_traits::FromPrimitive;
use oxiblas_core::scalar::{Field, Real, Scalar};

use super::error::EigenvalueError;
use super::utils::{csr_to_csc, dot, norm};

/// Configuration for shift-and-invert eigenvalue computation.
#[derive(Debug, Clone)]
pub struct ShiftInvertConfig<T> {
    /// Number of eigenvalues to compute.
    pub num_eigenvalues: usize,
    /// Shift value (target for interior eigenvalues).
    pub shift: T,
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence tolerance for residual norm.
    pub tolerance: T,
    /// Whether to compute eigenvectors.
    pub compute_eigenvectors: bool,
    /// Size of Krylov subspace.
    pub krylov_dimension: usize,
    /// Whether to use full reorthogonalization.
    pub full_reorthogonalization: bool,
    /// Whether the matrix is symmetric (use Cholesky instead of LU).
    pub symmetric: bool,
}

impl Default for ShiftInvertConfig<f64> {
    fn default() -> Self {
        Self {
            num_eigenvalues: 6,
            shift: 0.0,
            max_iterations: 300,
            tolerance: 1e-8,
            compute_eigenvectors: true,
            krylov_dimension: 20,
            full_reorthogonalization: true,
            symmetric: true,
        }
    }
}

impl Default for ShiftInvertConfig<f32> {
    fn default() -> Self {
        Self {
            num_eigenvalues: 6,
            shift: 0.0,
            max_iterations: 300,
            tolerance: 1e-6,
            compute_eigenvectors: true,
            krylov_dimension: 20,
            full_reorthogonalization: true,
            symmetric: true,
        }
    }
}

/// Result of shift-and-invert eigenvalue computation.
#[derive(Debug, Clone)]
pub struct ShiftInvertResult<T> {
    /// Computed eigenvalues of the original matrix A (near the shift).
    pub eigenvalues: Vec<T>,
    /// Eigenvectors (if requested), stored as column vectors.
    pub eigenvectors: Option<Vec<Vec<T>>>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Residual norms for each eigenpair.
    pub residual_norms: Vec<T>,
    /// Whether all requested eigenvalues converged.
    pub converged: bool,
    /// The shift used.
    pub shift: T,
}

/// Shift-and-invert Lanczos for finding interior eigenvalues.
///
/// Computes eigenvalues of a symmetric matrix A near a specified shift σ
/// by applying Lanczos iteration to (A - σI)^{-1}.
///
/// If λ is an eigenvalue of A, then θ = 1/(λ - σ) is an eigenvalue of (A - σI)^{-1}.
/// Eigenvalues of (A - σI)^{-1} with largest magnitude correspond to eigenvalues of A
/// closest to σ.
///
/// # Algorithm
///
/// 1. Factorize (A - σI) using sparse Cholesky (symmetric) or LU (general)
/// 2. Run Lanczos iteration on (A - σI)^{-1}
/// 3. Transform eigenvalues: λ = σ + 1/θ
///
/// # Example
///
/// ```ignore
/// use oxiblas_sparse::csr::CsrMatrix;
/// use oxiblas_sparse::linalg::eigenvalue::{ShiftInvertLanczos, ShiftInvertConfig};
///
/// // Create a sparse symmetric matrix
/// let a = CsrMatrix::<f64>::eye(100);
///
/// let config = ShiftInvertConfig {
///     num_eigenvalues: 5,
///     shift: 0.5,  // Find eigenvalues near 0.5
///     ..Default::default()
/// };
///
/// let solver = ShiftInvertLanczos::new(config);
/// let result = solver.compute(&a, None).unwrap();
/// println!("Eigenvalues near 0.5: {:?}", result.eigenvalues);
/// ```
pub struct ShiftInvertLanczos<T> {
    config: ShiftInvertConfig<T>,
}

impl<T: Scalar<Real = T> + Clone + Field + Real + FromPrimitive> ShiftInvertLanczos<T> {
    /// Create a new shift-and-invert Lanczos solver.
    pub fn new(config: ShiftInvertConfig<T>) -> Self {
        Self { config }
    }

    /// Compute eigenvalues near the shift using shift-and-invert Lanczos.
    ///
    /// # Arguments
    ///
    /// * `a` - Symmetric sparse matrix in CSR format
    /// * `initial_vector` - Optional starting vector
    ///
    /// # Returns
    ///
    /// Eigenvalues of A closest to the shift σ.
    pub fn compute(
        &self,
        a: &CsrMatrix<T>,
        initial_vector: Option<&[T]>,
    ) -> Result<ShiftInvertResult<T>, EigenvalueError> {
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

        // Build shifted matrix: B = A - σI (as CSR)
        let shifted_csr = self.build_shifted_matrix(a)?;

        // Convert to CSC for factorization
        let shifted_csc = csr_to_csc(&shifted_csr);

        // Factorize the shifted matrix
        let (solve_fn, _use_cholesky) = if self.config.symmetric {
            // Try Cholesky first (for SPD matrices shifted to remain SPD)
            match SparseCholesky::new(&shifted_csc) {
                Ok(chol) => {
                    let solve: Box<dyn Fn(&[T]) -> Vec<T>> = Box::new(move |b: &[T]| chol.solve(b));
                    (solve, true)
                }
                Err(_) => {
                    // Fall back to LU if Cholesky fails
                    let lu = SparseLU::new(&shifted_csc).map_err(|e| {
                        EigenvalueError::ComputationError(format!("LU factorization failed: {}", e))
                    })?;
                    let solve: Box<dyn Fn(&[T]) -> Vec<T>> = Box::new(move |b: &[T]| lu.solve(b));
                    (solve, false)
                }
            }
        } else {
            let lu = SparseLU::new(&shifted_csc).map_err(|e| {
                EigenvalueError::ComputationError(format!("LU factorization failed: {}", e))
            })?;
            let solve: Box<dyn Fn(&[T]) -> Vec<T>> = Box::new(move |b: &[T]| lu.solve(b));
            (solve, false)
        };

        // Run Lanczos on (A - σI)^{-1}
        let lanczos_result = self.run_lanczos_with_operator(n, m, k, &solve_fn, initial_vector)?;

        // Transform eigenvalues back: λ = σ + 1/θ
        let eigenvalues = self.transform_eigenvalues(&lanczos_result.0, &lanczos_result.1);

        // Eigenvectors are the same (shared between A and (A-σI)^{-1})
        let eigenvectors = lanczos_result.2;
        let residual_norms = lanczos_result.3;
        let iterations = lanczos_result.4;
        let converged = lanczos_result.5;

        Ok(ShiftInvertResult {
            eigenvalues,
            eigenvectors,
            iterations,
            residual_norms,
            converged,
            shift: self.config.shift.clone(),
        })
    }

    /// Build the shifted matrix A - σI.
    fn build_shifted_matrix(&self, a: &CsrMatrix<T>) -> Result<CsrMatrix<T>, EigenvalueError> {
        let n = a.nrows();
        let shift = self.config.shift.clone();

        // Create A - σI by modifying diagonal entries
        let mut row_ptrs = a.row_ptrs().to_vec();
        let mut col_indices = a.col_indices().to_vec();
        let mut values = a.values().to_vec();

        // Check if diagonal entries exist, add them if not
        for i in 0..n {
            let row_start = row_ptrs[i];
            let row_end = row_ptrs[i + 1];

            let mut found_diag = false;
            for j in row_start..row_end {
                if col_indices[j] == i {
                    // Subtract shift from diagonal
                    values[j] = values[j].clone() - shift.clone();
                    found_diag = true;
                    break;
                }
            }

            if !found_diag {
                // Need to insert diagonal element -σ
                // This is expensive but handles the case where diagonal is zero
                // For most practical matrices, diagonal exists
                let insert_pos = row_end;
                col_indices.insert(insert_pos, i);
                values.insert(insert_pos, T::zero() - shift.clone());
                // Update row pointers for subsequent rows
                for rp in row_ptrs.iter_mut().skip(i + 1) {
                    *rp += 1;
                }
            }
        }

        CsrMatrix::new(n, n, row_ptrs, col_indices, values).map_err(|e| {
            EigenvalueError::ComputationError(format!("Failed to create shifted matrix: {}", e))
        })
    }

    /// Run Lanczos iteration with a custom linear operator.
    fn run_lanczos_with_operator<F>(
        &self,
        n: usize,
        m: usize,
        k: usize,
        apply_op: &F,
        initial_vector: Option<&[T]>,
    ) -> Result<(Vec<T>, Vec<usize>, Option<Vec<Vec<T>>>, Vec<T>, usize, bool), EigenvalueError>
    where
        F: Fn(&[T]) -> Vec<T>,
    {
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

        // Storage for Lanczos vectors
        let mut lanczos_vectors: Vec<Vec<T>> = Vec::with_capacity(m);
        lanczos_vectors.push(v.clone());

        // Tridiagonal matrix elements
        let mut alpha: Vec<T> = Vec::with_capacity(m);
        let mut beta: Vec<T> = Vec::with_capacity(m);

        // Working vectors
        let mut v_prev = vec![T::zero(); n];

        let tol_breakdown = <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or_else(T::one);

        // Lanczos iteration with custom operator
        for j in 0..m {
            // w = Op * v (instead of A * v)
            let mut w = apply_op(&v);

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

            // Full reorthogonalization
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

            if beta_j <= tol_breakdown {
                break;
            }

            beta.push(beta_j.clone());

            if j + 1 < m {
                v_prev.clone_from_slice(&v);
                for i in 0..n {
                    v[i] = w[i].clone() / beta_j.clone();
                }
                lanczos_vectors.push(v.clone());
            }
        }

        let actual_dim = alpha.len();

        // Solve tridiagonal eigenvalue problem
        let (ritz_values, ritz_vectors) = self.solve_tridiagonal(&alpha, &beta)?;

        // Select eigenvalues with largest magnitude (closest to shift after transform)
        let (selected_indices, selected_values) =
            self.select_eigenvalues_by_magnitude(&ritz_values, k);

        // Compute residual norms
        let mut residual_norms = Vec::with_capacity(k);
        let mut converged_count = 0;

        for &idx in &selected_indices {
            let residual = if idx < ritz_vectors.len() && actual_dim > 0 && !beta.is_empty() {
                let s_last = ritz_vectors[idx][actual_dim - 1].clone();
                Scalar::abs(beta.last().expect("collection should be non-empty").clone() * s_last)
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
                    let y = &ritz_vectors[idx];
                    let mut x = vec![T::zero(); n];

                    for (j, qj) in lanczos_vectors.iter().enumerate() {
                        if j < y.len() {
                            for i in 0..n {
                                x[i] = x[i].clone() + y[j].clone() * qj[i].clone();
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

                    evecs.push(x);
                }
            }

            Some(evecs)
        } else {
            None
        };

        let converged = converged_count >= k;

        Ok((
            selected_values,
            selected_indices,
            eigenvectors,
            residual_norms,
            actual_dim,
            converged,
        ))
    }

    /// Solve symmetric tridiagonal eigenvalue problem.
    fn solve_tridiagonal(
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
            vec![T::zero(); n - 1]
        } else {
            let mut e = beta.to_vec();
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

        let max_qr_iter = 30 * n;
        let tol = <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or_else(T::one);
        let two = T::from_f64(2.0).unwrap_or_else(T::one);

        for _iter in 0..max_qr_iter {
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
                let f = s.clone() * e[i].clone();
                let b = c.clone() * e[i].clone();

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

        let eigenvalues = d;
        let eigenvectors: Vec<Vec<T>> = (0..n)
            .map(|i| z.iter().map(|row| row[i].clone()).collect())
            .collect();

        Ok((eigenvalues, eigenvectors))
    }

    /// Select eigenvalues by largest magnitude.
    fn select_eigenvalues_by_magnitude(&self, eigenvalues: &[T], k: usize) -> (Vec<usize>, Vec<T>) {
        if eigenvalues.is_empty() {
            return (vec![], vec![]);
        }

        let n = eigenvalues.len();
        let k = k.min(n);

        let mut indexed: Vec<(usize, T)> = eigenvalues
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.clone()))
            .collect();

        // Sort by largest magnitude (for shift-invert, largest θ = closest λ to σ)
        indexed.sort_by(|a, b| {
            Scalar::abs(b.1.clone())
                .partial_cmp(&Scalar::abs(a.1.clone()))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let indices: Vec<usize> = indexed.iter().take(k).map(|(i, _)| *i).collect();
        let values: Vec<T> = indexed.iter().take(k).map(|(_, v)| v.clone()).collect();

        (indices, values)
    }

    /// Transform eigenvalues from (A-σI)^{-1} back to A.
    ///
    /// λ = σ + 1/θ where θ is eigenvalue of (A-σI)^{-1}
    fn transform_eigenvalues(&self, theta_values: &[T], _indices: &[usize]) -> Vec<T> {
        let shift = self.config.shift.clone();
        let tol = <T as Scalar>::epsilon() * T::from_f64(1e6).unwrap_or_else(T::one);

        theta_values
            .iter()
            .map(|theta| {
                if Scalar::abs(theta.clone()) <= tol {
                    // θ ≈ 0 means λ very far from σ (or at infinity)
                    // Return a large value
                    shift.clone() + T::from_f64(1e10).unwrap_or_else(T::one)
                } else {
                    shift.clone() + T::one() / theta.clone()
                }
            })
            .collect()
    }
}
