//! Generalized Eigenvalue Problem: A*x = lambda*B*x
//!
//! This module provides solvers for generalized eigenvalue problems where
//! we seek eigenvalues lambda and eigenvectors x satisfying A*x = lambda*B*x,
//! with B being symmetric positive definite (SPD).
//!
//! # Supported Modes
//!
//! - **Standard**: Transforms to B^{-1}*A standard problem. Best when B is
//!   well-conditioned and you want extreme eigenvalues.
//!
//! - **Shift-Invert**: Uses operator (A - sigma*B)^{-1}*B to find eigenvalues near sigma.
//!   The eigenvalues of this operator are mu = 1/(lambda - sigma), so eigenvalues of
//!   the original problem near sigma become large in magnitude.
//!
//! - **Buckling**: Uses operator (A - sigma*B)^{-1}*A for buckling problems.
//!
//! - **Cayley**: Uses operator (A - sigma*B)^{-1}*(A + sigma*B) for interior eigenvalues.
//!
//! # Algorithm
//!
//! For symmetric problems (A symmetric, B SPD):
//! - Uses B-orthogonal Lanczos iteration
//! - Maintains B-orthonormality: V^T * B * V = I
//! - Implicit restarts via QR shifts

use crate::csr::CsrMatrix;
use crate::linalg::cholesky::SparseCholesky;
use crate::linalg::lu::SparseLU;
use crate::ops::spmv;
use num_traits::FromPrimitive;
use oxiblas_core::scalar::{Field, Real, Scalar};

use super::error::{EigenvalueError, WhichEigenvalues};
use super::utils::{add_scaled_matrices, csr_to_csc, dot, norm, subtract_scaled_matrices};

// =============================================================================
// Generalized Eigenvalue Problem: A*x = lambda*B*x
// =============================================================================

/// Mode for generalized eigenvalue computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GeneralizedMode {
    /// Standard mode: compute eigenvalues of B^{-1}*A
    /// Best for well-conditioned B and eigenvalues with large magnitude.
    Standard,
    /// Shift-and-invert mode: compute eigenvalues of (A - sigma*B)^{-1}*B
    /// Finds eigenvalues near the shift sigma.
    #[default]
    ShiftInvert,
    /// Buckling mode: compute eigenvalues of (A - sigma*B)^{-1}*A
    /// Useful for buckling problems where A is the stiffness matrix.
    Buckling,
    /// Cayley mode: compute eigenvalues of (A - sigma*B)^{-1}*(A + sigma*B)
    /// Good for interior eigenvalues of the generalized problem.
    Cayley,
}

/// Configuration for generalized eigenvalue solver.
#[derive(Debug, Clone)]
pub struct GeneralizedEigenConfig<T> {
    /// Number of eigenvalues to compute.
    pub num_eigenvalues: usize,
    /// Which eigenvalues to compute.
    pub which: WhichEigenvalues,
    /// Maximum number of outer iterations.
    pub max_iterations: usize,
    /// Convergence tolerance.
    pub tolerance: T,
    /// Whether to compute eigenvectors.
    pub compute_eigenvectors: bool,
    /// Size of Krylov subspace.
    pub krylov_dimension: usize,
    /// Whether A is symmetric.
    pub symmetric: bool,
    /// Mode of operation.
    pub mode: GeneralizedMode,
    /// Shift value for shift-invert modes.
    pub sigma: T,
}

impl Default for GeneralizedEigenConfig<f64> {
    fn default() -> Self {
        Self {
            num_eigenvalues: 6,
            which: WhichEigenvalues::LargestMagnitude,
            max_iterations: 300,
            tolerance: 1e-8,
            compute_eigenvectors: true,
            krylov_dimension: 20,
            symmetric: true,
            mode: GeneralizedMode::ShiftInvert,
            sigma: 0.0,
        }
    }
}

impl Default for GeneralizedEigenConfig<f32> {
    fn default() -> Self {
        Self {
            num_eigenvalues: 6,
            which: WhichEigenvalues::LargestMagnitude,
            max_iterations: 300,
            tolerance: 1e-6,
            compute_eigenvectors: true,
            krylov_dimension: 20,
            symmetric: true,
            mode: GeneralizedMode::ShiftInvert,
            sigma: 0.0,
        }
    }
}

/// Result of generalized eigenvalue computation.
#[derive(Debug, Clone)]
pub struct GeneralizedEigenResult<T> {
    /// Computed eigenvalues (real parts for real matrices).
    pub eigenvalues: Vec<T>,
    /// Eigenvectors (if requested), stored as column vectors.
    /// These are already B-orthonormal: x_i^T * B * x_j = delta_{ij}
    pub eigenvectors: Option<Vec<Vec<T>>>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Residual norms ||A*x - lambda*B*x|| for each eigenpair.
    pub residual_norms: Vec<T>,
    /// Whether all requested eigenvalues converged.
    pub converged: bool,
    /// Number of converged eigenvalues.
    pub num_converged: usize,
}

/// Generalized Eigenvalue Solver for A*x = lambda*B*x.
///
/// Computes eigenvalues and eigenvectors of the generalized eigenvalue problem
/// A*x = lambda*B*x where B is symmetric positive definite (SPD).
///
/// # Supported Modes
///
/// - **Standard**: Transforms to B^{-1}*A standard problem. Best when B is
///   well-conditioned and you want extreme eigenvalues.
///
/// - **Shift-Invert**: Uses operator (A - sigma*B)^{-1}*B to find eigenvalues near sigma.
///   The eigenvalues of this operator are mu = 1/(lambda - sigma), so eigenvalues of
///   the original problem near sigma become large in magnitude.
///
/// - **Buckling**: Uses operator (A - sigma*B)^{-1}*A for buckling problems.
///
/// - **Cayley**: Uses operator (A - sigma*B)^{-1}*(A + sigma*B) for interior eigenvalues.
///
/// # Algorithm
///
/// For symmetric problems (A symmetric, B SPD):
/// - Uses B-orthogonal Lanczos iteration
/// - Maintains B-orthonormality: V^T * B * V = I
/// - Implicit restarts via QR shifts
///
/// # Example
///
/// ```ignore
/// use oxiblas_sparse::csr::CsrMatrix;
/// use oxiblas_sparse::linalg::eigenvalue::{
///     GeneralizedEigen, GeneralizedEigenConfig, GeneralizedMode, WhichEigenvalues
/// };
///
/// // A = stiffness matrix, B = mass matrix
/// let a = create_stiffness_matrix();
/// let b = create_mass_matrix(); // SPD
///
/// let config = GeneralizedEigenConfig {
///     num_eigenvalues: 5,
///     which: WhichEigenvalues::SmallestMagnitude,
///     mode: GeneralizedMode::ShiftInvert,
///     sigma: 0.0,  // Find smallest eigenvalues
///     symmetric: true,
///     ..Default::default()
/// };
///
/// let solver = GeneralizedEigen::new(config);
/// let result = solver.compute(&a, &b, None)?;
/// ```
pub struct GeneralizedEigen<T> {
    config: GeneralizedEigenConfig<T>,
}

impl<T: Scalar<Real = T> + Clone + Field + Real + FromPrimitive> GeneralizedEigen<T> {
    /// Create a new generalized eigenvalue solver with the given configuration.
    pub fn new(config: GeneralizedEigenConfig<T>) -> Self {
        Self { config }
    }

    /// Compute eigenvalues (and optionally eigenvectors) of A*x = lambda*B*x.
    ///
    /// # Arguments
    ///
    /// * `a` - Square sparse matrix A in CSR format
    /// * `b` - Square sparse SPD matrix B in CSR format
    /// * `initial_vector` - Optional starting vector
    ///
    /// # Returns
    ///
    /// Computed eigenvalues and eigenvectors.
    pub fn compute(
        &self,
        a: &CsrMatrix<T>,
        b: &CsrMatrix<T>,
        initial_vector: Option<&[T]>,
    ) -> Result<GeneralizedEigenResult<T>, EigenvalueError> {
        let n = a.nrows();

        // Validate dimensions
        if a.ncols() != n {
            return Err(EigenvalueError::NotSquare {
                nrows: n,
                ncols: a.ncols(),
            });
        }
        if b.nrows() != n || b.ncols() != n {
            return Err(EigenvalueError::DimensionMismatch {
                expected: n,
                actual: b.nrows(),
            });
        }

        let nev = self.config.num_eigenvalues;
        let ncv = self.config.krylov_dimension.max(nev + 2).min(n);

        if nev > n {
            return Err(EigenvalueError::TooManyEigenvalues {
                requested: nev,
                max_allowed: n,
            });
        }

        match self.config.mode {
            GeneralizedMode::Standard => self.compute_standard(a, b, n, nev, ncv, initial_vector),
            GeneralizedMode::ShiftInvert => {
                self.compute_shift_invert(a, b, n, nev, ncv, initial_vector)
            }
            GeneralizedMode::Buckling => self.compute_buckling(a, b, n, nev, ncv, initial_vector),
            GeneralizedMode::Cayley => self.compute_cayley(a, b, n, nev, ncv, initial_vector),
        }
    }

    /// Standard mode: work with B^{-1}*A.
    fn compute_standard(
        &self,
        a: &CsrMatrix<T>,
        b: &CsrMatrix<T>,
        n: usize,
        nev: usize,
        ncv: usize,
        initial_vector: Option<&[T]>,
    ) -> Result<GeneralizedEigenResult<T>, EigenvalueError> {
        // Convert B to CSC for Cholesky factorization
        let b_csc = csr_to_csc(b);

        // Factor B for efficient solves
        let b_factor = SparseCholesky::new(&b_csc)
            .map_err(|e| EigenvalueError::ComputationError(format!("B must be SPD: {:?}", e)))?;

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

        // B-orthogonal Lanczos for symmetric case
        if self.config.symmetric {
            self.lanczos_b_orthogonal(a, b, &b_factor, n, nev, ncv, v, |x| {
                // Operator: B^{-1}*A*x
                let mut ax = vec![T::zero(); n];
                spmv(T::one(), a, x, T::zero(), &mut ax);
                b_factor.solve(&ax)
            })
        } else {
            // For non-symmetric, use general Arnoldi
            self.arnoldi_generalized(a, b, &b_factor, n, nev, ncv, v, |x| {
                let mut ax = vec![T::zero(); n];
                spmv(T::one(), a, x, T::zero(), &mut ax);
                b_factor.solve(&ax)
            })
        }
    }

    /// Shift-invert mode: work with (A - sigma*B)^{-1}*B.
    fn compute_shift_invert(
        &self,
        a: &CsrMatrix<T>,
        b: &CsrMatrix<T>,
        n: usize,
        nev: usize,
        ncv: usize,
        initial_vector: Option<&[T]>,
    ) -> Result<GeneralizedEigenResult<T>, EigenvalueError> {
        let sigma = self.config.sigma.clone();

        // Compute C = A - sigma*B and convert to CSC
        let c = subtract_scaled_matrices(a, b, sigma.clone());
        let c_csc = csr_to_csc(&c);

        // Factor C = A - sigma*B for efficient solves
        let c_factor = SparseLU::new(&c_csc).map_err(|e| {
            EigenvalueError::ComputationError(format!(
                "Failed to factor (A - sigma*B): {:?}. Try a different shift.",
                e
            ))
        })?;

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

        // Convert B to CSC for Cholesky factorization
        let b_csc = csr_to_csc(b);

        // For shift-invert, eigenvalues of op are mu = 1/(lambda - sigma)
        // We compute with op = (A - sigma*B)^{-1}*B
        let result = if self.config.symmetric {
            // Factor B for B-inner product
            let b_factor = SparseCholesky::new(&b_csc).map_err(|e| {
                EigenvalueError::ComputationError(format!("B must be SPD: {:?}", e))
            })?;

            self.lanczos_b_orthogonal(a, b, &b_factor, n, nev, ncv, v, |x| {
                // Operator: (A - sigma*B)^{-1}*B*x
                let mut bx = vec![T::zero(); n];
                spmv(T::one(), b, x, T::zero(), &mut bx);
                c_factor.solve(&bx)
            })?
        } else {
            let b_factor = SparseCholesky::new(&b_csc).map_err(|e| {
                EigenvalueError::ComputationError(format!("B must be SPD: {:?}", e))
            })?;

            self.arnoldi_generalized(a, b, &b_factor, n, nev, ncv, v, |x| {
                let mut bx = vec![T::zero(); n];
                spmv(T::one(), b, x, T::zero(), &mut bx);
                c_factor.solve(&bx)
            })?
        };

        // Transform eigenvalues back: lambda = sigma + 1/mu
        let eigenvalues: Vec<T> = result
            .eigenvalues
            .iter()
            .map(|mu| {
                if Scalar::abs(mu.clone()) > <T as Scalar>::epsilon() {
                    sigma.clone() + T::one() / mu.clone()
                } else {
                    // Eigenvalue at infinity - shouldn't happen for well-posed problems
                    sigma.clone()
                }
            })
            .collect();

        Ok(GeneralizedEigenResult {
            eigenvalues,
            eigenvectors: result.eigenvectors,
            iterations: result.iterations,
            residual_norms: result.residual_norms,
            converged: result.converged,
            num_converged: result.num_converged,
        })
    }

    /// Buckling mode: work with (A - sigma*B)^{-1}*A.
    fn compute_buckling(
        &self,
        a: &CsrMatrix<T>,
        b: &CsrMatrix<T>,
        n: usize,
        nev: usize,
        ncv: usize,
        initial_vector: Option<&[T]>,
    ) -> Result<GeneralizedEigenResult<T>, EigenvalueError> {
        let sigma = self.config.sigma.clone();

        // Compute C = A - sigma*B and convert to CSC
        let c = subtract_scaled_matrices(a, b, sigma.clone());
        let c_csc = csr_to_csc(&c);

        let c_factor = SparseLU::new(&c_csc).map_err(|e| {
            EigenvalueError::ComputationError(format!(
                "Failed to factor (A - sigma*B): {:?}. Try a different shift.",
                e
            ))
        })?;

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

        // Convert B to CSC for Cholesky factorization
        let b_csc = csr_to_csc(b);

        // For buckling mode, we use A-inner product
        // Eigenvalues of (A - sigma*B)^{-1}*A are mu = lambda/(lambda - sigma)
        let b_factor = SparseCholesky::new(&b_csc)
            .map_err(|e| EigenvalueError::ComputationError(format!("B must be SPD: {:?}", e)))?;

        let result = self.lanczos_b_orthogonal(a, b, &b_factor, n, nev, ncv, v, |x| {
            // Operator: (A - sigma*B)^{-1}*A*x
            let mut ax = vec![T::zero(); n];
            spmv(T::one(), a, x, T::zero(), &mut ax);
            c_factor.solve(&ax)
        })?;

        // Transform eigenvalues back: mu = lambda/(lambda - sigma) => lambda = sigma*mu/(mu - 1)
        let eigenvalues: Vec<T> = result
            .eigenvalues
            .iter()
            .map(|mu| {
                let denom = mu.clone() - T::one();
                if Scalar::abs(denom.clone()) > <T as Scalar>::epsilon() {
                    sigma.clone() * mu.clone() / denom
                } else {
                    sigma.clone()
                }
            })
            .collect();

        Ok(GeneralizedEigenResult {
            eigenvalues,
            eigenvectors: result.eigenvectors,
            iterations: result.iterations,
            residual_norms: result.residual_norms,
            converged: result.converged,
            num_converged: result.num_converged,
        })
    }

    /// Cayley mode: work with (A - sigma*B)^{-1}*(A + sigma*B).
    fn compute_cayley(
        &self,
        a: &CsrMatrix<T>,
        b: &CsrMatrix<T>,
        n: usize,
        nev: usize,
        ncv: usize,
        initial_vector: Option<&[T]>,
    ) -> Result<GeneralizedEigenResult<T>, EigenvalueError> {
        let sigma = self.config.sigma.clone();

        // C1 = A - sigma*B, C2 = A + sigma*B
        let c1 = subtract_scaled_matrices(a, b, sigma.clone());
        let c1_csc = csr_to_csc(&c1);
        let c2 = add_scaled_matrices(a, b, sigma.clone());

        let c1_factor = SparseLU::new(&c1_csc).map_err(|e| {
            EigenvalueError::ComputationError(format!(
                "Failed to factor (A - sigma*B): {:?}. Try a different shift.",
                e
            ))
        })?;

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

        // Convert B to CSC for Cholesky factorization
        let b_csc = csr_to_csc(b);

        let b_factor = SparseCholesky::new(&b_csc)
            .map_err(|e| EigenvalueError::ComputationError(format!("B must be SPD: {:?}", e)))?;

        // Cayley transform: eigenvalues are mu = (lambda + sigma)/(lambda - sigma)
        let result = self.lanczos_b_orthogonal(a, b, &b_factor, n, nev, ncv, v, |x| {
            // Operator: (A - sigma*B)^{-1}*(A + sigma*B)*x
            let mut c2x = vec![T::zero(); n];
            spmv(T::one(), &c2, x, T::zero(), &mut c2x);
            c1_factor.solve(&c2x)
        })?;

        // Transform eigenvalues back: mu = (lambda + sigma)/(lambda - sigma) => lambda = sigma*(mu + 1)/(mu - 1)
        let eigenvalues: Vec<T> = result
            .eigenvalues
            .iter()
            .map(|mu| {
                let denom = mu.clone() - T::one();
                if Scalar::abs(denom.clone()) > <T as Scalar>::epsilon() {
                    sigma.clone() * (mu.clone() + T::one()) / denom
                } else {
                    sigma.clone()
                }
            })
            .collect();

        Ok(GeneralizedEigenResult {
            eigenvalues,
            eigenvectors: result.eigenvectors,
            iterations: result.iterations,
            residual_norms: result.residual_norms,
            converged: result.converged,
            num_converged: result.num_converged,
        })
    }

    /// B-orthogonal Lanczos iteration for symmetric generalized eigenvalue problems.
    ///
    /// Maintains V such that V^T * B * V = I (B-orthonormality).
    fn lanczos_b_orthogonal<F>(
        &self,
        _a: &CsrMatrix<T>,
        b: &CsrMatrix<T>,
        _b_factor: &SparseCholesky<T>,
        n: usize,
        nev: usize,
        ncv: usize,
        initial_v: Vec<T>,
        op: F,
    ) -> Result<GeneralizedEigenResult<T>, EigenvalueError>
    where
        F: Fn(&[T]) -> Vec<T>,
    {
        let p = ncv - nev;
        let two = T::from_f64(2.0).unwrap_or_else(T::one);

        // Storage for Lanczos vectors V (n x ncv)
        let mut v_storage: Vec<Vec<T>> = Vec::with_capacity(ncv + 1);

        // B-normalize initial vector
        let mut v = initial_v;
        let mut bv = vec![T::zero(); n];
        spmv(T::one(), b, &v, T::zero(), &mut bv);
        let b_norm = Real::sqrt(dot(&v, &bv));

        if b_norm <= <T as Scalar>::epsilon() {
            return Err(EigenvalueError::Breakdown {
                iteration: 0,
                description: "Initial vector is B-zero".to_string(),
            });
        }

        for i in 0..n {
            v[i] = v[i].clone() / b_norm.clone();
        }
        v_storage.push(v.clone());

        // Tridiagonal matrix: alpha (diagonal), beta (off-diagonal)
        let mut alpha = Vec::with_capacity(ncv);
        let mut beta = Vec::with_capacity(ncv);

        // Residual vector for next iteration
        let mut f = vec![T::zero(); n];
        let mut beta_prev = T::zero();

        // Outer restart loop
        for iter in 0..self.config.max_iterations {
            let k_start = v_storage.len() - 1;

            // Build Lanczos factorization from current size to ncv
            for j in k_start..ncv {
                // w = op(v_j)
                let w = op(&v_storage[j]);

                // Compute alpha_j = v_j^T * B * w
                let mut bw = vec![T::zero(); n];
                spmv(T::one(), b, &w, T::zero(), &mut bw);
                let alpha_j = dot(&v_storage[j], &bw);
                alpha.push(alpha_j.clone());

                // f = w - alpha_j * v_j - beta_{j-1} * v_{j-1}
                for i in 0..n {
                    f[i] = w[i].clone() - alpha_j.clone() * v_storage[j][i].clone();
                }
                if j > 0 {
                    for i in 0..n {
                        f[i] = f[i].clone() - beta_prev.clone() * v_storage[j - 1][i].clone();
                    }
                }

                // Re-orthogonalize against all previous vectors (full reorthogonalization)
                for k in 0..=j {
                    let mut bvk = vec![T::zero(); n];
                    spmv(T::one(), b, &v_storage[k], T::zero(), &mut bvk);
                    let h_kj = dot(&f, &bvk);
                    for i in 0..n {
                        f[i] = f[i].clone() - h_kj.clone() * v_storage[k][i].clone();
                    }
                }

                // Compute B-norm of f
                let mut bf = vec![T::zero(); n];
                spmv(T::one(), b, &f, T::zero(), &mut bf);
                let beta_j = Real::sqrt(dot(&f, &bf));

                if j < ncv - 1 {
                    beta.push(beta_j.clone());
                    beta_prev = beta_j.clone();

                    if beta_j > <T as Scalar>::epsilon() {
                        let v_next: Vec<T> =
                            f.iter().map(|fi| fi.clone() / beta_j.clone()).collect();
                        v_storage.push(v_next);
                    } else {
                        // Lucky breakdown - invariant subspace found
                        break;
                    }
                }
            }

            // Compute eigenvalues of tridiagonal matrix T
            let m = alpha.len();
            let (ritz_values, ritz_vectors) = Self::symmetric_tridiag_qr(&alpha, &beta, m);

            // Sort eigenvalues according to which we want
            let mut indices: Vec<usize> = (0..m).collect();
            match self.config.which {
                WhichEigenvalues::LargestMagnitude => {
                    indices.sort_by(|&i, &j| {
                        let ai = Scalar::abs(ritz_values[i].clone());
                        let aj = Scalar::abs(ritz_values[j].clone());
                        aj.partial_cmp(&ai).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
                WhichEigenvalues::SmallestMagnitude => {
                    indices.sort_by(|&i, &j| {
                        let ai = Scalar::abs(ritz_values[i].clone());
                        let aj = Scalar::abs(ritz_values[j].clone());
                        ai.partial_cmp(&aj).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
                WhichEigenvalues::LargestAlgebraic | WhichEigenvalues::NearTarget => {
                    indices.sort_by(|&i, &j| {
                        ritz_values[j]
                            .partial_cmp(&ritz_values[i])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
                WhichEigenvalues::SmallestAlgebraic => {
                    indices.sort_by(|&i, &j| {
                        ritz_values[i]
                            .partial_cmp(&ritz_values[j])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
            }

            // Compute residual norms for wanted Ritz pairs
            let mut residual_norms = Vec::with_capacity(nev);
            let mut converged_count = 0;

            let last_beta = if !beta.is_empty() {
                beta[beta.len() - 1].clone()
            } else {
                T::zero()
            };

            for &idx in indices.iter().take(nev) {
                // Residual norm = |beta_{m-1}| * |last component of Ritz vector|
                let last_comp = if idx < ritz_vectors.len() && !ritz_vectors[idx].is_empty() {
                    Scalar::abs(ritz_vectors[idx][m - 1].clone())
                } else {
                    T::one()
                };
                let res = last_beta.clone() * last_comp;
                residual_norms.push(res.clone());

                if res <= self.config.tolerance {
                    converged_count += 1;
                }
            }

            // Check convergence
            if converged_count >= nev {
                // Extract converged eigenvalues and eigenvectors
                let eigenvalues: Vec<T> = indices
                    .iter()
                    .take(nev)
                    .map(|&i| ritz_values[i].clone())
                    .collect();

                let eigenvectors = if self.config.compute_eigenvectors {
                    let mut evecs = Vec::with_capacity(nev);
                    for &idx in indices.iter().take(nev) {
                        // Compute eigenvector: x = V * y where y is Ritz vector
                        let mut x = vec![T::zero(); n];
                        for j in 0..m.min(v_storage.len()) {
                            let coef = if idx < ritz_vectors.len() && j < ritz_vectors[idx].len() {
                                ritz_vectors[idx][j].clone()
                            } else {
                                T::zero()
                            };
                            for i in 0..n {
                                x[i] = x[i].clone() + coef.clone() * v_storage[j][i].clone();
                            }
                        }
                        evecs.push(x);
                    }
                    Some(evecs)
                } else {
                    None
                };

                return Ok(GeneralizedEigenResult {
                    eigenvalues,
                    eigenvectors,
                    iterations: iter + 1,
                    residual_norms,
                    converged: true,
                    num_converged: converged_count,
                });
            }

            // Implicit restart: apply p = ncv - nev shifts
            // Using the unwanted Ritz values as shifts

            // Get unwanted Ritz values (those we don't want)
            let shifts: Vec<T> = indices
                .iter()
                .skip(nev)
                .take(p)
                .map(|&i| ritz_values[i].clone())
                .collect();

            if shifts.is_empty() {
                break;
            }

            // Apply implicit QR shifts to the tridiagonal matrix
            // This compresses the factorization to dimension nev

            // First, we need to apply shifted QR steps
            let mut q_total = vec![vec![T::zero(); m]; m];
            for i in 0..m {
                q_total[i][i] = T::one();
            }

            for shift in &shifts {
                // Compute one step of shifted QR on tridiagonal matrix
                let (q, new_alpha, new_beta) = Self::tridiag_qr_step(&alpha, &beta, shift.clone());

                // Update tridiagonal elements
                for i in 0..new_alpha.len() {
                    alpha[i] = new_alpha[i].clone();
                }
                for i in 0..new_beta.len() {
                    beta[i] = new_beta[i].clone();
                }

                // Accumulate Q
                let mut q_new = vec![vec![T::zero(); m]; m];
                for i in 0..m {
                    for j in 0..m {
                        for k in 0..m {
                            if k < q.len() && j < q[k].len() {
                                q_new[i][j] =
                                    q_new[i][j].clone() + q_total[i][k].clone() * q[k][j].clone();
                            }
                        }
                    }
                }
                q_total = q_new;
            }

            // Update V = V * Q and truncate to nev+1 vectors
            let keep = nev + 1;
            let mut v_new: Vec<Vec<T>> = Vec::with_capacity(keep);

            for j in 0..keep.min(m) {
                let mut v_j = vec![T::zero(); n];
                for k in 0..m.min(v_storage.len()) {
                    for i in 0..n {
                        v_j[i] = v_j[i].clone() + q_total[k][j].clone() * v_storage[k][i].clone();
                    }
                }
                // Re-B-normalize
                let mut bvj = vec![T::zero(); n];
                spmv(T::one(), b, &v_j, T::zero(), &mut bvj);
                let b_norm_j = Real::sqrt(dot(&v_j, &bvj));
                if b_norm_j > <T as Scalar>::epsilon() {
                    for i in 0..n {
                        v_j[i] = v_j[i].clone() / b_norm_j.clone();
                    }
                }
                v_new.push(v_j);
            }

            v_storage = v_new;

            // Truncate alpha and beta
            alpha.truncate(nev);
            beta.truncate(nev.saturating_sub(1));

            // Update beta_prev for continuation
            beta_prev = if !beta.is_empty() {
                beta[beta.len() - 1].clone()
            } else {
                T::zero()
            };

            // The residual for continuation is scaled by the (nev+1,nev) element
            // of the Q-transformed matrix
            if let Some(last_v) = v_storage.last() {
                let op_v = op(last_v);
                let mut b_op_v = vec![T::zero(); n];
                spmv(T::one(), b, &op_v, T::zero(), &mut b_op_v);

                // Compute continuation coefficient
                if nev < v_storage.len() {
                    let cont_coef = dot(&v_storage[nev - 1], &b_op_v);
                    beta_prev = cont_coef.clone();
                    let beta_len = beta.len();
                    if beta_len > 0 {
                        beta[beta_len - 1] = cont_coef;
                    }
                }
            }

            // Update f for continuation
            if v_storage.len() > nev {
                let last = v_storage.len() - 1;
                f = v_storage[last].clone();
                for i in 0..n {
                    f[i] = f[i].clone() * (beta_prev.clone() * two.clone());
                }
            }
        }

        // Max iterations reached
        let m = alpha.len();
        let (ritz_values, _ritz_vectors) = Self::symmetric_tridiag_qr(&alpha, &beta, m);

        let mut indices: Vec<usize> = (0..m).collect();
        match self.config.which {
            WhichEigenvalues::LargestMagnitude => {
                indices.sort_by(|&i, &j| {
                    let ai = Scalar::abs(ritz_values[i].clone());
                    let aj = Scalar::abs(ritz_values[j].clone());
                    aj.partial_cmp(&ai).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            _ => {
                indices.sort_by(|&i, &j| {
                    ritz_values[i]
                        .partial_cmp(&ritz_values[j])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        let eigenvalues: Vec<T> = indices
            .iter()
            .take(nev)
            .map(|&i| ritz_values[i].clone())
            .collect();

        Err(EigenvalueError::MaxIterations {
            iterations: self.config.max_iterations,
            converged_count: eigenvalues.len().min(nev),
        })
    }

    /// Arnoldi iteration for non-symmetric generalized problems.
    fn arnoldi_generalized<F>(
        &self,
        _a: &CsrMatrix<T>,
        b: &CsrMatrix<T>,
        _b_factor: &SparseCholesky<T>,
        n: usize,
        nev: usize,
        ncv: usize,
        initial_v: Vec<T>,
        op: F,
    ) -> Result<GeneralizedEigenResult<T>, EigenvalueError>
    where
        F: Fn(&[T]) -> Vec<T>,
    {
        // For non-symmetric case, we use standard Arnoldi but with B-inner product
        // This is simplified - for full generality would need more sophisticated approach

        let mut v_storage: Vec<Vec<T>> = Vec::with_capacity(ncv + 1);

        // B-normalize initial vector
        let mut v = initial_v;
        let mut bv = vec![T::zero(); n];
        spmv(T::one(), b, &v, T::zero(), &mut bv);
        let b_norm = Real::sqrt(dot(&v, &bv));

        if b_norm <= <T as Scalar>::epsilon() {
            return Err(EigenvalueError::Breakdown {
                iteration: 0,
                description: "Initial vector is B-zero".to_string(),
            });
        }

        for i in 0..n {
            v[i] = v[i].clone() / b_norm.clone();
        }
        v_storage.push(v);

        // Upper Hessenberg matrix H
        let mut h: Vec<Vec<T>> = Vec::with_capacity(ncv);

        for _iter in 0..self.config.max_iterations {
            // Build Arnoldi factorization
            for j in v_storage.len() - 1..ncv {
                let w = op(&v_storage[j]);

                // Modified Gram-Schmidt with B-inner product
                let mut h_col = vec![T::zero(); j + 2];
                let mut w_orth = w.clone();

                for i in 0..=j {
                    let mut bvi = vec![T::zero(); n];
                    spmv(T::one(), b, &v_storage[i], T::zero(), &mut bvi);
                    h_col[i] = dot(&w_orth, &bvi);

                    for k in 0..n {
                        w_orth[k] = w_orth[k].clone() - h_col[i].clone() * v_storage[i][k].clone();
                    }
                }

                // B-norm of orthogonalized vector
                let mut bw = vec![T::zero(); n];
                spmv(T::one(), b, &w_orth, T::zero(), &mut bw);
                h_col[j + 1] = Real::sqrt(dot(&w_orth, &bw));

                if j < h.len() {
                    h[j] = h_col.clone();
                } else {
                    h.push(h_col.clone());
                }

                if h_col[j + 1] > <T as Scalar>::epsilon() {
                    let v_new: Vec<T> = w_orth
                        .iter()
                        .map(|wi| wi.clone() / h_col[j + 1].clone())
                        .collect();
                    v_storage.push(v_new);
                } else {
                    break;
                }
            }

            // Compute eigenvalues of H
            let m = h.len();
            let (eig_real, eig_imag, evecs) = Self::hessenberg_qr(&h, m);

            // For now, just use real parts (assuming real eigenvalues for symmetric problems)
            let mut indices: Vec<usize> = (0..m).collect();
            match self.config.which {
                WhichEigenvalues::LargestMagnitude => {
                    indices.sort_by(|&i, &j| {
                        let ai = Real::sqrt(
                            eig_real[i].clone() * eig_real[i].clone()
                                + eig_imag[i].clone() * eig_imag[i].clone(),
                        );
                        let aj = Real::sqrt(
                            eig_real[j].clone() * eig_real[j].clone()
                                + eig_imag[j].clone() * eig_imag[j].clone(),
                        );
                        aj.partial_cmp(&ai).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
                _ => {
                    indices.sort_by(|&i, &j| {
                        eig_real[i]
                            .partial_cmp(&eig_real[j])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
            }

            // Check convergence using residual bounds
            let last_h = if m > 0 && !h[m - 1].is_empty() {
                Scalar::abs(h[m - 1][m].clone())
            } else {
                T::one()
            };

            let mut converged_count = 0;
            let mut residual_norms = Vec::with_capacity(nev);

            for &idx in indices.iter().take(nev) {
                let last_comp = if idx < evecs.len() && m > 0 && m - 1 < evecs[idx].len() {
                    Scalar::abs(evecs[idx][m - 1].clone())
                } else {
                    T::one()
                };
                let res = last_h.clone() * last_comp;
                residual_norms.push(res.clone());
                if res <= self.config.tolerance {
                    converged_count += 1;
                }
            }

            if converged_count >= nev {
                let eigenvalues: Vec<T> = indices
                    .iter()
                    .take(nev)
                    .map(|&i| eig_real[i].clone())
                    .collect();

                let eigenvectors = if self.config.compute_eigenvectors {
                    let mut evs = Vec::with_capacity(nev);
                    for &idx in indices.iter().take(nev) {
                        let mut x = vec![T::zero(); n];
                        for j in 0..m.min(v_storage.len()) {
                            let coef = if idx < evecs.len() && j < evecs[idx].len() {
                                evecs[idx][j].clone()
                            } else {
                                T::zero()
                            };
                            for i in 0..n {
                                x[i] = x[i].clone() + coef.clone() * v_storage[j][i].clone();
                            }
                        }
                        evs.push(x);
                    }
                    Some(evs)
                } else {
                    None
                };

                return Ok(GeneralizedEigenResult {
                    eigenvalues,
                    eigenvectors,
                    iterations: 1,
                    residual_norms,
                    converged: true,
                    num_converged: converged_count,
                });
            }

            // For simplicity, restart from scratch with a new random vector
            // A full implementation would use implicit restarts
            break;
        }

        Err(EigenvalueError::MaxIterations {
            iterations: self.config.max_iterations,
            converged_count: 0,
        })
    }

    /// QR algorithm for symmetric tridiagonal matrix.
    fn symmetric_tridiag_qr(alpha: &[T], beta: &[T], m: usize) -> (Vec<T>, Vec<Vec<T>>) {
        if m == 0 {
            return (vec![], vec![]);
        }

        // Copy tridiagonal elements
        let mut d = alpha[..m].to_vec();
        let mut e = if beta.len() >= m - 1 {
            beta[..m - 1].to_vec()
        } else {
            let mut e = beta.to_vec();
            e.resize(m.saturating_sub(1), T::zero());
            e
        };

        // Initialize eigenvector matrix to identity
        let mut z: Vec<Vec<T>> = (0..m)
            .map(|i| {
                let mut row = vec![T::zero(); m];
                row[i] = T::one();
                row
            })
            .collect();

        let max_iter = 30 * m;
        let eps = <T as Scalar>::epsilon();
        let two = T::from_f64(2.0).unwrap_or_else(T::one);

        for _iter in 0..max_iter {
            // Find unreduced block
            let mut l = 0;
            while l < m - 1 {
                if Scalar::abs(e[l].clone())
                    <= eps.clone() * (Scalar::abs(d[l].clone()) + Scalar::abs(d[l + 1].clone()))
                {
                    e[l] = T::zero();
                    l += 1;
                } else {
                    break;
                }
            }

            if l >= m - 1 {
                break;
            }

            // Find end of unreduced block
            let mut n_end = l + 1;
            while n_end < m - 1 {
                if Scalar::abs(e[n_end].clone())
                    <= eps.clone()
                        * (Scalar::abs(d[n_end].clone()) + Scalar::abs(d[n_end + 1].clone()))
                {
                    e[n_end] = T::zero();
                    break;
                }
                n_end += 1;
            }

            // Wilkinson shift
            let dd = (d[n_end].clone() - d[n_end - 1].clone()) / two.clone();
            let ee = e[n_end - 1].clone();
            let shift = if Scalar::abs(dd.clone()) > eps.clone() {
                let sign = if dd >= T::zero() {
                    T::one()
                } else {
                    T::zero() - T::one()
                };
                d[n_end].clone()
                    - ee.clone() * ee.clone()
                        / (dd.clone()
                            + sign * Real::sqrt(dd.clone() * dd.clone() + ee.clone() * ee.clone()))
            } else {
                d[n_end].clone() - Scalar::abs(ee.clone())
            };

            // Implicit QR step
            let mut x = d[l].clone() - shift.clone();
            let mut z_val = e[l].clone();

            for k in l..n_end {
                // Givens rotation to zero out z_val
                let r = Real::sqrt(x.clone() * x.clone() + z_val.clone() * z_val.clone());
                let c = if r > eps.clone() {
                    x.clone() / r.clone()
                } else {
                    T::one()
                };
                let s = if r > eps.clone() {
                    z_val.clone() / r.clone()
                } else {
                    T::zero()
                };

                // Update tridiagonal matrix
                if k > l {
                    e[k - 1] = r.clone();
                }

                let d_k = d[k].clone();
                let d_k1 = d[k + 1].clone();
                let e_k = if k < m - 1 { e[k].clone() } else { T::zero() };

                d[k] = c.clone() * c.clone() * d_k.clone()
                    + two.clone() * c.clone() * s.clone() * e_k.clone()
                    + s.clone() * s.clone() * d_k1.clone();
                d[k + 1] = s.clone() * s.clone() * d_k.clone()
                    - two.clone() * c.clone() * s.clone() * e_k.clone()
                    + c.clone() * c.clone() * d_k1.clone();

                if k < m - 1 {
                    e[k] = c.clone() * s.clone() * (d_k.clone() - d_k1.clone())
                        + (c.clone() * c.clone() - s.clone() * s.clone()) * e_k.clone();
                }

                // Prepare for next iteration
                if k < n_end - 1 {
                    x = e[k].clone();
                    z_val = T::zero() - s.clone() * e[k + 1].clone();
                    e[k + 1] = c.clone() * e[k + 1].clone();
                }

                // Update eigenvectors
                for i in 0..m {
                    let temp = c.clone() * z[i][k].clone() + s.clone() * z[i][k + 1].clone();
                    z[i][k + 1] =
                        T::zero() - s.clone() * z[i][k].clone() + c.clone() * z[i][k + 1].clone();
                    z[i][k] = temp;
                }
            }
        }

        // Sort eigenvalues and eigenvectors
        let mut indices: Vec<usize> = (0..m).collect();
        indices.sort_by(|&i, &j| d[i].partial_cmp(&d[j]).unwrap_or(std::cmp::Ordering::Equal));

        let eigenvalues: Vec<T> = indices.iter().map(|&i| d[i].clone()).collect();
        let eigenvectors: Vec<Vec<T>> = indices
            .iter()
            .map(|&idx| (0..m).map(|i| z[i][idx].clone()).collect())
            .collect();

        (eigenvalues, eigenvectors)
    }

    /// Single implicit QR step on tridiagonal matrix with shift.
    fn tridiag_qr_step(alpha: &[T], beta: &[T], shift: T) -> (Vec<Vec<T>>, Vec<T>, Vec<T>) {
        let m = alpha.len();
        if m == 0 {
            return (vec![], vec![], vec![]);
        }

        let mut d = alpha.to_vec();
        let mut e = beta.to_vec();
        e.resize(m.saturating_sub(1), T::zero());

        // Q accumulator
        let mut q: Vec<Vec<T>> = (0..m)
            .map(|i| {
                let mut row = vec![T::zero(); m];
                row[i] = T::one();
                row
            })
            .collect();

        let eps = <T as Scalar>::epsilon();
        let two = T::from_f64(2.0).unwrap_or_else(T::one);

        // Apply shift
        d[0] = d[0].clone() - shift.clone();
        if m > 1 {
            d[m - 1] = d[m - 1].clone() - shift.clone();
        }

        let mut x = d[0].clone();
        let mut z_val = if !e.is_empty() {
            e[0].clone()
        } else {
            T::zero()
        };

        for k in 0..m.saturating_sub(1) {
            let r = Real::sqrt(x.clone() * x.clone() + z_val.clone() * z_val.clone());
            let c = if r > eps.clone() {
                x.clone() / r.clone()
            } else {
                T::one()
            };
            let s = if r > eps.clone() {
                z_val.clone() / r.clone()
            } else {
                T::zero()
            };

            if k > 0 {
                e[k - 1] = r.clone();
            }

            let d_k = d[k].clone() + shift.clone();
            let d_k1 = d[k + 1].clone() + shift.clone();
            let e_k = if k < e.len() { e[k].clone() } else { T::zero() };

            d[k] = c.clone() * c.clone() * d_k.clone()
                + two.clone() * c.clone() * s.clone() * e_k.clone()
                + s.clone() * s.clone() * d_k1.clone()
                - shift.clone();
            d[k + 1] = s.clone() * s.clone() * d_k.clone()
                - two.clone() * c.clone() * s.clone() * e_k.clone()
                + c.clone() * c.clone() * d_k1.clone()
                - shift.clone();

            if k < e.len() {
                e[k] = c.clone() * s.clone() * (d_k - d_k1)
                    + (c.clone() * c.clone() - s.clone() * s.clone()) * e_k;
            }

            if k + 1 < e.len() {
                x = e[k].clone();
                z_val = T::zero() - s.clone() * e[k + 1].clone();
                e[k + 1] = c.clone() * e[k + 1].clone();
            }

            // Update Q
            for i in 0..m {
                let temp = c.clone() * q[i][k].clone() + s.clone() * q[i][k + 1].clone();
                q[i][k + 1] =
                    T::zero() - s.clone() * q[i][k].clone() + c.clone() * q[i][k + 1].clone();
                q[i][k] = temp;
            }
        }

        (q, d, e)
    }

    /// QR algorithm for upper Hessenberg matrix (general non-symmetric case).
    fn hessenberg_qr(h: &[Vec<T>], m: usize) -> (Vec<T>, Vec<T>, Vec<Vec<T>>) {
        if m == 0 {
            return (vec![], vec![], vec![]);
        }

        // Copy H matrix to a full m x m matrix
        let mut h_mat: Vec<Vec<T>> = vec![vec![T::zero(); m]; m];
        for j in 0..m {
            if j < h.len() {
                for i in 0..h[j].len().min(m) {
                    h_mat[i][j] = h[j][i].clone();
                }
            }
        }

        // Eigenvector accumulator
        let mut z: Vec<Vec<T>> = (0..m)
            .map(|i| {
                let mut row = vec![T::zero(); m];
                row[i] = T::one();
                row
            })
            .collect();

        let eps = <T as Scalar>::epsilon();
        let max_iter = 30 * m;

        for _iter in 0..max_iter {
            // Check for convergence
            let mut converged = true;
            for i in 1..m {
                if Scalar::abs(h_mat[i][i - 1].clone())
                    > eps.clone()
                        * (Scalar::abs(h_mat[i - 1][i - 1].clone())
                            + Scalar::abs(h_mat[i][i].clone()))
                {
                    converged = false;
                    break;
                }
            }
            if converged {
                break;
            }

            // Single QR step (Francis double shift for real matrices)
            // Simplified: using single shift for now
            let shift = h_mat[m - 1][m - 1].clone();

            // Apply shift
            for i in 0..m {
                h_mat[i][i] = h_mat[i][i].clone() - shift.clone();
            }

            // QR decomposition using Givens rotations
            for k in 0..m - 1 {
                let a = h_mat[k][k].clone();
                let b = h_mat[k + 1][k].clone();
                let r = Real::sqrt(a.clone() * a.clone() + b.clone() * b.clone());

                if r <= eps.clone() {
                    continue;
                }

                let c = a / r.clone();
                let s = b / r.clone();

                // Apply Givens rotation to H from left
                for j in k..m {
                    let temp =
                        c.clone() * h_mat[k][j].clone() + s.clone() * h_mat[k + 1][j].clone();
                    h_mat[k + 1][j] = T::zero() - s.clone() * h_mat[k][j].clone()
                        + c.clone() * h_mat[k + 1][j].clone();
                    h_mat[k][j] = temp;
                }

                // Apply Givens rotation to H from right
                for i in 0..(k + 2).min(m) {
                    let temp =
                        c.clone() * h_mat[i][k].clone() + s.clone() * h_mat[i][k + 1].clone();
                    h_mat[i][k + 1] = T::zero() - s.clone() * h_mat[i][k].clone()
                        + c.clone() * h_mat[i][k + 1].clone();
                    h_mat[i][k] = temp;
                }

                // Accumulate eigenvectors
                for i in 0..m {
                    let temp = c.clone() * z[i][k].clone() + s.clone() * z[i][k + 1].clone();
                    z[i][k + 1] =
                        T::zero() - s.clone() * z[i][k].clone() + c.clone() * z[i][k + 1].clone();
                    z[i][k] = temp;
                }
            }

            // Remove shift
            for i in 0..m {
                h_mat[i][i] = h_mat[i][i].clone() + shift.clone();
            }
        }

        // Extract eigenvalues from diagonal
        let eig_real: Vec<T> = (0..m).map(|i| h_mat[i][i].clone()).collect();
        let eig_imag: Vec<T> = vec![T::zero(); m]; // Assuming real for now

        // Eigenvectors
        let eigenvectors: Vec<Vec<T>> = (0..m)
            .map(|j| (0..m).map(|i| z[i][j].clone()).collect())
            .collect();

        (eig_real, eig_imag, eigenvectors)
    }
}
