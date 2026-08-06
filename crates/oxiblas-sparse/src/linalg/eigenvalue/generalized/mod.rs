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
    /// Computed eigenvalues (real parts).
    ///
    /// For a real non-symmetric pencil `(A, B)` the eigenvalues may occur in
    /// complex-conjugate pairs; the imaginary parts are reported in
    /// [`GeneralizedEigenResult::eigenvalues_imag`]. For symmetric problems all
    /// imaginary parts are zero.
    pub eigenvalues: Vec<T>,
    /// Imaginary parts of the computed eigenvalues.
    ///
    /// Non-zero entries indicate genuinely complex eigenvalues of a real
    /// non-symmetric pencil (recovered from the 2x2 blocks of the real Schur
    /// form). Complex eigenvalues appear as conjugate pairs `a +/- b i`.
    pub eigenvalues_imag: Vec<T>,
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
/// ```
/// use oxiblas_sparse::csr::CsrMatrix;
/// use oxiblas_sparse::linalg::eigenvalue::{
///     GeneralizedEigen, GeneralizedEigenConfig, GeneralizedMode, WhichEigenvalues,
/// };
///
/// // A = stiffness matrix (diagonal, eigenvalues 1, 2, 3), B = mass matrix (identity).
/// // With B = I the generalized problem A*x = lambda*B*x reduces to A's own
/// // eigenvalues, which makes the expected result easy to check.
/// let a = CsrMatrix::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0, 2.0, 3.0]).unwrap();
/// let b = CsrMatrix::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0, 1.0, 1.0]).unwrap(); // SPD
///
/// let config = GeneralizedEigenConfig {
///     num_eigenvalues: 2,
///     which: WhichEigenvalues::SmallestMagnitude,
///     mode: GeneralizedMode::ShiftInvert,
///     sigma: 0.0, // Find smallest eigenvalues
///     symmetric: true,
///     ..Default::default()
/// };
///
/// let solver = GeneralizedEigen::new(config);
/// let result = solver.compute(&a, &b, None)?;
/// assert_eq!(result.eigenvalues.len(), 2);
/// # Ok::<(), oxiblas_sparse::linalg::EigenvalueError>(())
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
        let b_csc = csr_to_csc(b)?;

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

        // For standard mode the operator is B^{-1}*A, whose eigenvalues equal the
        // generalized eigenvalues lambda directly (mu = lambda), so the mu -> lambda
        // transform passed to the inner solvers is the identity.
        if self.config.symmetric {
            self.lanczos_b_orthogonal(
                a,
                b,
                &b_factor,
                n,
                nev,
                ncv,
                v,
                |x| {
                    // Operator: B^{-1}*A*x
                    let mut ax = vec![T::zero(); n];
                    spmv(T::one(), a, x, T::zero(), &mut ax);
                    b_factor.solve(&ax)
                },
                |mu: &T| mu.clone(),
            )
        } else {
            // For non-symmetric, use general Arnoldi
            self.arnoldi_generalized(
                a,
                b,
                &b_factor,
                n,
                nev,
                ncv,
                v,
                |x| {
                    let mut ax = vec![T::zero(); n];
                    spmv(T::one(), a, x, T::zero(), &mut ax);
                    b_factor.solve(&ax)
                },
                |mu_re: &T, mu_im: &T| (mu_re.clone(), mu_im.clone()),
            )
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
        let c = subtract_scaled_matrices(a, b, sigma.clone())?;
        let c_csc = csr_to_csc(&c)?;

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
        let b_csc = csr_to_csc(b)?;

        // For shift-invert, eigenvalues of op are mu = 1/(lambda - sigma), so the
        // generalized eigenvalues are lambda = sigma + 1/mu. The transform is supplied
        // to the inner solver so that the returned eigenvalues and the residuals
        // ||A x - lambda B x|| are expressed in terms of the original pencil (A, B).
        if self.config.symmetric {
            // Factor B for B-inner product
            let b_factor = SparseCholesky::new(&b_csc).map_err(|e| {
                EigenvalueError::ComputationError(format!("B must be SPD: {:?}", e))
            })?;
            let sigma_t = sigma.clone();

            self.lanczos_b_orthogonal(
                a,
                b,
                &b_factor,
                n,
                nev,
                ncv,
                v,
                |x| {
                    // Operator: (A - sigma*B)^{-1}*B*x
                    let mut bx = vec![T::zero(); n];
                    spmv(T::one(), b, x, T::zero(), &mut bx);
                    c_factor.solve(&bx)
                },
                move |mu: &T| {
                    if Scalar::abs(mu.clone()) > <T as Scalar>::epsilon() {
                        sigma_t.clone() + T::one() / mu.clone()
                    } else {
                        // Eigenvalue at infinity - shouldn't happen for well-posed problems
                        sigma_t.clone()
                    }
                },
            )
        } else {
            let b_factor = SparseCholesky::new(&b_csc).map_err(|e| {
                EigenvalueError::ComputationError(format!("B must be SPD: {:?}", e))
            })?;
            let sigma_t = sigma.clone();

            self.arnoldi_generalized(
                a,
                b,
                &b_factor,
                n,
                nev,
                ncv,
                v,
                |x| {
                    let mut bx = vec![T::zero(); n];
                    spmv(T::one(), b, x, T::zero(), &mut bx);
                    c_factor.solve(&bx)
                },
                move |mu_re: &T, mu_im: &T| {
                    // lambda = sigma + 1/mu using the complex reciprocal.
                    let den = mu_re.clone() * mu_re.clone() + mu_im.clone() * mu_im.clone();
                    if den > <T as Scalar>::epsilon() {
                        (
                            sigma_t.clone() + mu_re.clone() / den.clone(),
                            T::zero() - mu_im.clone() / den,
                        )
                    } else {
                        (sigma_t.clone(), T::zero())
                    }
                },
            )
        }
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
        let c = subtract_scaled_matrices(a, b, sigma.clone())?;
        let c_csc = csr_to_csc(&c)?;

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
        let b_csc = csr_to_csc(b)?;

        // For buckling mode, we use A-inner product
        // Eigenvalues of (A - sigma*B)^{-1}*A are mu = lambda/(lambda - sigma)
        let b_factor = SparseCholesky::new(&b_csc)
            .map_err(|e| EigenvalueError::ComputationError(format!("B must be SPD: {:?}", e)))?;

        // Buckling: mu = lambda/(lambda - sigma) => lambda = sigma*mu/(mu - 1). The
        // transform is applied inside the solver so that the returned eigenvalues and
        // the residuals ||A x - lambda B x|| refer to the original pencil (A, B).
        let sigma_t = sigma.clone();
        self.lanczos_b_orthogonal(
            a,
            b,
            &b_factor,
            n,
            nev,
            ncv,
            v,
            |x| {
                // Operator: (A - sigma*B)^{-1}*A*x
                let mut ax = vec![T::zero(); n];
                spmv(T::one(), a, x, T::zero(), &mut ax);
                c_factor.solve(&ax)
            },
            move |mu: &T| {
                let denom = mu.clone() - T::one();
                if Scalar::abs(denom.clone()) > <T as Scalar>::epsilon() {
                    sigma_t.clone() * mu.clone() / denom
                } else {
                    sigma_t.clone()
                }
            },
        )
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
        let c1 = subtract_scaled_matrices(a, b, sigma.clone())?;
        let c1_csc = csr_to_csc(&c1)?;
        let c2 = add_scaled_matrices(a, b, sigma.clone())?;

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
        let b_csc = csr_to_csc(b)?;

        let b_factor = SparseCholesky::new(&b_csc)
            .map_err(|e| EigenvalueError::ComputationError(format!("B must be SPD: {:?}", e)))?;

        // Cayley transform: eigenvalues are mu = (lambda + sigma)/(lambda - sigma)
        // Cayley: mu = (lambda + sigma)/(lambda - sigma) => lambda = sigma*(mu + 1)/(mu - 1).
        // The transform is applied inside the solver so that the returned eigenvalues and
        // the residuals ||A x - lambda B x|| refer to the original pencil (A, B).
        let sigma_t = sigma.clone();
        self.lanczos_b_orthogonal(
            a,
            b,
            &b_factor,
            n,
            nev,
            ncv,
            v,
            |x| {
                // Operator: (A - sigma*B)^{-1}*(A + sigma*B)*x
                let mut c2x = vec![T::zero(); n];
                spmv(T::one(), &c2, x, T::zero(), &mut c2x);
                c1_factor.solve(&c2x)
            },
            move |mu: &T| {
                let denom = mu.clone() - T::one();
                if Scalar::abs(denom.clone()) > <T as Scalar>::epsilon() {
                    sigma_t.clone() * (mu.clone() + T::one()) / denom
                } else {
                    sigma_t.clone()
                }
            },
        )
    }

    /// B-orthogonal Lanczos iteration for symmetric generalized eigenvalue problems.
    ///
    /// Maintains V such that V^T * B * V = I (B-orthonormality).
    fn lanczos_b_orthogonal<F, L>(
        &self,
        a: &CsrMatrix<T>,
        b: &CsrMatrix<T>,
        _b_factor: &SparseCholesky<T>,
        n: usize,
        nev: usize,
        ncv: usize,
        initial_v: Vec<T>,
        op: F,
        lambda_of_mu: L,
    ) -> Result<GeneralizedEigenResult<T>, EigenvalueError>
    where
        F: Fn(&[T]) -> Vec<T>,
        L: Fn(&T) -> T,
    {
        let p = ncv - nev;

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
        // B-norm of the current residual f (the true Arnoldi/Lanczos residual bound).
        let mut f_bnorm = T::zero();

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

                // Compute B-norm of f (the residual norm of the current factorization).
                let mut bf = vec![T::zero(); n];
                spmv(T::one(), b, &f, T::zero(), &mut bf);
                let beta_j = Real::sqrt(dot(&f, &bf));
                f_bnorm = beta_j.clone();

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
            let (ritz_values, ritz_vectors) = self.symmetric_tridiag_qr(&alpha, &beta, m);

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

            // Convergence is judged by the standard Lanczos residual bound for the Ritz
            // pairs of the (transformed) operator: ||f||_B * |last component of the Ritz
            // vector|, where ||f||_B is the B-norm of the current residual. This is the
            // correct, cheap convergence indicator for the Krylov iteration itself; the
            // residual actually reported to the caller is the documented
            // ||A x - lambda B x|| computed below against the original pencil.
            let mut converged_count = 0;
            let last_beta = f_bnorm.clone();

            for &idx in indices.iter().take(nev) {
                let last_comp = if idx < ritz_vectors.len() && !ritz_vectors[idx].is_empty() {
                    Scalar::abs(ritz_vectors[idx][m - 1].clone())
                } else {
                    T::one()
                };
                let res = last_beta.clone() * last_comp;
                let scale = Scalar::abs(ritz_values[idx].clone()) + T::one();
                if res <= self.config.tolerance * scale {
                    converged_count += 1;
                }
            }

            // Check convergence
            if converged_count >= nev {
                // Build the eigenvectors x = V * y in the original coordinates. They are
                // needed both for the (optionally returned) eigenvectors and for the true
                // residual ||A x - lambda B x||.
                let mut evecs = Vec::with_capacity(nev);
                for &idx in indices.iter().take(nev) {
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

                // Transform the operator Ritz values mu back to the generalized
                // eigenvalues lambda, and compute the documented residual norm
                // ||A x - lambda B x|| against the original matrices A and B.
                let mut eigenvalues = Vec::with_capacity(nev);
                let mut residual_norms = Vec::with_capacity(nev);
                for (k, &idx) in indices.iter().take(nev).enumerate() {
                    let lambda = lambda_of_mu(&ritz_values[idx]);
                    let res = Self::generalized_residual(a, b, &lambda, &evecs[k], n);
                    eigenvalues.push(lambda);
                    residual_norms.push(res);
                }

                let eigenvectors = if self.config.compute_eigenvectors {
                    Some(evecs)
                } else {
                    None
                };

                return Ok(GeneralizedEigenResult {
                    eigenvalues,
                    eigenvalues_imag: vec![T::zero(); nev],
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

            // Correct implicit-restart continuation (Sorensen). The residual of the
            // compressed length-nev factorization is
            //     f^+ = beta_k^+ * v^+_{nev} + sigma * f,
            // where beta_k^+ = T^+[nev][nev-1] is the transformed subdiagonal,
            // v^+_{nev} is column nev of V*Q, sigma = Q[m-1][nev-1] is the last row of the
            // accumulated shift rotation, and f is the residual before the restart.
            let beta_k = if nev >= 1 && nev - 1 < beta.len() {
                beta[nev - 1].clone()
            } else {
                T::zero()
            };
            let sigma = if m >= 1 && nev >= 1 {
                q_total[m - 1][nev - 1].clone()
            } else {
                T::zero()
            };

            let mut f_new = vec![T::zero(); n];
            if nev < v_new.len() {
                for i in 0..n {
                    f_new[i] =
                        beta_k.clone() * v_new[nev][i].clone() + sigma.clone() * f[i].clone();
                }
            }

            // Keep only the nev leading (B-orthonormal) basis vectors.
            v_new.truncate(nev);
            alpha.truncate(nev);
            beta.truncate(nev.saturating_sub(1));

            // Re-B-orthogonalize the new residual against the kept basis (numerical
            // robustness), then normalize it into the next Lanczos vector.
            for k in 0..v_new.len() {
                let mut bvk = vec![T::zero(); n];
                spmv(T::one(), b, &v_new[k], T::zero(), &mut bvk);
                let coef = dot(&f_new, &bvk);
                for i in 0..n {
                    f_new[i] = f_new[i].clone() - coef.clone() * v_new[k][i].clone();
                }
            }
            let mut bf_new = vec![T::zero(); n];
            spmv(T::one(), b, &f_new, T::zero(), &mut bf_new);
            let beta_new = Real::sqrt(dot(&f_new, &bf_new));

            if beta_new <= <T as Scalar>::epsilon() {
                // Invariant subspace reached; stop restarting.
                break;
            }

            let v_cont: Vec<T> = f_new.iter().map(|x| x.clone() / beta_new.clone()).collect();
            v_new.push(v_cont);
            beta.push(beta_new.clone());
            beta_prev = beta_new.clone();
            f = f_new;
            f_bnorm = beta_new;
            v_storage = v_new;
        }

        // Max iterations reached
        let m = alpha.len();
        let (ritz_values, _ritz_vectors) = self.symmetric_tridiag_qr(&alpha, &beta, m);

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

    /// Compute the documented generalized residual norm ||A x - lambda B x||_2 against
    /// the original pencil (A, B).
    fn generalized_residual(
        a: &CsrMatrix<T>,
        b: &CsrMatrix<T>,
        lambda: &T,
        x: &[T],
        n: usize,
    ) -> T {
        let mut ax = vec![T::zero(); n];
        spmv(T::one(), a, x, T::zero(), &mut ax);
        let mut bx = vec![T::zero(); n];
        spmv(T::one(), b, x, T::zero(), &mut bx);
        let mut r = vec![T::zero(); n];
        for i in 0..n {
            r[i] = ax[i].clone() - lambda.clone() * bx[i].clone();
        }
        norm(&r)
    }

    /// Arnoldi iteration for non-symmetric generalized problems with implicit restart.
    ///
    /// This is an Implicitly Restarted Arnoldi Method (IRAM) adapted to the generalized
    /// problem A x = lambda B x. The Krylov basis V is kept B-orthonormal
    /// (V^T B V = I), the projected operator H is upper Hessenberg, and after every
    /// Arnoldi sweep the unwanted Ritz values are used as exact shifts in implicit QR
    /// steps that compress the factorization from `ncv` back to `nev` before it is
    /// extended again. Complex-conjugate Ritz pairs of a real pencil are recovered from
    /// the 2x2 blocks of the real Schur form of H, so genuinely complex eigenvalues are
    /// reported through `eigenvalues` / `eigenvalues_imag` rather than being discarded.
    fn arnoldi_generalized<F, L>(
        &self,
        a: &CsrMatrix<T>,
        b: &CsrMatrix<T>,
        _b_factor: &SparseCholesky<T>,
        n: usize,
        nev: usize,
        ncv: usize,
        initial_v: Vec<T>,
        op: F,
        lambda_of_mu: L,
    ) -> Result<GeneralizedEigenResult<T>, EigenvalueError>
    where
        F: Fn(&[T]) -> Vec<T>,
        L: Fn(&T, &T) -> (T, T),
    {
        let p = ncv.saturating_sub(nev);
        let tol_breakdown = <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or_else(T::one);

        // B-normalize the initial vector so that v^T B v = 1.
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
        for vi in &mut v {
            *vi = vi.clone() / b_norm.clone();
        }

        let mut arnoldi_vectors: Vec<Vec<T>> = Vec::with_capacity(ncv + 1);
        arnoldi_vectors.push(v);

        // Row-major upper Hessenberg matrix H with H[i][j] = <op(v_j), B v_i>.
        let mut h: Vec<Vec<T>> = vec![vec![T::zero(); ncv]; ncv];
        // Arnoldi residual vector f (B-orthogonal to the current basis).
        let mut f = vec![T::zero(); n];

        // Build the initial B-orthonormal Arnoldi factorization up to ncv vectors.
        for j in 0..ncv {
            let mut w = op(&arnoldi_vectors[j]);
            self.b_orthogonalize(&mut w, &arnoldi_vectors, &mut h, j, b, n);
            let mut bw = vec![T::zero(); n];
            spmv(T::one(), b, &w, T::zero(), &mut bw);
            let f_norm = Real::sqrt(dot(&w, &bw));
            if f_norm <= tol_breakdown {
                f.clone_from_slice(&w);
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
                f.clone_from_slice(&w);
            }
        }

        let mut converged_count = 0;
        let mut iterations_done = 0;

        for iter in 0..self.config.max_iterations {
            iterations_done = iter + 1;
            let current_dim = arnoldi_vectors.len().min(ncv);
            if current_dim < 2 {
                break;
            }

            let (ritz_real, ritz_imag) = self.solve_hessenberg_eigenvalues(&h, current_dim);
            let (wanted, unwanted_real, unwanted_imag) =
                self.select_shifts_general(&ritz_real, &ritz_imag, nev, p);

            // B-norm of the Arnoldi residual (used for the breakdown check below).
            let mut bf = vec![T::zero(); n];
            spmv(T::one(), b, &f, T::zero(), &mut bf);
            let f_norm = Real::sqrt(dot(&f, &bf));

            // Convergence is measured directly by the documented residual
            // ||A x_i - lambda_i B x_i|| against the original pencil. This is robust to
            // the numerical drift of the restarted factorization (whose internal residual
            // estimate can under-report), and matches what is returned to the caller. For
            // a complex Ritz pair the (complex) eigenvector cannot be represented in real
            // storage, so the conservative ||f||_B bound is used instead.
            converged_count = 0;
            for (idx, &wi) in wanted.iter().enumerate() {
                if idx >= nev || wi >= current_dim {
                    continue;
                }
                if Scalar::abs(ritz_imag[wi].clone()) > <T as Scalar>::epsilon() {
                    let ritz_mag = Real::sqrt(
                        ritz_real[wi].clone() * ritz_real[wi].clone()
                            + ritz_imag[wi].clone() * ritz_imag[wi].clone(),
                    );
                    if Scalar::abs(f_norm.clone()) <= self.config.tolerance * ritz_mag.max(T::one())
                    {
                        converged_count += 1;
                    }
                    continue;
                }

                // Real Ritz pair: form x_i = V y_i, B-normalize, and measure the true
                // generalized residual.
                let y = self.hessenberg_eigenvector(&h, current_dim, &ritz_real, &ritz_imag, wi);
                let mut x = vec![T::zero(); n];
                for (jv, vj) in arnoldi_vectors.iter().enumerate() {
                    if jv < y.len() {
                        for i in 0..n {
                            x[i] = x[i].clone() + y[jv].clone() * vj[i].clone();
                        }
                    }
                }
                let mut bx = vec![T::zero(); n];
                spmv(T::one(), b, &x, T::zero(), &mut bx);
                let bn = Real::sqrt(dot(&x, &bx));
                if bn > <T as Scalar>::epsilon() {
                    for xi in &mut x {
                        *xi = xi.clone() / bn.clone();
                    }
                }
                let (lambda_re, _) = lambda_of_mu(&ritz_real[wi], &ritz_imag[wi]);
                let res = Self::generalized_residual(a, b, &lambda_re, &x, n);
                let scale = Scalar::abs(lambda_re) + T::one();
                if res <= self.config.tolerance * scale {
                    converged_count += 1;
                }
            }

            if converged_count >= nev {
                break;
            }

            if unwanted_real.is_empty() {
                break;
            }

            // Implicit restart: apply the unwanted Ritz values as exact QR shifts, which
            // filters the Krylov basis toward the wanted invariant subspace.
            self.apply_implicit_qr_shifts_b(
                &mut h,
                &mut arnoldi_vectors,
                &mut f,
                &unwanted_real,
                &unwanted_imag,
                nev,
                ncv,
                b,
            );

            let mut bf2 = vec![T::zero(); n];
            spmv(T::one(), b, &f, T::zero(), &mut bf2);
            let f_norm2 = Real::sqrt(dot(&f, &bf2));
            if f_norm2 <= tol_breakdown {
                break;
            }

            arnoldi_vectors.truncate(nev);
            let mut v_next = vec![T::zero(); n];
            for i in 0..n {
                v_next[i] = f[i].clone() / f_norm2.clone();
            }
            arnoldi_vectors.push(v_next);
            h[nev][nev - 1] = f_norm2.clone();

            // Extend the Arnoldi factorization from nev back up to ncv.
            while arnoldi_vectors.len() < ncv {
                let j = arnoldi_vectors.len() - 1;
                let mut w = op(&arnoldi_vectors[j]);
                self.b_orthogonalize(&mut w, &arnoldi_vectors, &mut h, j, b, n);
                let mut bw = vec![T::zero(); n];
                spmv(T::one(), b, &w, T::zero(), &mut bw);
                let f_norm_j = Real::sqrt(dot(&w, &bw));
                if f_norm_j <= tol_breakdown {
                    f.clone_from_slice(&w);
                    break;
                }
                if j + 1 < ncv {
                    h[j + 1][j] = f_norm_j.clone();
                    let mut vn = vec![T::zero(); n];
                    for idx in 0..n {
                        vn[idx] = w[idx].clone() / f_norm_j.clone();
                    }
                    arnoldi_vectors.push(vn);
                } else {
                    f.clone_from_slice(&w);
                }
            }
        }

        // Final Ritz decomposition, eigenvector recovery and (true) residual evaluation.
        let current_dim = arnoldi_vectors.len().min(ncv);
        let (ritz_real, ritz_imag) = self.solve_hessenberg_eigenvalues(&h, current_dim);
        let (wanted, _, _) = self.select_shifts_general(&ritz_real, &ritz_imag, nev, p);

        let mut eigenvalues = Vec::with_capacity(nev);
        let mut eigenvalues_imag = Vec::with_capacity(nev);
        let mut residual_norms = Vec::with_capacity(nev);
        let mut eigenvectors_x: Vec<Vec<T>> = Vec::with_capacity(nev);

        for &wi in wanted.iter().take(nev) {
            if wi >= current_dim {
                continue;
            }
            // Approximate eigenvector of H (real inverse iteration for real Ritz values),
            // then map to the original space via x = V y and B-normalize it.
            let y = self.hessenberg_eigenvector(&h, current_dim, &ritz_real, &ritz_imag, wi);
            let mut x = vec![T::zero(); n];
            for (jv, vj) in arnoldi_vectors.iter().enumerate() {
                if jv < y.len() {
                    for i in 0..n {
                        x[i] = x[i].clone() + y[jv].clone() * vj[i].clone();
                    }
                }
            }
            let mut bx = vec![T::zero(); n];
            spmv(T::one(), b, &x, T::zero(), &mut bx);
            let bn = Real::sqrt(dot(&x, &bx));
            if bn > <T as Scalar>::epsilon() {
                for xi in &mut x {
                    *xi = xi.clone() / bn.clone();
                }
            }

            let (lambda_re, lambda_im) = lambda_of_mu(&ritz_real[wi], &ritz_imag[wi]);
            let res = Self::generalized_residual(a, b, &lambda_re, &x, n);
            eigenvalues.push(lambda_re);
            eigenvalues_imag.push(lambda_im);
            residual_norms.push(res);
            eigenvectors_x.push(x);
        }

        // Pad (degenerate case where fewer than nev Ritz pairs were available).
        while eigenvalues.len() < nev {
            eigenvalues.push(T::zero());
            eigenvalues_imag.push(T::zero());
            residual_norms.push(T::zero());
        }

        let eigenvectors = if self.config.compute_eigenvectors {
            Some(eigenvectors_x)
        } else {
            None
        };

        Ok(GeneralizedEigenResult {
            eigenvalues,
            eigenvalues_imag,
            eigenvectors,
            iterations: iterations_done,
            residual_norms,
            converged: converged_count >= nev,
            num_converged: converged_count.min(nev),
        })
    }

    /// Modified Gram-Schmidt of `w` against the first `j+1` Arnoldi vectors in the
    /// B-inner product, storing the coefficients into column `j` of the Hessenberg
    /// matrix H.
    ///
    /// A second orthogonalization pass (classical iterative refinement, "DGKS") is
    /// performed and its corrections are accumulated into H. This is essential for
    /// numerical robustness: a single pass loses B-orthogonality for stiff operators,
    /// which corrupts the projected Hessenberg matrix and hence the Ritz values.
    fn b_orthogonalize(
        &self,
        w: &mut [T],
        arnoldi_vectors: &[Vec<T>],
        h: &mut [Vec<T>],
        j: usize,
        b: &CsrMatrix<T>,
        n: usize,
    ) {
        // Precompute B v_i for i = 0..=j (reused across both passes).
        let mut bvs: Vec<Vec<T>> = Vec::with_capacity(j + 1);
        for i in 0..=j {
            if i >= arnoldi_vectors.len() {
                break;
            }
            let mut bvi = vec![T::zero(); n];
            spmv(T::one(), b, &arnoldi_vectors[i], T::zero(), &mut bvi);
            bvs.push(bvi);
        }

        for pass in 0..2 {
            for (i, bvi) in bvs.iter().enumerate() {
                let coeff = dot(w, bvi);
                if i < h.len() && j < h[i].len() {
                    // Overwrite on the first pass (clears any stale value left by a
                    // previous restart), accumulate the refinement on the second.
                    h[i][j] = if pass == 0 {
                        coeff.clone()
                    } else {
                        h[i][j].clone() + coeff.clone()
                    };
                }
                for k in 0..n {
                    w[k] = w[k].clone() - coeff.clone() * arnoldi_vectors[i][k].clone();
                }
            }
        }
    }
}

mod kernel;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr::CsrMatrix;

    fn dense_to_csr(n: usize, rows: &[Vec<(usize, f64)>]) -> CsrMatrix<f64> {
        let mut row_ptrs = vec![0usize];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for r in rows {
            for &(c, v) in r {
                col_indices.push(c);
                values.push(v);
            }
            row_ptrs.push(col_indices.len());
        }
        CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap()
    }

    /// Finding #1: complex-conjugate eigenvalues of a real non-symmetric pencil must be
    /// recovered (real and imaginary parts), not silently dropped.
    #[test]
    fn nonsymmetric_complex_conjugate_eigenvalues() {
        // Block-diagonal, normal: [[1,-1],[1,1]] -> 1 +/- i, [[2,-1],[1,2]] -> 2 +/- i.
        let n = 4;
        let rows = vec![
            vec![(0, 1.0), (1, -1.0)],
            vec![(0, 1.0), (1, 1.0)],
            vec![(2, 2.0), (3, -1.0)],
            vec![(2, 1.0), (3, 2.0)],
        ];
        let a = dense_to_csr(n, &rows);
        let b = CsrMatrix::<f64>::eye(n);

        let config = GeneralizedEigenConfig {
            num_eigenvalues: 2,
            which: WhichEigenvalues::LargestMagnitude,
            max_iterations: 100,
            tolerance: 1e-8,
            compute_eigenvectors: false,
            krylov_dimension: 4,
            symmetric: false,
            mode: GeneralizedMode::Standard,
            sigma: 0.0,
        };

        let result = GeneralizedEigen::new(config).compute(&a, &b, None).unwrap();

        // The dominant pair is 2 +/- i: non-zero (and conjugate) imaginary parts.
        assert_eq!(result.eigenvalues_imag.len(), 2);
        let max_abs_im = result
            .eigenvalues_imag
            .iter()
            .fold(0.0_f64, |m, &x| m.max(x.abs()));
        assert!(
            max_abs_im > 0.5,
            "a genuinely complex eigenvalue must be reported, got {:?}",
            result.eigenvalues_imag
        );
        let imag_sum: f64 = result.eigenvalues_imag.iter().sum();
        assert!(imag_sum.abs() < 1e-3, "conjugate imag parts must cancel");
        for (re, im) in result
            .eigenvalues
            .iter()
            .zip(result.eigenvalues_imag.iter())
        {
            assert!((re - 2.0).abs() < 1e-4, "real part should be ~2, got {re}");
            let mag = (re * re + im * im).sqrt();
            assert!(
                (mag - 5.0_f64.sqrt()).abs() < 1e-3,
                "magnitude should be ~sqrt(5)"
            );
        }
    }

    /// Finding #2: residual_norms must be the documented ||A x - lambda B x|| against the
    /// original pencil, i.e. small for a converged eigenpair.
    #[test]
    fn residual_norms_are_true_generalized_residuals() {
        let n = 12;
        let mut a_rows: Vec<Vec<(usize, f64)>> = Vec::new();
        let mut b_rows: Vec<Vec<(usize, f64)>> = Vec::new();
        for i in 0..n {
            let mut ar = Vec::new();
            let mut br = Vec::new();
            if i > 0 {
                ar.push((i - 1, -1.0));
                br.push((i - 1, -0.25));
            }
            ar.push((i, 2.0));
            br.push((i, 2.0));
            if i < n - 1 {
                ar.push((i + 1, -1.0));
                br.push((i + 1, -0.25));
            }
            a_rows.push(ar);
            b_rows.push(br);
        }
        let a = dense_to_csr(n, &a_rows);
        let b = dense_to_csr(n, &b_rows);

        let config = GeneralizedEigenConfig {
            num_eigenvalues: 3,
            which: WhichEigenvalues::LargestMagnitude,
            max_iterations: 200,
            tolerance: 1e-8,
            compute_eigenvectors: true,
            krylov_dimension: 12,
            symmetric: true,
            mode: GeneralizedMode::Standard,
            sigma: 0.0,
        };

        let result = GeneralizedEigen::new(config).compute(&a, &b, None).unwrap();
        assert!(result.converged);

        let evecs = result.eigenvectors.as_ref().unwrap();
        for (k, x) in evecs.iter().enumerate() {
            let lam = result.eigenvalues[k];
            let mut ax = vec![0.0; n];
            let mut bx = vec![0.0; n];
            for (i, row) in a_rows.iter().enumerate() {
                for &(c, v) in row {
                    ax[i] += v * x[c];
                }
            }
            for (i, row) in b_rows.iter().enumerate() {
                for &(c, v) in row {
                    bx[i] += v * x[c];
                }
            }
            let r: f64 = (0..n)
                .map(|i| (ax[i] - lam * bx[i]).powi(2))
                .sum::<f64>()
                .sqrt();
            // Reported residual matches the independently computed ||A x - lambda B x||...
            assert!(
                (r - result.residual_norms[k]).abs() < 1e-6,
                "reported residual must equal ||A x - lambda B x||: {} vs {}",
                result.residual_norms[k],
                r
            );
            // ...and it is small for a converged eigenpair.
            assert!(r < 1e-5, "true residual should be small, got {r}");
        }
    }

    /// Finding #3: with ncv < n the implicit-restart loop must iterate and converge
    /// (rather than doing a single Krylov pass), yielding small residuals.
    #[test]
    fn implicit_restart_converges() {
        let n = 40;
        let mut a_rows: Vec<Vec<(usize, f64)>> = Vec::new();
        for i in 0..n {
            let mut ar = Vec::new();
            if i > 0 {
                ar.push((i - 1, -1.0));
            }
            ar.push((i, 2.0 + i as f64 * 0.1));
            if i < n - 1 {
                ar.push((i + 1, -1.0));
            }
            a_rows.push(ar);
        }
        let a = dense_to_csr(n, &a_rows);
        let b = CsrMatrix::<f64>::eye(n);

        let config = GeneralizedEigenConfig {
            num_eigenvalues: 3,
            which: WhichEigenvalues::LargestMagnitude,
            max_iterations: 300,
            tolerance: 1e-7,
            compute_eigenvectors: true,
            krylov_dimension: 12, // ncv = 12 << n = 40 forces restarts
            symmetric: true,
            mode: GeneralizedMode::Standard,
            sigma: 0.0,
        };

        let result = GeneralizedEigen::new(config).compute(&a, &b, None).unwrap();
        assert!(result.converged, "restart loop should converge");
        assert!(result.iterations >= 1);
        for &r in &result.residual_norms {
            assert!(r < 1e-3, "converged residual should be small, got {r}");
        }
    }
}
