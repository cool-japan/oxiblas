//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::error::WhichEigenvalues;


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
