//! Error and result types for iterative solvers.

/// Error type for iterative solvers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IterativeError {
    /// Maximum iterations reached without convergence.
    MaxIterations {
        /// Number of iterations performed.
        iterations: usize,
        /// Final residual norm.
        residual: String,
    },
    /// Breakdown during iteration.
    Breakdown {
        /// Iteration where breakdown occurred.
        iteration: usize,
        /// Description of breakdown.
        description: String,
    },
    /// Dimension mismatch.
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },
}

impl core::fmt::Display for IterativeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MaxIterations {
                iterations,
                residual,
            } => {
                write!(
                    f,
                    "Max iterations ({iterations}) reached, residual = {residual}"
                )
            }
            Self::Breakdown {
                iteration,
                description,
            } => {
                write!(f, "Breakdown at iteration {iteration}: {description}")
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for IterativeError {}

/// Result of Conjugate Gradient (CG) solver.
pub struct CgResult<T> {
    /// Solution vector.
    pub x: Vec<T>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final residual norm.
    pub residual_norm: T,
    /// Whether convergence was achieved.
    pub converged: bool,
}

/// Result of GMRES (Generalized Minimal Residual) solver.
pub struct GmresResult<T> {
    /// Solution vector.
    pub x: Vec<T>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Number of restarts performed.
    pub restarts: usize,
    /// Final residual norm.
    pub residual_norm: T,
    /// Whether convergence was achieved.
    pub converged: bool,
    /// Residual history (norm at each iteration).
    pub residual_history: Vec<T>,
}

/// Result of MINRES (Minimum Residual) solver.
pub struct MinresResult<T> {
    /// Solution vector.
    pub x: Vec<T>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final residual norm.
    pub residual_norm: T,
    /// Whether convergence was achieved.
    pub converged: bool,
    /// Residual history (norm at each iteration).
    pub residual_history: Vec<T>,
}

/// Result of TFQMR (Transpose-Free Quasi-Minimal Residual) solver.
pub struct TfqmrResult<T> {
    /// Solution vector.
    pub x: Vec<T>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final residual norm.
    pub residual_norm: T,
    /// Whether convergence was achieved.
    pub converged: bool,
    /// Residual history (norm at each half-iteration).
    pub residual_history: Vec<T>,
}

/// Result of QMR (Quasi-Minimal Residual) solver.
pub struct QmrResult<T> {
    /// Solution vector.
    pub x: Vec<T>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final residual norm.
    pub residual_norm: T,
    /// Whether convergence was achieved.
    pub converged: bool,
    /// Residual history (norm at each iteration).
    pub residual_history: Vec<T>,
}

/// Result of Block Conjugate Gradient solver.
pub struct BlockCgResult<T> {
    /// Solution matrix (each column is a solution vector).
    pub x: Vec<Vec<T>>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Final residual norms for each right-hand side.
    pub residual_norms: Vec<T>,
    /// Whether convergence was achieved for all systems.
    pub converged: bool,
    /// Number of converged systems.
    pub num_converged: usize,
}

/// Result of FGMRES (Flexible GMRES) solver.
pub struct FgmresResult<T> {
    /// Solution vector.
    pub x: Vec<T>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Number of restarts performed.
    pub restarts: usize,
    /// Final residual norm.
    pub residual_norm: T,
    /// Whether convergence was achieved.
    pub converged: bool,
    /// Residual history (norm at each iteration).
    pub residual_history: Vec<T>,
}

/// Result of IDR(s) (Induced Dimension Reduction) solver.
pub struct IdrSResult<T> {
    /// Solution vector
    pub x: Vec<T>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final residual norm
    pub residual_norm: T,
    /// Whether convergence was achieved
    pub converged: bool,
    /// History of residual norms
    pub residual_history: Vec<T>,
}

/// Result of Block GMRES solver.
pub struct BlockGmresResult<T> {
    /// Solution matrix (each element is a solution vector)
    pub x: Vec<Vec<T>>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Number of restarts performed
    pub restarts: usize,
    /// Final residual norm (Frobenius norm of residual matrix)
    pub residual_norm: T,
    /// Whether convergence was achieved
    pub converged: bool,
    /// History of residual norms
    pub residual_history: Vec<T>,
}
