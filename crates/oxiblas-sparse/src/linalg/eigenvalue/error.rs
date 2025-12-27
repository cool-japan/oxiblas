//! Sparse Eigenvalue Solvers.
//!
//! This module provides iterative methods for computing eigenvalues and eigenvectors
//! of large sparse matrices:
//!
//! - **Lanczos iteration**: For symmetric/Hermitian matrices, computes extremal eigenvalues
//! - **Arnoldi iteration**: For general matrices, computes eigenvalues
//! - **Shift-and-invert**: For interior eigenvalues (eigenvalues near a specified shift)
//! - **IRAM**: Implicitly Restarted Arnoldi Method for memory-efficient eigenvalue computation
//!
//! These algorithms work with the matrix only through matrix-vector products,
//! making them ideal for large sparse matrices where factorization is infeasible.

use core::fmt;

/// Error type for eigenvalue solvers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EigenvalueError {
    /// Maximum iterations reached without convergence.
    MaxIterations {
        /// Number of iterations performed.
        iterations: usize,
        /// Number of converged eigenvalues.
        converged_count: usize,
    },
    /// Dimension mismatch.
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },
    /// Matrix is not square.
    NotSquare {
        /// Number of rows.
        nrows: usize,
        /// Number of columns.
        ncols: usize,
    },
    /// Breakdown in iteration (invariant subspace found).
    Breakdown {
        /// Iteration where breakdown occurred.
        iteration: usize,
        /// Description of breakdown.
        description: String,
    },
    /// Requested more eigenvalues than matrix dimension.
    TooManyEigenvalues {
        /// Requested number.
        requested: usize,
        /// Maximum allowed.
        max_allowed: usize,
    },
    /// Internal computation error.
    ComputationError(String),
}

impl fmt::Display for EigenvalueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MaxIterations {
                iterations,
                converged_count,
            } => {
                write!(
                    f,
                    "Max iterations ({iterations}) reached, {converged_count} eigenvalues converged"
                )
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {expected}, got {actual}")
            }
            Self::NotSquare { nrows, ncols } => {
                write!(f, "Matrix is not square: {nrows}x{ncols}")
            }
            Self::Breakdown {
                iteration,
                description,
            } => {
                write!(f, "Breakdown at iteration {iteration}: {description}")
            }
            Self::TooManyEigenvalues {
                requested,
                max_allowed,
            } => {
                write!(f, "Requested {requested} eigenvalues, max is {max_allowed}")
            }
            Self::ComputationError(msg) => write!(f, "Computation error: {msg}"),
        }
    }
}

impl std::error::Error for EigenvalueError {}

/// Which eigenvalues to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhichEigenvalues {
    /// Largest magnitude eigenvalues.
    LargestMagnitude,
    /// Smallest magnitude eigenvalues.
    SmallestMagnitude,
    /// Largest algebraic eigenvalues (most positive).
    LargestAlgebraic,
    /// Smallest algebraic eigenvalues (most negative).
    SmallestAlgebraic,
    /// Eigenvalues closest to a target (requires shift-and-invert).
    NearTarget,
}
