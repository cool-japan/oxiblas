//! Error types for preconditioners.

use crate::csr::CsrError;

/// Error type for preconditioner operations.
#[derive(Debug, Clone, PartialEq)]
pub enum PreconditionerError {
    /// Invalid matrix (not square, wrong dimensions, etc.)
    InvalidMatrix(String),
    /// Zero diagonal element at position
    ZeroDiagonal(usize),
    /// Singular block at index
    SingularBlock(usize),
    /// Internal sparse matrix construction failed while building a
    /// preconditioner (e.g. an AMG/SAMG interpolation, restriction, or
    /// Galerkin coarse-grid operator produced structurally invalid CSR
    /// data, typically because a coarse-level matrix was singular or
    /// degenerate).
    MatrixConstruction(String),
}

impl std::fmt::Display for PreconditionerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PreconditionerError::InvalidMatrix(msg) => write!(f, "Invalid matrix: {}", msg),
            PreconditionerError::ZeroDiagonal(i) => write!(f, "Zero diagonal at position {}", i),
            PreconditionerError::SingularBlock(i) => write!(f, "Singular block at index {}", i),
            PreconditionerError::MatrixConstruction(msg) => {
                write!(f, "Preconditioner matrix construction failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for PreconditionerError {}

impl From<CsrError> for PreconditionerError {
    fn from(err: CsrError) -> Self {
        PreconditionerError::MatrixConstruction(err.to_string())
    }
}
