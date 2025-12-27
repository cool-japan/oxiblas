//! Error types for preconditioners.

/// Error type for preconditioner operations.
#[derive(Debug, Clone, PartialEq)]
pub enum PreconditionerError {
    /// Invalid matrix (not square, wrong dimensions, etc.)
    InvalidMatrix(String),
    /// Zero diagonal element at position
    ZeroDiagonal(usize),
    /// Singular block at index
    SingularBlock(usize),
}

impl std::fmt::Display for PreconditionerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PreconditionerError::InvalidMatrix(msg) => write!(f, "Invalid matrix: {}", msg),
            PreconditionerError::ZeroDiagonal(i) => write!(f, "Zero diagonal at position {}", i),
            PreconditionerError::SingularBlock(i) => write!(f, "Singular block at index {}", i),
        }
    }
}

impl std::error::Error for PreconditionerError {}
