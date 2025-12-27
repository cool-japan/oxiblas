//! Types and enums for triangular solve operations.

/// Specifies which side the triangular matrix is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// Solve A·X = α·B (A is on the left).
    Left,
    /// Solve X·A = α·B (A is on the right).
    Right,
}

/// Specifies which triangle of the matrix is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Uplo {
    /// Lower triangular matrix.
    Lower,
    /// Upper triangular matrix.
    Upper,
}

/// Specifies whether to transpose the matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trans {
    /// No transpose.
    NoTrans,
    /// Transpose.
    Trans,
    /// Conjugate transpose (for complex types).
    ConjTrans,
}

/// Specifies whether the matrix has unit diagonal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Diag {
    /// Non-unit diagonal (use actual diagonal values).
    NonUnit,
    /// Unit diagonal (assume diagonal is all ones).
    Unit,
}

/// Error type for triangular solve operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrsmError {
    /// Matrix A is not square.
    NotSquare,
    /// Dimension mismatch.
    DimensionMismatch,
    /// Matrix is singular (zero on diagonal).
    Singular,
}

impl core::fmt::Display for TrsmError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotSquare => write!(f, "Matrix A is not square"),
            Self::DimensionMismatch => write!(f, "Dimension mismatch"),
            Self::Singular => write!(f, "Matrix is singular (zero on diagonal)"),
        }
    }
}

impl std::error::Error for TrsmError {}
