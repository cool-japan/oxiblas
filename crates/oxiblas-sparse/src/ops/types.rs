//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

/// Triangular part of a sparse matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriangularPart {
    /// Lower triangular (including diagonal).
    Lower,
    /// Upper triangular (including diagonal).
    Upper,
    /// Strictly lower triangular (excluding diagonal).
    StrictlyLower,
    /// Strictly upper triangular (excluding diagonal).
    StrictlyUpper,
}
