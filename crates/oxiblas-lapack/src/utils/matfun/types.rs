//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

/// Error type for matrix functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatFunError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Matrix is not square.
    NotSquare {
        /// Number of rows.
        nrows: usize,
        /// Number of columns.
        ncols: usize,
    },
    /// Function is not defined for this matrix.
    NotDefined,
    /// Computation did not converge.
    NotConverged,
    /// Eigenvalue decomposition failed.
    EvdFailed,
}
