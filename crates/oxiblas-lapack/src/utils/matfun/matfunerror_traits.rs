//! # MatFunError - Trait Implementations
//!
//! This module contains trait implementations for `MatFunError`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//! - `Error`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::MatFunError;

impl core::fmt::Display for MatFunError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Empty matrix"),
            Self::NotSquare { nrows, ncols } => {
                write!(f, "Matrix is not square: {nrows} x {ncols}")
            }
            Self::NotDefined => write!(f, "Function is not defined for this matrix"),
            Self::NotConverged => write!(f, "Computation did not converge"),
            Self::EvdFailed => write!(f, "Eigenvalue decomposition failed"),
        }
    }
}

impl std::error::Error for MatFunError {}
