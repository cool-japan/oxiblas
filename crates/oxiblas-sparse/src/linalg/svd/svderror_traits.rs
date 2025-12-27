//! # SVDError - Trait Implementations
//!
//! This module contains trait implementations for `SVDError`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//! - `Error`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::SVDError;

impl std::fmt::Display for SVDError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SVDError::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            SVDError::ComputationError(msg) => write!(f, "Computation error: {}", msg),
            SVDError::ConvergenceError(msg) => write!(f, "Convergence error: {}", msg),
        }
    }
}

impl std::error::Error for SVDError {}
