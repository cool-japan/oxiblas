//! # `IRAMConfig` - Trait Implementations
//!
//! This module contains trait implementations for `IRAMConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::error::WhichEigenvalues;

use super::types::IRAMConfig;

impl Default for IRAMConfig<f64> {
    fn default() -> Self {
        Self {
            num_eigenvalues: 6,
            which: WhichEigenvalues::LargestMagnitude,
            max_iterations: 300,
            tolerance: 1e-8,
            compute_eigenvectors: true,
            krylov_dimension: 20,
            symmetric: false,
        }
    }
}
impl Default for IRAMConfig<f32> {
    fn default() -> Self {
        Self {
            num_eigenvalues: 6,
            which: WhichEigenvalues::LargestMagnitude,
            max_iterations: 300,
            tolerance: 1e-6,
            compute_eigenvectors: true,
            krylov_dimension: 20,
            symmetric: false,
        }
    }
}
