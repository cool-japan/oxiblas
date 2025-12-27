//! # RandomizedSparseSvdConfig - Trait Implementations
//!
//! This module contains trait implementations for `RandomizedSparseSvdConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::RandomizedSparseSvdConfig;

impl Default for RandomizedSparseSvdConfig {
    fn default() -> Self {
        Self {
            num_singular_values: 6,
            oversampling: 10,
            power_iterations: 2,
            seed: None,
            compute_vectors: true,
        }
    }
}
