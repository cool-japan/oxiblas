//! # TruncatedSVDConfig - Trait Implementations
//!
//! This module contains trait implementations for `TruncatedSVDConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::TruncatedSVDConfig;

use num_traits::FromPrimitive;
use oxiblas_core::scalar::Real;
impl<T: Real + FromPrimitive> Default for TruncatedSVDConfig<T> {
    fn default() -> Self {
        Self {
            num_singular_values: 6,
            max_iterations: 1000,
            tolerance: T::from_f64(1e-10).unwrap_or_else(T::zero),
            compute_vectors: true,
            krylov_dimension: 20,
            full_reorthogonalization: true,
        }
    }
}
