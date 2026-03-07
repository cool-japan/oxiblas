//! # IncrementalSVDConfig - Trait Implementations
//!
//! This module contains trait implementations for `IncrementalSVDConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::IncrementalSVDConfig;

use num_traits::FromPrimitive;
use oxiblas_core::scalar::Real;
impl<T: Real + FromPrimitive> Default for IncrementalSVDConfig<T> {
    fn default() -> Self {
        Self {
            max_rank: 10,
            tolerance: T::from_f64(1e-10).unwrap_or_else(T::zero),
            reorthogonalize: true,
        }
    }
}
