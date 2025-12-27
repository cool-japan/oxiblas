//! # NestedDissectionConfig - Trait Implementations
//!
//! This module contains trait implementations for `NestedDissectionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::NestedDissectionConfig;

impl Default for NestedDissectionConfig {
    fn default() -> Self {
        Self {
            min_size: 20,
            max_depth: 50,
            balance_tolerance: 0.2,
        }
    }
}
