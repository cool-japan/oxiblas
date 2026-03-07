//! Auto-generated module structure

pub mod functions;
pub mod multilevel;
pub mod nesteddissectionconfig_traits;
pub mod types;

// Re-export all types
pub use functions::*;
pub use multilevel::{
    MultilevelPartitioner, OrderingError, Partition, PartitionConfig, multilevel_nested_dissection,
};
pub use types::*;
