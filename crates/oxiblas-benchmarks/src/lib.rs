//! OxiBLAS Benchmarks library.
//!
//! Exposes the performance regression framework so tests and integration
//! harnesses can depend on it without requiring criterion.

#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]

pub mod quick_perf;
pub mod regression;
