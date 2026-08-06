//! Special eigenvalue methods: interval-based and polynomial filtering.
//!
//! This module provides specialized eigenvalue solvers for computing eigenvalues
//! within a specified interval or using polynomial filtering techniques:
//!
//! - [`IntervalEigen`]: Computes all eigenvalues within a given interval [low, high]
//!   using Lanczos iteration with Sturm sequence counting.
//!
//! - [`PolynomialFilteredLanczos`]: Uses Chebyshev polynomial filtering to compute
//!   interior eigenvalues without expensive matrix factorization.
//!
//! # Example
//!
//! ```
//! use oxiblas_sparse::csr::CsrMatrix;
//! use oxiblas_sparse::linalg::eigenvalue::{
//!     IntervalEigen, IntervalEigenConfig, PolynomialFilterConfig, PolynomialFilteredLanczos,
//! };
//!
//! // A diagonal matrix has its diagonal entries as eigenvalues: 1..=5.
//! let matrix =
//!     CsrMatrix::new(5, 5, vec![0, 1, 2, 3, 4, 5], vec![0, 1, 2, 3, 4], vec![
//!         1.0, 2.0, 3.0, 4.0, 5.0,
//!     ])
//!     .unwrap();
//!
//! // Compute eigenvalues in [1.5, 3.5] using interval method -> {2.0, 3.0}
//! let config = IntervalEigenConfig::new(1.5, 3.5);
//! let solver = IntervalEigen::new(config);
//! let result = solver.compute(&matrix, None)?;
//! assert_eq!(result.eigenvalues.len(), 2);
//!
//! // Compute eigenvalues using polynomial filtering, targeting the same interval
//! let config = PolynomialFilterConfig::new(1.5, 3.5);
//! let solver = PolynomialFilteredLanczos::new(config);
//! let result = solver.compute(&matrix, None)?;
//! assert!(!result.eigenvalues.is_empty());
//! # Ok::<(), oxiblas_sparse::linalg::EigenvalueError>(())
//! ```

mod interval;
mod polynomial_filtered;

pub use interval::*;
pub use polynomial_filtered::*;
