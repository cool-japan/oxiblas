//! Implicitly Restarted Arnoldi Method (IRAM) for sparse eigenvalue problems.
//!
//! IRAM is a memory-efficient eigenvalue algorithm that computes a few eigenvalues
//! of large sparse matrices. It maintains a Krylov subspace of bounded size through
//! implicit restarts using shifted QR iterations.
//!
//! # Algorithm Overview
//!
//! 1. Build initial Arnoldi factorization: A*V_m = V_m*H_m + f_m*e_m^T
//! 2. Compute Ritz values (eigenvalues of H_m)
//! 3. Select p = m - k unwanted Ritz values as shifts
//! 4. Apply p implicit QR shifts to compress factorization to dimension k
//! 5. Continue Arnoldi from dimension k back to m
//! 6. Repeat until convergence
//!
//! For symmetric matrices, IRAM reduces to Implicitly Restarted Lanczos (IRL),
//! where H is tridiagonal and the algorithm is more efficient.

pub mod iram_impl;
pub mod iram_impl_2;
pub mod iram_type;
pub mod iramconfig_traits;
pub mod types;

// Re-export all types
pub use iram_type::*;
pub use types::*;
