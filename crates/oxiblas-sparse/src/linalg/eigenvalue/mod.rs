//! Sparse Eigenvalue Solvers.
//!
//! This module provides iterative methods for computing eigenvalues and eigenvectors
//! of large sparse matrices:
//!
//! - **Lanczos iteration**: For symmetric/Hermitian matrices, computes extremal eigenvalues
//! - **Arnoldi iteration**: For general matrices, computes eigenvalues
//! - **Shift-and-invert**: For interior eigenvalues (eigenvalues near a specified shift)
//! - **IRAM**: Implicitly Restarted Arnoldi Method for memory-efficient eigenvalue computation
//! - **Generalized eigenvalue**: Solves A*x = λ*B*x for various modes
//! - **Block methods**: Block Lanczos and Block Arnoldi for computing multiple eigenvalues
//! - **Interval methods**: Find eigenvalues within a specified interval
//! - **Polynomial filtering**: Chebyshev polynomial filtering for targeted eigenvalues
//!
//! These algorithms work with the matrix only through matrix-vector products,
//! making them ideal for large sparse matrices where factorization is infeasible.

// Submodules
mod arnoldi;
mod block;
mod error;
mod generalized;
mod iram;
mod lanczos;
pub mod lobpcg;
mod shift_invert;
mod special;
pub mod thick_restart;
pub(crate) mod utils;

#[cfg(test)]
mod tests;

// Re-export error types
pub use error::{EigenvalueError, WhichEigenvalues};

// Re-export Lanczos solver
pub use lanczos::{Lanczos, LanczosConfig, LanczosResult};

// Re-export Arnoldi solver
pub use arnoldi::{Arnoldi, ArnoldiResult};

// Re-export shift-and-invert solver
pub use shift_invert::{ShiftInvertConfig, ShiftInvertLanczos, ShiftInvertResult};

// Re-export IRAM solver
pub use iram::{IRAM, IRAMConfig, IRAMResult};

// Re-export generalized eigenvalue solver
pub use generalized::{
    GeneralizedEigen, GeneralizedEigenConfig, GeneralizedEigenResult, GeneralizedMode,
};

// Re-export block solvers
pub use block::{
    BlockArnoldi, BlockArnoldiConfig, BlockArnoldiResult, BlockLanczos, BlockLanczosConfig,
    BlockLanczosResult,
};

// Re-export Thick-Restart Lanczos solver
pub use thick_restart::{EigenvalueTarget, ThickRestartLanczos, TrlConfig, TrlError, TrlResult};

// Re-export LOBPCG solver
pub use lobpcg::{Lobpcg, LobpcgConfig, LobpcgError, LobpcgResult, LobpcgTarget};

// Re-export special solvers (interval and polynomial filtering)
pub use special::{
    IntervalEigen, IntervalEigenConfig, IntervalEigenResult, PolynomialFilterConfig,
    PolynomialFilteredLanczos, PolynomialFilteredResult, count_eigenvalues_in_interval,
    eigenvalues_in_interval, polynomial_filtered_eigenvalues,
};
