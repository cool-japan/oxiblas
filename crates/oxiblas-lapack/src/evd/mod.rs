//! Eigenvalue Decomposition.
//!
//! Computes eigenvalues and eigenvectors of matrices.
//!
//! This module provides:
//!
//! - **SymmetricEvd**: For symmetric/Hermitian matrices. Eigenvalues are always real.
//! - **GeneralEvd**: For general (non-symmetric) matrices. Eigenvalues may be complex.
//! - **SymmetricGeneralizedEvd**: For generalized symmetric-definite problems A*x = λ*B*x.
//! - **GeneralizedEvd**: For general matrix pencils.
//! - **Qz**: QZ algorithm for generalized Schur decomposition.
//! - **Hessenberg**: Reduction to upper Hessenberg form.
//! - **ComplexHessenberg**: Reduction to upper Hessenberg form for complex matrices.
//! - **Schur**: Schur decomposition (real Schur form).
//! - **ComplexSchur**: Complex Schur decomposition (upper triangular form).
//! - **Balance**: Matrix balancing to improve eigenvalue accuracy.
//! - **TridiagEvd**: Bisection and inverse iteration for tridiagonal matrices.
//!   Efficient for computing selected eigenvalues/eigenvectors.
//! - **MrrrEvd**: MRRR algorithm (Multiple Relatively Robust Representations)
//!   for efficient eigenvector computation with clustered eigenvalues.
//! - **ParallelSymmetricEvd** (feature = "parallel"): Multi-threaded divide-and-conquer
//!   for symmetric eigenvalue decomposition using Rayon.
//!
//! # Example
//!
//! ```
//! use oxiblas_lapack::evd::{SymmetricEvd, GeneralEvd};
//! use oxiblas_matrix::Mat;
//!
//! // Symmetric matrix - real eigenvalues
//! let a = Mat::from_rows(&[
//!     &[4.0f64, 1.0],
//!     &[1.0, 3.0],
//! ]);
//! let evd = SymmetricEvd::compute(a.as_ref()).unwrap();
//! let eigenvalues = evd.eigenvalues();
//!
//! // General matrix - eigenvalues may be complex
//! let b = Mat::from_rows(&[
//!     &[0.0f64, -1.0],
//!     &[1.0, 0.0],
//! ]);
//! let evd_gen = GeneralEvd::eigenvalues_only(b.as_ref()).unwrap();
//! // Rotation matrix has complex eigenvalues: ±i
//! ```

pub mod balance;
mod complex_general;
mod complex_hessenberg;
mod complex_schur;
mod general;
mod generalized;
mod hermitian;
mod hermitian_dc;
mod hessenberg;
mod mrrr;
mod qz;
mod schur;
mod symmetric;
mod symmetric_dc;
mod tridiag_evd;

#[cfg(feature = "parallel")]
mod parallel_evd;

pub use balance::{Balance, BalanceError, BalanceJob, BalanceSide, gebak, gebal};
pub use complex_general::{ComplexGeneralEvd, ComplexGeneralEvdError};
pub use complex_hessenberg::{
    ComplexHessenberg, ComplexHessenbergError, ComplexHessenbergFactors, zgehrd, zgehrd_range,
    zunhhr, zunmhr,
};
pub use complex_schur::{ComplexSchur, ComplexSchurError};
pub use general::{GeneralEvd, GeneralEvdError};
pub use generalized::{GeneralizedEvd, GeneralizedEvdError, SymmetricGeneralizedEvd};
pub use hermitian::{HermitianEvd, HermitianEvdError};
pub use hermitian_dc::{HermitianEvdDc, HermitianEvdDcError};
pub use hessenberg::{
    Hessenberg, HessenbergError, HessenbergFactors, Side, Trans, gehrd, gehrd_range, orghr, ormhr,
    unghr, unmhr,
};
pub use mrrr::{MrrrError, MrrrEvd};
pub use qz::{GeneralizedEigenvalue, Qz, QzError};
pub use schur::{Eigenvalue, Schur, SchurError, trevc_left, trevc_right, trsna_s, trsna_sep};
pub use symmetric::{
    SymmetricEvd, SymmetricEvdError, TridiagFactors, TridiagSide, TridiagTrans, Uplo, orgtr, ormtr,
    sytrd, ungtr, unmtr,
};
pub use symmetric_dc::{SymmetricEvdDc, SymmetricEvdDcError};
pub use tridiag_evd::{
    EigenvalueSelector, TridiagEvd, TridiagEvdError, count_eigenvalues, eigenvalue_bounds,
    eigenvalues_by_index, eigenvalues_in_range,
};

#[cfg(feature = "parallel")]
pub use parallel_evd::{
    ParallelEvdError, ParallelSymmetricEvd, parallel_bisection_eigenvalues,
    parallel_inverse_iteration,
};
