//! Singular Value Decomposition (SVD).
//!
//! Computes the SVD: A = U·Σ·V^T where:
//! - U is m×m orthogonal (left singular vectors)
//! - Σ is m×n diagonal (singular values, non-negative, descending)
//! - V is n×n orthogonal (right singular vectors)
//!
//! This module provides:
//!
//! - **Svd**: One-sided Jacobi algorithm. Simple and robust for small matrices.
//! - **SvdDc**: Divide-and-conquer algorithm. More efficient for large matrices.
//! - **ComplexSvd**: One-sided Jacobi algorithm for complex matrices.
//! - **ComplexSvdDc**: Divide-and-conquer algorithm for complex matrices. Uses complex
//!   bidiagonalization to reduce to real bidiagonal form, then applies real D&C SVD.
//! - **QrSvd**: QR-based algorithm (Golub-Kahan-Reinsch). Classical LAPACK approach.
//! - **TruncatedSvd**: Computes only the k largest singular values. Efficient for
//!   low-rank approximations and dimensionality reduction.
//! - **RandomizedSvd**: Randomized algorithm for fast truncated SVD of large matrices.
//! - **SelectiveSvd**: Computes only selected singular values/vectors (by index or value range).
//!   Equivalent to LAPACK's GESVDX.
//! - **Bidiagonal reduction**: `gebrd`, `ormbr`/`unmbr`, `orgbr`/`ungbr` for
//!   working with bidiagonal transformations.
//! - **ParallelSvdDc** (feature = "parallel"): Multi-threaded divide-and-conquer SVD
//!   using Rayon for parallel execution.
//!
//! # Example
//!
//! ```
//! use oxiblas_lapack::svd::{Svd, SvdDc, TruncatedSvd};
//! use oxiblas_matrix::Mat;
//!
//! let a = Mat::from_rows(&[
//!     &[1.0f64, 2.0],
//!     &[3.0, 4.0],
//!     &[5.0, 6.0],
//! ]);
//!
//! // Jacobi method (simple, robust)
//! let svd = Svd::compute(a.as_ref()).unwrap();
//!
//! // Divide-and-conquer (faster for large matrices)
//! let svd_dc = SvdDc::compute(a.as_ref()).unwrap();
//!
//! // Both compute the same singular values
//! let sigma = svd.singular_values();
//! assert!(sigma[0] >= sigma[1]);
//!
//! // Truncated SVD (k largest singular values)
//! let tsvd = TruncatedSvd::compute(a.as_ref(), 1).unwrap();
//! assert_eq!(tsvd.singular_values().len(), 1);
//! ```
//!
//! # Bidiagonal Reduction
//!
//! ```
//! use oxiblas_lapack::svd::{gebrd, orgbr, ormbr, BidiagVect, Side, Trans};
//! use oxiblas_matrix::Mat;
//!
//! let a = Mat::from_rows(&[
//!     &[1.0f64, 2.0, 3.0],
//!     &[4.0, 5.0, 6.0],
//!     &[7.0, 8.0, 9.0],
//!     &[10.0, 11.0, 12.0],
//! ]);
//!
//! // Compute bidiagonal reduction: A = Q * B * P^T
//! let factors = gebrd(a.as_ref()).unwrap();
//!
//! // Generate Q explicitly
//! let q = orgbr(&factors, BidiagVect::Q).unwrap();
//!
//! // Or apply Q to a matrix without forming it
//! let c = Mat::from_rows(&[&[1.0], &[2.0], &[3.0], &[4.0]]);
//! let qc = ormbr(&factors, BidiagVect::Q, Side::Left, Trans::NoTrans, c.as_ref()).unwrap();
//! ```

mod bidiag_reduce;
mod bidiagonal;
mod complex;
mod complex_dc;
mod divide_conquer;
mod qr_based;
mod randomized;
mod selective;
mod truncated;

#[cfg(feature = "parallel")]
mod parallel_svd;

pub use bidiag_reduce::{
    BidiagError, BidiagFactors, BidiagVect, Side, Trans, gebrd, orgbr, ormbr, ungbr, unmbr,
};
pub use bidiagonal::{Svd, SvdError};
pub use complex::{ComplexSvd, ComplexSvdError};
pub use complex_dc::{ComplexSvdDc, ComplexSvdDcError};
pub use divide_conquer::{SvdDc, SvdDcError};
pub use qr_based::{QrSvd, QrSvdError};
pub use randomized::{
    RandomizedSvd, RandomizedSvdConfig, RandomizedSvdError, low_rank_approximation, rsvd,
    rsvd_power,
};
pub use selective::{
    SelectiveSvd, SelectiveSvdError, SingularValueSelector, count_singular_values_above,
    singular_value_bounds,
};
pub use truncated::{
    TruncatedSvd, TruncatedSvdError, numerical_rank, optimal_rank_for_energy, rank_k_approximation,
    thin_svd,
};

#[cfg(feature = "parallel")]
pub use parallel_svd::{ParallelSvdDc, ParallelSvdError};
