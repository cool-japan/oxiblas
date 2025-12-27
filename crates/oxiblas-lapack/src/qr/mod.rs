//! QR Decomposition and related factorizations.
//!
//! Factorizes a matrix A into Q·R where:
//! - Q is an orthogonal matrix (Q^T·Q = I)
//! - R is an upper triangular matrix
//!
//! This module provides several variants:
//!
//! - **Qr**: Standard QR decomposition using Householder reflections.
//! - **QrPivot**: QR decomposition with column pivoting, useful for
//!   rank-revealing decomposition of rank-deficient matrices.
//! - **Rq**: RQ decomposition (A = R·Q) with R upper trapezoidal.
//! - **Lq**: LQ decomposition (A = L·Q) with L lower trapezoidal.
//! - **CompleteOrthogonalDecomp**: Complete orthogonal decomposition for
//!   rank-deficient matrices.
//!
//! # Example
//!
//! ```
//! use oxiblas_lapack::qr::{Qr, QrPivot};
//! use oxiblas_matrix::Mat;
//!
//! let a = Mat::from_rows(&[
//!     &[1.0f64, 2.0, 3.0],
//!     &[4.0, 5.0, 6.0],
//!     &[7.0, 8.0, 10.0],
//! ]);
//!
//! // Standard QR (faster, usually sufficient)
//! let qr = Qr::compute(a.as_ref()).unwrap();
//! let q = qr.q();
//! let r = qr.r();
//!
//! // QR with column pivoting (rank-revealing)
//! let qr_pivot = QrPivot::compute(a.as_ref()).unwrap();
//! let rank = qr_pivot.rank();
//! ```

mod col_pivot;
mod complete_orthogonal;
mod householder;
mod lq;
mod ortho;
mod ql;
mod rq;
mod unitary;

pub use col_pivot::{QrPivot, QrPivotError};
pub use complete_orthogonal::CompleteOrthogonalDecomp;
pub use householder::{Qr, QrError};
pub use lq::Lq;
pub use ortho::{OrthoError, Side, Trans, orgqr, ormqr, ungqr, unmqr};
pub use ql::Ql;
pub use rq::Rq;
pub use unitary::{UnitaryQr, UnitaryQrError};
