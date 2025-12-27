//! Matrix utilities.
//!
//! This module provides common matrix utility functions:
//!
//! - **Determinant**: Compute matrix determinant
//! - **Inverse**: Compute matrix inverse
//! - **Pseudoinverse**: Moore-Penrose pseudoinverse via SVD
//! - **Condition number**: Estimate matrix conditioning
//! - **Norms**: Various matrix norms
//! - **Rank**: Numerical rank estimation
//! - **Kronecker product**: Tensor product and related operations
//! - **Matrix functions**: expm, logm, sqrtm, powm, signm, cosm, sinm
//! - **Balancing**: gebal/gebak for eigenvalue preprocessing
//! - **Equilibration**: Row/column scaling (geequ, geequb, syequ)
//! - **Error bounds**: Backward/forward error analysis for linear systems
//!
//! # Example
//!
//! ```
//! use oxiblas_lapack::utils::{det, inv, pinv, cond, norm_frobenius, rank, kron, expm};
//! use oxiblas_matrix::Mat;
//!
//! let a = Mat::from_rows(&[
//!     &[4.0f64, 7.0],
//!     &[2.0, 6.0],
//! ]);
//!
//! let determinant = det(a.as_ref()).unwrap();
//! let inverse = inv(a.as_ref()).unwrap();
//! let condition = cond(a.as_ref()).unwrap();
//! let frobenius = norm_frobenius(a.as_ref());
//!
//! // Kronecker product
//! let b = Mat::<f64>::eye(2);
//! let k = kron(a.as_ref(), b.as_ref());
//! assert_eq!(k.nrows(), 4);
//!
//! // Matrix exponential
//! let exp_a = expm(a.as_ref()).unwrap();
//! ```

mod condition;
mod determinant;
pub mod equilibrate;
pub mod error_bounds;
mod inverse;
mod kronecker;
mod matfun;
mod norms;
mod rank_null;

pub use crate::evd::balance::{Balance, BalanceError, BalanceJob, BalanceSide, gebak, gebal};
pub use condition::{CondError, cond, cond_1, cond_inf, rcond, rcond_estimate};
pub use determinant::{DetError, det, det_lu};
pub use equilibrate::{
    EquilibrateError, EquilibrationInfo, apply_col_scale, apply_row_scale, apply_scale, geequ,
    geequb, scale_rhs, syequ, unscale_solution,
};
pub use error_bounds::{
    LinearSystemError, analyze_linear_system_error, backward_error, compute_residual,
    eigenvalue_residual, forward_error_bound, matrix_norm_1, matrix_norm_frobenius,
    matrix_norm_inf, orthogonality_defect, relative_residual_norm, residual_norm,
    svd_singular_value_residual, vector_norm_2, vector_norm_inf,
};
pub use inverse::{InvError, PinvResult, inv, pinv, pinv_default};
pub use kronecker::{
    commutation_matrix, duplication_matrix, elimination_matrix, khatri_rao, kron, kron_sum,
    kron_vec, unvec, vec_mat,
};
pub use matfun::{
    MatFunError, cond_expm, cosm, expm, frechet_expm, frechet_logm, frechet_sqrtm, logm, powm,
    signm, sinm, sqrtm,
};
pub use norms::{norm_1, norm_2, norm_frobenius, norm_inf, norm_max, norm_nuclear, trace};
pub use rank_null::{RankError, col_space, left_null_space, null_space, nullity, rank, row_space};
