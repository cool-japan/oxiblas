//! OxiBLAS LAPACK - Pure Rust LAPACK implementation.
//!
//! This crate provides LAPACK (Linear Algebra PACKage) operations
//! implemented in pure Rust.
//!
//! # Decompositions
//!
//! - **LU**: LU decomposition with partial/full pivoting ✓
//! - **Cholesky**: LL^T and LDL^T decomposition for positive definite matrices ✓
//! - **QR**: QR decomposition with optional column pivoting ✓
//! - **EVD**: Eigenvalue decomposition (symmetric and general) ✓
//! - **SVD**: Singular value decomposition (Jacobi, divide-and-conquer, randomized) ✓
//!
//! # Example
//!
//! ```
//! use oxiblas_lapack::lu::Lu;
//! use oxiblas_matrix::Mat;
//!
//! let a: Mat<f64> = Mat::from_rows(&[
//!     &[2.0, 1.0],
//!     &[1.0, 3.0],
//! ]);
//!
//! let lu = Lu::compute(a.as_ref()).expect("Matrix is not singular");
//! let det = lu.determinant();
//! assert!((det - 5.0).abs() < 1e-10); // det = 2*3 - 1*1 = 5
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]
// LAPACK code often uses manual assign patterns for clarity
#![allow(clippy::assign_op_pattern)]
// Numerical code often has complex generic types
#![allow(clippy::type_complexity)]
// Loop index variables are common in matrix operations
#![allow(clippy::needless_range_loop)]
// LAPACK functions have many parameters by design
#![allow(clippy::too_many_arguments)]
// Bounds in two places for clarity in generic numerical code
#![allow(clippy::multiple_bound_locations)]
// Doc formatting variations
#![allow(clippy::doc_overindented_list_items)]
// Vec vs slice in internal APIs is acceptable
#![allow(clippy::ptr_arg)]
// Manual slice copying for explicit control
#![allow(clippy::manual_memcpy)]
// Balance algorithm has intentional range mutation
#![allow(clippy::mut_range_bound)]
// Iterative algorithms may have unusual patterns
#![allow(clippy::collapsible_if)]
#![allow(clippy::only_used_in_recursion)]
#![allow(clippy::manual_strip)]
#![allow(clippy::iter_cloned_collect)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::eq_op)]
#![allow(clippy::unwrap_or_default)]
#![allow(clippy::unnecessary_unwrap)]
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod cholesky;
pub mod error;
pub mod evd;
pub mod info;
pub mod lu;
pub mod qr;
pub mod solve;
pub mod svd;
pub mod utils;
pub mod workspace;

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::cholesky::{
        BandCholesky, BandCholeskyError, Cholesky, CholeskyError, Ldlt, LdltError,
        band_lower_to_dense, dense_to_band_lower,
    };
    pub use crate::error::{
        ErrorCategory, ErrorCode, HasInfoCode, INFO_SUCCESS, InfoCode, IntoLapackError,
        LapackError, LapackResult,
    };
    pub use crate::evd::{
        Eigenvalue, EigenvalueSelector, GeneralEvd, GeneralEvdError, Hessenberg, HessenbergError,
        Schur, SchurError, SymmetricEvd, SymmetricEvdDc, SymmetricEvdDcError, SymmetricEvdError,
        TridiagEvd, TridiagEvdError, count_eigenvalues, eigenvalue_bounds, eigenvalues_by_index,
        eigenvalues_in_range,
    };
    pub use crate::info::{
        CholeskyInfo, EvdInfo, LuInfo, QrInfo, SvdInfo, SymmetricEvdInfo, compute_cholesky_info,
        compute_general_evd_info, compute_lu_info, compute_qr_info, compute_svd_info,
        compute_symmetric_evd_info,
    };
    pub use crate::lu::{
        BandLu, BandLuError, Lu, LuError, LuFullPiv, LuFullPivError, band_to_dense, dense_to_band,
    };
    pub use crate::qr::{Qr, QrError, QrPivot, QrPivotError};
    pub use crate::solve::{
        Equilibrate, ExpertCholeskySolveError, ExpertCholeskySolveResult, ExpertSolveError,
        ExpertSolveResult, ExpertSymmetricSolveError, ExpertSymmetricSolveResult,
        LeastSquaresResult, LstSqError, RefinementError, RefinementResult, SolveError,
        TriangularKind, TriangularSolveError, TridiagError, TridiagFactors, TridiagSPDFactors,
        lstsq, refine_solution, refine_solution_cholesky, refine_solution_symmetric, solve,
        solve_cholesky_expert, solve_expert, solve_multiple, solve_symmetric_expert,
        solve_triangular, solve_triangular_multiple, tridiag_factor, tridiag_factor_spd,
        tridiag_solve, tridiag_solve_factored, tridiag_solve_factored_spd, tridiag_solve_multiple,
        tridiag_solve_spd,
    };
    pub use crate::svd::{
        BidiagError, BidiagFactors, BidiagVect, RandomizedSvd, RandomizedSvdConfig,
        RandomizedSvdError, SelectiveSvd, SelectiveSvdError, Side, SingularValueSelector, Svd,
        SvdDc, SvdDcError, SvdError, Trans, count_singular_values_above, gebrd,
        low_rank_approximation, orgbr, ormbr, rsvd, rsvd_power, singular_value_bounds, ungbr,
        unmbr,
    };
    pub use crate::utils::{
        CondError, DetError, InvError, PinvResult, RankError, col_space, cond, cond_1, cond_inf,
        det, det_lu, inv, left_null_space, norm_1, norm_2, norm_frobenius, norm_inf, norm_max,
        norm_nuclear, null_space, nullity, pinv, pinv_default, rank, rcond, rcond_estimate,
        row_space, trace,
    };
    pub use crate::workspace::{
        EvdWorkspaceQuery, SvdWorkspaceQuery, Workspace, WorkspaceQuery, WorkspaceQueryWithInt,
        band_lu_workspace, bidiag_workspace, cholesky_solve_workspace, cholesky_workspace,
        general_evd_workspace, generalized_evd_workspace, hermitian_evd_workspace,
        hessenberg_workspace, ldlt_workspace, least_squares_workspace, lu_solve_workspace,
        lu_workspace, orgqr_workspace, ormqr_workspace, qr_pivot_workspace, qr_workspace,
        qz_workspace, schur_workspace, svd_dc_workspace, svd_workspace, symmetric_evd_dc_workspace,
        symmetric_evd_workspace, triangular_solve_workspace, tridiagonal_solve_workspace,
    };
}
