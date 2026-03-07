//! Linear system solvers.
//!
//! This module provides solvers for various types of linear systems:
//!
//! - **Triangular solvers**: Forward/back substitution for triangular systems
//! - **General solvers**: LU-based solvers for general square systems
//! - **Expert solvers**: Advanced solve with equilibration and error bounds
//! - **Least squares**: QR-based solvers for overdetermined systems
//! - **Tridiagonal solvers**: O(n) Thomas algorithm for tridiagonal systems
//!
//! # Example
//!
//! ```
//! use oxiblas_lapack::solve::{solve, solve_triangular, tridiag_solve};
//! use oxiblas_matrix::Mat;
//!
//! // Solve Ax = b where A is general
//! let a = Mat::from_rows(&[
//!     &[2.0f64, 1.0],
//!     &[1.0, 3.0],
//! ]);
//! let b = Mat::from_rows(&[&[5.0], &[7.0]]);
//!
//! let x = solve(a.as_ref(), b.as_ref()).unwrap();
//! // x ≈ [1.6, 1.8]
//!
//! // Solve tridiagonal system
//! let dl = [-1.0f64, -1.0];  // sub-diagonal
//! let d = [2.0f64, 2.0, 2.0]; // main diagonal
//! let du = [-1.0f64, -1.0];   // super-diagonal
//! let b_tri = [1.0f64, 0.0, 1.0];
//!
//! let x_tri = tridiag_solve(&dl, &d, &du, &b_tri).unwrap();
//! // x_tri = [1, 1, 1]
//! ```

pub mod expert;
pub mod expert_cholesky;
pub mod expert_symmetric;
mod general;
pub mod iterative_refinement;
mod least_squares;
pub mod mixed_precision;
mod triangular;
mod tridiagonal;

pub use expert::{Equilibrate, ExpertSolveError, ExpertSolveResult, solve_expert};
pub use expert_cholesky::{
    ExpertCholeskySolveError, ExpertCholeskySolveResult, solve_cholesky_expert,
};
pub use expert_symmetric::{
    ExpertSymmetricSolveError, ExpertSymmetricSolveResult, solve_symmetric_expert,
};
pub use general::{SolveError, solve, solve_multiple};
pub use iterative_refinement::{
    RefinementError, RefinementResult, refine_solution, refine_solution_cholesky,
    refine_solution_symmetric,
};
pub use least_squares::{LeastSquaresResult, LstSqError, lstsq};
pub use mixed_precision::{
    MixedPrecisionResult, mixed_precision_solve, mixed_precision_solve_cholesky,
    mixed_precision_solve_lu, mixed_precision_solve_qr, mixed_precision_solve_symmetric,
};
pub use triangular::{
    TriangularKind, TriangularSolveError, solve_triangular, solve_triangular_multiple,
};
pub use tridiagonal::{
    TridiagError, TridiagFactors, TridiagSPDFactors, tridiag_factor, tridiag_factor_spd,
    tridiag_solve, tridiag_solve_factored, tridiag_solve_factored_spd, tridiag_solve_multiple,
    tridiag_solve_spd,
};
