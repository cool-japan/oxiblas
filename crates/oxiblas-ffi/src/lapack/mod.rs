//! LAPACK FFI - Linear algebra routines.
//!
//! This module provides C-compatible FFI for LAPACK routines:
//!
//! ## Factorization
//! - `oblas_sgetrf`, `oblas_dgetrf`, `oblas_zgetrf` - LU factorization
//! - `oblas_spotrf`, `oblas_dpotrf` - Cholesky factorization
//! - `oblas_sgeqrf`, `oblas_dgeqrf` - QR factorization
//!
//! ## Solve
//! - `oblas_sgesv`, `oblas_dgesv` - General linear system solve
//! - `oblas_sgetrs`, `oblas_dgetrs` - Solve from LU factors
//! - `oblas_spotrs`, `oblas_dpotrs` - Solve from Cholesky factors
//! - `oblas_sgels`, `oblas_dgels` - Least squares solve
//!
//! ## Expert Solve
//! - `oblas_sgesvx`, `oblas_dgesvx` - Expert general solve with equilibration
//! - `oblas_sposvx`, `oblas_dposvx` - Expert Cholesky solve with equilibration
//! - `oblas_ssysvx`, `oblas_dsysvx` - Expert symmetric solve with equilibration
//!
//! ## Iterative Refinement
//! - `oblas_sgerfs`, `oblas_dgerfs` - General system iterative refinement
//! - `oblas_sporfs`, `oblas_dporfs` - SPD system iterative refinement (Cholesky)
//! - `oblas_ssyrfs`, `oblas_dsyrfs` - Symmetric system iterative refinement (LDL^T)
//!
//! ## Eigenvalue Decomposition
//! - `oblas_ssyev`, `oblas_dsyev` - Symmetric eigenvalues (QR algorithm)
//! - `oblas_ssyevd`, `oblas_dsyevd` - Symmetric eigenvalues (divide-and-conquer)
//! - `oblas_sgeev`, `oblas_dgeev` - General eigenvalues
//!
//! ## Singular Value Decomposition
//! - `oblas_sgesvd`, `oblas_dgesvd` - SVD (full)
//! - `oblas_sgesvd_thin`, `oblas_dgesvd_thin` - Thin (economy) SVD
//! - `oblas_sgesdd_thin`, `oblas_dgesdd_thin` - Thin SVD using divide-and-conquer
//!
//! ## Inverse
//! - `oblas_sinv`, `oblas_dinv` - Direct matrix inverse
//! - `oblas_sgetri`, `oblas_dgetri` - Inverse from LU factors
//! - `oblas_spotri`, `oblas_dpotri` - Inverse from Cholesky factors
//!
//! ## Orthogonal Matrix Operations
//! - `oblas_sorgqr`, `oblas_dorgqr` - Generate Q from QR
//! - `oblas_sormqr`, `oblas_dormqr` - Multiply by Q
//!
//! ## Condition Number
//! - `oblas_sgecon`, `oblas_dgecon` - Condition number estimation
//! - `oblas_scond`, `oblas_dcond` - 2-norm condition number
//! - `oblas_scond1`, `oblas_dcond1` - 1-norm condition number
//! - `oblas_scondinf`, `oblas_dcondinf` - Infinity-norm condition number
//! - `oblas_srcond`, `oblas_drcond` - Reciprocal condition number
//! - `oblas_srcond_est`, `oblas_drcond_est` - Fast reciprocal condition estimate
//!
//! ## Matrix Norms
//! - `oblas_slange`, `oblas_dlange` - General matrix norm (1, inf, max, Frobenius)
//! - `oblas_slansy`, `oblas_dlansy` - Symmetric matrix norm
//! - `oblas_slantr`, `oblas_dlantr` - Triangular matrix norm
//! - `oblas_snorm2`, `oblas_dnorm2` - Spectral norm (2-norm via SVD)
//!
//! ## Determinant
//! - `oblas_sdet`, `oblas_ddet` - Matrix determinant
//! - `oblas_sabsdet`, `oblas_dabsdet` - Absolute value of determinant
//! - `oblas_sdetlu`, `oblas_ddetlu` - Determinant from LU factors
//! - `oblas_slogdet`, `oblas_dlogdet` - Log-determinant
//! - `oblas_sdet_chol`, `oblas_ddet_chol` - Determinant via Cholesky (for SPD matrices)
//! - `oblas_slogdet_chol`, `oblas_dlogdet_chol` - Log-determinant via Cholesky (for SPD matrices)
//!
//! ## Pseudoinverse
//! - `oblas_spinv`, `oblas_dpinv` - Moore-Penrose pseudoinverse
//! - `oblas_spinv_default`, `oblas_dpinv_default` - Pseudoinverse with automatic tolerance
//!
//! ## Matrix Functions
//! - `oblas_sexpm`, `oblas_dexpm` - Matrix exponential
//! - `oblas_slogm`, `oblas_dlogm` - Matrix logarithm
//! - `oblas_ssqrtm`, `oblas_dsqrtm` - Matrix square root
//! - `oblas_spowm`, `oblas_dpowm` - Matrix power (integer exponent)
//!
//! ## Matrix Rank
//! - `oblas_srank`, `oblas_drank` - Matrix rank
//! - `oblas_snullity`, `oblas_dnullity` - Matrix nullity
//!
//! ## Trace and Nuclear Norm
//! - `oblas_strace`, `oblas_dtrace` - Matrix trace
//! - `oblas_snormnuc`, `oblas_dnormnuc` - Nuclear norm
//!
//! ## Kronecker Product
//! - `oblas_skron`, `oblas_dkron` - Kronecker product
//! - `oblas_skronsum`, `oblas_dkronsum` - Kronecker sum
//! - `oblas_skron_vec`, `oblas_dkron_vec` - Efficient Kronecker-vector product
//!
//! ## Schur Decomposition
//! - `oblas_sgees`, `oblas_dgees` - Schur decomposition A = Q T Q^T
//!
//! ## LDL^T Factorization
//! - `oblas_ssytrf`, `oblas_dsytrf` - LDL^T factorization for symmetric matrices
//! - `oblas_ssytrs`, `oblas_dsytrs` - Solve from LDL^T factors
//! - `oblas_sinertia`, `oblas_dinertia` - Matrix inertia (eigenvalue sign counts)
//! - `oblas_sisposdef`, `oblas_disposdef` - Positive definiteness check
//! - `oblas_sisnegdef`, `oblas_disnegdef` - Negative definiteness check
//! - `oblas_slogabsdet_ldlt`, `oblas_dlogabsdet_ldlt` - Log absolute determinant via LDLT
//!
//! ## Hessenberg Reduction
//! - `oblas_sgehrd`, `oblas_dgehrd` - Reduce to upper Hessenberg form A = Q H Q^T
//!
//! ## Generalized Eigenvalue Decomposition
//! - `oblas_ssygv`, `oblas_dsygv` - Symmetric generalized EVD (B positive definite)
//! - `oblas_sggev`, `oblas_dggev` - General generalized EVD
//!
//! ## QZ Decomposition (Generalized Schur)
//! - `oblas_sgges`, `oblas_dgges` - Generalized Schur decomposition
//!
//! ## QR with Column Pivoting
//! - `oblas_sgeqp3`, `oblas_dgeqp3` - Rank-revealing QR factorization
//!
//! ## LU with Full Pivoting
//! - `oblas_sgetc2`, `oblas_dgetc2` - LU with complete pivoting (PAQ = LU)
//! - `oblas_sgesc2`, `oblas_dgesc2` - Solve from LU full pivoting factors
//!
//! ## Subspace Computations
//! - `oblas_snull`, `oblas_dnull` - Null space basis
//! - `oblas_scolspace`, `oblas_dcolspace` - Column space basis
//! - `oblas_srowspace`, `oblas_drowspace` - Row space basis
//! - `oblas_slnull`, `oblas_dlnull` - Left null space basis
//!
//! ## SVD Divide-and-Conquer
//! - `oblas_sgesdd`, `oblas_dgesdd` - SVD using divide-and-conquer algorithm
//!
//! ## Additional Kronecker Operations
//! - `oblas_skhatri_rao`, `oblas_dkhatri_rao` - Khatri-Rao product
//! - `oblas_svec`, `oblas_dvec` - Vectorize matrix
//! - `oblas_sunvec`, `oblas_dunvec` - Reshape vector to matrix
//! - `oblas_scommutation`, `oblas_dcommutation` - Commutation matrix
//! - `oblas_sduplication`, `oblas_dduplication` - Duplication matrix
//! - `oblas_selimination`, `oblas_delimination` - Elimination matrix
//!
//! ## Tridiagonal Systems
//! - `oblas_sgtsv`, `oblas_dgtsv` - General tridiagonal solve (Thomas algorithm)
//! - `oblas_sgttrf`, `oblas_dgttrf` - Tridiagonal LU factorization
//! - `oblas_sgttrs`, `oblas_dgttrs` - Solve from tridiagonal factors
//! - `oblas_sptsv`, `oblas_dptsv` - SPD tridiagonal solve
//!
//! ## Band Matrix Operations
//! - `oblas_sgbtrf`, `oblas_dgbtrf` - Band LU factorization
//! - `oblas_sgbtrs`, `oblas_dgbtrs` - Solve from band LU factors
//! - `oblas_sgbsv`, `oblas_dgbsv` - Direct band system solve
//! - `oblas_spbtrf`, `oblas_dpbtrf` - Band Cholesky factorization (SPD)
//! - `oblas_spbtrs`, `oblas_dpbtrs` - Solve from band Cholesky factors
//! - `oblas_spbsv`, `oblas_dpbsv` - Direct band SPD system solve

pub mod bandmatrix;
pub mod condition;
pub mod determinant;
pub mod eigen;
pub mod factorization;
pub mod gevd;
pub mod hessenberg;
pub mod inverse;
pub mod kronecker;
pub mod ldlt;
pub mod lufullpiv;
pub mod matfun;
pub mod norms;
pub mod orthogonal;
pub mod pseudoinverse;
pub mod qrpivot;
pub mod qz;
pub mod rank;
pub mod schur;
pub mod solve;
pub mod subspace;
pub mod svd;
pub mod trace;
pub mod tridiagonal;

// Re-export all public functions
pub use bandmatrix::*;
pub use condition::*;
pub use determinant::*;
pub use eigen::*;
pub use factorization::*;
pub use gevd::*;
pub use hessenberg::*;
pub use inverse::*;
pub use kronecker::*;
pub use ldlt::*;
pub use lufullpiv::*;
pub use matfun::*;
pub use norms::*;
pub use orthogonal::*;
pub use pseudoinverse::*;
pub use qrpivot::*;
pub use qz::*;
pub use rank::*;
pub use schur::*;
pub use solve::*;
pub use subspace::*;
pub use svd::*;
pub use trace::*;
pub use tridiagonal::*;
