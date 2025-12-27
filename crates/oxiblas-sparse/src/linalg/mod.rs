//! Sparse linear algebra operations.
//!
//! This module provides:
//! - Sparse Cholesky decomposition (with IC0, ICT preconditioners)
//! - Sparse LU decomposition (with ILU0, ILUT, ILUTP preconditioners)
//! - Supernodal factorizations (SupernodalCholesky, SupernodalLU) for BLAS-3 performance
//! - Sparse QR decomposition
//! - Sparse triangular solvers
//! - Iterative solvers (CG, BiCGStab, GMRES, MINRES, TFQMR, QMR, Block-CG, IDR(s), Block-GMRES)
//! - Convergence monitoring (stagnation detection, divergence detection, rate estimation)
//! - Eigenvalue solvers (Lanczos, Arnoldi, IRAM, Block Lanczos, Block Arnoldi)
//! - SVD solvers (Truncated SVD, Randomized SVD, Incremental SVD)
//! - Preconditioners (Jacobi, Block Jacobi, Gauss-Seidel, SOR, SSOR, AMG, SPAI, AINV, Additive Schwarz)
//! - Fill-reducing orderings (RCM, AMD, Nested Dissection)

pub mod cholesky;
pub mod convergence;
pub mod eigenvalue;
pub mod iterative;
pub mod lu;
pub mod ordering;
pub mod precond;
pub mod qr;
pub mod supernodal;
pub mod svd;
pub mod triangular;

pub use cholesky::{IC0, ICT, SparseCholError, SparseCholesky};
pub use convergence::{
    ConvergenceConfig, ConvergenceInfo, ConvergenceMonitor, ConvergenceStatus, StoppingCriteria,
    asymptotic_convergence_factor, estimate_iterations_to_convergence,
};
pub use eigenvalue::{
    Arnoldi, ArnoldiResult, BlockArnoldi, BlockArnoldiConfig, BlockArnoldiResult, BlockLanczos,
    BlockLanczosConfig, BlockLanczosResult, EigenvalueError, GeneralizedEigen,
    GeneralizedEigenConfig, GeneralizedEigenResult, GeneralizedMode, IRAM, IRAMConfig, IRAMResult,
    IntervalEigen, IntervalEigenConfig, IntervalEigenResult, Lanczos, LanczosConfig, LanczosResult,
    PolynomialFilterConfig, PolynomialFilteredLanczos, PolynomialFilteredResult, ShiftInvertConfig,
    ShiftInvertLanczos, ShiftInvertResult, WhichEigenvalues, count_eigenvalues_in_interval,
    eigenvalues_in_interval, polynomial_filtered_eigenvalues,
};
pub use iterative::{
    BlockCgResult, BlockGmresResult, CgResult, FgmresResult, GmresResult, IdrSResult,
    IterativeError, MinresResult, QmrResult, TfqmrResult, bicgstab, block_cg, block_gmres,
    block_pcg, cg, fgmres, fgmres_ir, gmres, idrs, minres, pgmres, pidrs, pminres, pqmr, ptfqmr,
    qmr, tfqmr,
};
pub use lu::{ILU0, ILUT, ILUTP, SparseLU, SparseLUError};
pub use ordering::{
    EliminationTree, NestedDissectionConfig, SymbolicCholesky, approximate_minimum_degree,
    nested_dissection, reverse_cuthill_mckee,
};
pub use precond::{
    AINV, AINVConfig, AMG, AMGConfig, AMGCycleType, AdditiveSchwarz, AdditiveSchwarzConfig,
    BlockJacobi, GaussSeidel, Jacobi, LocalSolverType, Polynomial, PolynomialConfig,
    PolynomialType, PreconditionerError, SAMG, SAMGConfig, SOR, SPAI, SPAIConfig, SSOR,
};
pub use qr::{SparseQR, SparseQRError, SparseQRGivens};
pub use supernodal::{SupernodalCholesky, SupernodalError, SupernodalLU, Supernode};
pub use svd::{
    IncrementalSVD, IncrementalSVDConfig, RandomizedSparseSvd, RandomizedSparseSvdConfig,
    RandomizedSparseSvdResult, SVDError, TruncatedSVD, TruncatedSVDConfig, TruncatedSVDResult,
    randomized_sparse_svd,
};
pub use triangular::{solve_lower, solve_upper};

/// Prelude for convenient imports.
pub mod prelude {
    pub use super::cholesky::{IC0, ICT, SparseCholError, SparseCholesky};
    pub use super::convergence::{
        ConvergenceConfig, ConvergenceInfo, ConvergenceMonitor, ConvergenceStatus,
        StoppingCriteria, asymptotic_convergence_factor, estimate_iterations_to_convergence,
    };
    pub use super::eigenvalue::{
        Arnoldi, ArnoldiResult, BlockArnoldi, BlockArnoldiConfig, BlockArnoldiResult, BlockLanczos,
        BlockLanczosConfig, BlockLanczosResult, EigenvalueError, GeneralizedEigen,
        GeneralizedEigenConfig, GeneralizedEigenResult, GeneralizedMode, IRAM, IRAMConfig,
        IRAMResult, IntervalEigen, IntervalEigenConfig, IntervalEigenResult, Lanczos,
        LanczosConfig, LanczosResult, PolynomialFilterConfig, PolynomialFilteredLanczos,
        PolynomialFilteredResult, ShiftInvertConfig, ShiftInvertLanczos, ShiftInvertResult,
        WhichEigenvalues, count_eigenvalues_in_interval, eigenvalues_in_interval,
        polynomial_filtered_eigenvalues,
    };
    pub use super::iterative::{
        BlockCgResult, BlockGmresResult, CgResult, FgmresResult, GmresResult, IdrSResult,
        IterativeError, MinresResult, QmrResult, TfqmrResult, bicgstab, block_cg, block_gmres,
        block_pcg, cg, fgmres, fgmres_ir, gmres, idrs, minres, pgmres, pidrs, pminres, pqmr,
        ptfqmr, qmr, tfqmr,
    };
    pub use super::lu::{ILU0, ILUT, ILUTP, SparseLU, SparseLUError};
    pub use super::ordering::{
        EliminationTree, NestedDissectionConfig, SymbolicCholesky, approximate_minimum_degree,
        nested_dissection, reverse_cuthill_mckee,
    };
    pub use super::precond::{
        AINV, AINVConfig, AMG, AMGConfig, AMGCycleType, AdditiveSchwarz, AdditiveSchwarzConfig,
        BlockJacobi, GaussSeidel, Jacobi, LocalSolverType, Polynomial, PolynomialConfig,
        PolynomialType, PreconditionerError, SAMG, SAMGConfig, SOR, SPAI, SPAIConfig, SSOR,
    };
    pub use super::qr::{SparseQR, SparseQRError, SparseQRGivens};
    pub use super::supernodal::{SupernodalCholesky, SupernodalError, SupernodalLU, Supernode};
    pub use super::svd::{
        IncrementalSVD, IncrementalSVDConfig, RandomizedSparseSvd, RandomizedSparseSvdConfig,
        RandomizedSparseSvdResult, SVDError, TruncatedSVD, TruncatedSVDConfig, TruncatedSVDResult,
        randomized_sparse_svd,
    };
    pub use super::triangular::{solve_lower, solve_upper};
}
