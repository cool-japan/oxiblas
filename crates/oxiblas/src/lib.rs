//! # OxiBLAS - Pure Rust BLAS/LAPACK Implementation
//!
//! OxiBLAS is a pure Rust implementation of BLAS (Basic Linear Algebra Subprograms)
//! and LAPACK (Linear Algebra PACKage), designed to be the defacto standard for
//! the scirs2 ecosystem.
//!
//! ## Features
//!
//! - **Pure Rust**: No C dependencies, fully portable
//! - **SIMD Optimized**: Custom SIMD layer using `core::arch` intrinsics
//! - **Cache Aware**: BLIS-style blocked algorithms for optimal cache usage
//! - **Parallel**: Optional rayon-based parallelization
//!
//! ## Crate Structure
//!
//! OxiBLAS is organized into several sub-crates:
//!
//! - `oxiblas_core`: Core traits and SIMD abstractions
//! - `oxiblas_matrix`: Matrix types (`Mat`, `MatRef`, `MatMut`)
//! - `oxiblas_blas`: BLAS operations (gemm, etc.)
//! - `oxiblas_lapack`: LAPACK operations (decompositions, solvers)
//! - `oxiblas_sparse`: Sparse matrix types and operations (with `sparse` feature, default)
//! - `oxiblas_ndarray`: ndarray integration (with `ndarray` feature)
//!
//! ## Quick Start
//!
//! ```
//! use oxiblas::prelude::*;
//!
//! // Create matrices
//! let a: Mat<f64> = Mat::filled(100, 50, 1.0);
//! let b: Mat<f64> = Mat::filled(50, 80, 2.0);
//! let mut c: Mat<f64> = Mat::zeros(100, 80);
//!
//! // GEMM: C = A * B
//! gemm(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());
//!
//! // Each element of C should be 100 (50 * 1.0 * 2.0)
//! assert!((c[(0, 0)] - 100.0).abs() < 1e-10);
//! ```
//!
//! ## Supported Types
//!
//! - `f32`: Single precision floating point
//! - `f64`: Double precision floating point
//! - `Complex32`: Single precision complex
//! - `Complex64`: Double precision complex
//!
//! ## SIMD Support
//!
//! OxiBLAS automatically detects and uses the best available SIMD:
//!
//! - **x86_64**: SSE4.2 (128-bit), AVX2/FMA (256-bit), AVX-512F (512-bit)
//!   - AVX-512BW for byte/word operations (i8, i16 SIMD)
//!   - AVX-512VNNI for neural network acceleration (quantized inference)
//! - **AArch64**: NEON (128-bit), SVE (128-2048 bit scalable vectors)
//! - **WASM**: SIMD128 (128-bit)
//! - **Fallback**: Scalar operations
//!
//! ## Scalar Specialization
//!
//! For performance-critical code, OxiBLAS provides compile-time type dispatch:
//!
//! - `SimdCompatible`: SIMD width hints per scalar type
//! - `ScalarBatch`: Vectorized batch operations (dot, axpy, fma)
//! - `ExtendedPrecision`: Accumulator type mapping (f32→f64)
//! - `KahanSum`, `pairwise_sum`: Compensated summation for accuracy
//!
//! ## Comparison with Other Libraries
//!
//! | Feature | OxiBLAS | ndarray-linalg | nalgebra | faer |
//! |---------|---------|----------------|----------|------|
//! | Pure Rust | yes | no (LAPACKE) | yes | yes |
//! | no_std | partial | no | yes | yes |
//! | Sparse support | yes | no | partial | no |
//! | Complex numbers | yes | yes | yes | partial |
//! | BLAS-compatible API | yes | yes | no | no |
//! | Extended precision | yes | no | no | no |
//! | Parallel (Rayon) | yes | partial | no | yes |
//!
//! ## Feature Flags
//!
//! - `sparse`: Enable sparse matrix operations (enabled by default)
//! - `parallel`: Enable rayon-based parallelization
//! - `ndarray`: Enable ndarray integration for interop with ndarray types
//! - `nalgebra`: Enable nalgebra integration for interop with nalgebra types
//! - `mmap`: Enable memory-mapped matrices for large datasets
//! - `f16`: Enable half-precision (f16) support
//! - `f128`: Enable quad-precision (f128) support
//! - `serde`: Enable serialization support for matrix types
//! - `full`: Enable all features
//!
//! ### SIMD Control Features
//!
//! These features control SIMD optimization levels (useful for debugging or compatibility):
//!
//! - `force-scalar`: Disable all SIMD optimizations, use scalar operations only
//! - `max-simd-128`: Limit SIMD to 128-bit registers (SSE/NEON)
//! - `max-simd-256`: Limit SIMD to 256-bit registers (AVX2)
//!
//! Use `detect_simd_level()` to check the active SIMD level, and
//! `detect_simd_level_raw()` to get the actual hardware capability.
//!
//! ## ndarray Integration
//!
//! With the `ndarray` feature enabled, you can use OxiBLAS with ndarray types:
//!
//! ```ignore
//! use oxiblas::ndarray::prelude::*;
//! use ndarray::Array2;
//!
//! let a = Array2::<f64>::from_shape_fn((100, 100), |idx| (idx.0 + idx.1) as f64);
//! let b = Array2::<f64>::from_shape_fn((100, 100), |idx| (idx.0 * idx.1) as f64);
//! let c = matmul(&a, &b);
//! ```
//!
//! ## nalgebra Integration
//!
//! With the `nalgebra` feature enabled, you can convert between OxiBLAS and nalgebra types:
//!
//! ```ignore
//! use oxiblas::prelude::*;
//! use oxiblas::{mat_to_dmatrix, dmatrix_to_mat, MatNalgebraExt, DMatrixOxiblasExt};
//! use nalgebra::DMatrix;
//!
//! // Convert from nalgebra to OxiBLAS
//! let na_mat = DMatrix::from_fn(3, 3, |row, col| (row + col) as f64);
//! let oxi_mat: Mat<f64> = na_mat.to_mat();
//!
//! // Convert from OxiBLAS to nalgebra
//! let mat: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
//! let dm: DMatrix<f64> = mat.to_dmatrix();
//! ```
//!
//! ## Lazy Evaluation
//!
//! OxiBLAS supports lazy evaluation for matrix operations, enabling expression trees
//! that are only evaluated when `.eval()` is called. This allows for operation fusion
//! and optimization.
//!
//! ```
//! use oxiblas::prelude::*;
//!
//! let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
//! let b: Mat<f64> = Mat::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);
//!
//! // Build an expression tree (no computation yet)
//! let expr = a.as_ref().lazy() + b.as_ref().lazy();
//!
//! // Evaluate to get the result
//! let result = expr.eval();
//! assert!((result[(0, 0)] - 6.0).abs() < 1e-10); // 1 + 5
//!
//! // Chained operations with scaling
//! let scaled = (a.as_ref().lazy() + b.as_ref().lazy()).scale(2.0);
//! let result2 = scaled.eval();
//! assert!((result2[(0, 0)] - 12.0).abs() < 1e-10); // (1 + 5) * 2
//!
//! // Fused multiply-add: alpha * A + beta * B
//! let fma_result = lazy_fma(2.0, a.as_ref().lazy(), 3.0, b.as_ref().lazy()).eval();
//! assert!((fma_result[(0, 0)] - 17.0).abs() < 1e-10); // 2*1 + 3*5
//! ```
//!
//! Available lazy operations:
//! - Element-wise: `+`, `-`, negation, `scale()`
//! - Matrix: `matmul()`, `t()` (transpose)
//! - Complex: `conj()`, `h()` (Hermitian/conjugate transpose)
//! - Fused: `lazy_fma()`, `lazy_gemm()`
//! - Optimizations: `.simplify()` for double-transpose/scale/negation elimination
//!
//! # Performance Guide
//!
//! ## Choosing the Right Algorithm
//!
//! OxiBLAS automatically selects optimized code paths based on matrix size:
//!
//! | Size | GEMM Strategy | Expected Performance |
//! |------|---------------|---------------------|
//! | < 32 | Unrolled small-matrix kernels | Minimal overhead |
//! | 32-512 | BLIS-style blocked GEMM | ~70% peak |
//! | > 512 | Auto-tuned blocking + parallel | ~80% peak |
//! | > 2048 | Consider Strassen (optional) | Reduced complexity |
//!
//! ## Memory Layout
//!
//! - **Column-major** (default): Best for BLAS/LAPACK compatibility
//! - **Contiguous data**: Avoid strided views when possible for 2-10x speedup
//! - **Aligned allocation**: `Mat<T>` uses 64-byte alignment for SIMD
//!
//! ## Parallelization
//!
//! Enable the `parallel` feature and use `Par::Rayon` for large matrices:
//!
//! ```ignore
//! use oxiblas::prelude::*;
//! use oxiblas::core::Par;
//!
//! // Sequential (default)
//! gemm(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());
//!
//! // Parallel - recommended for n >= 256
//! gemm_with_par(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut(), Par::Rayon);
//! ```
//!
//! ## Cache Optimization Tips
//!
//! 1. **Blocking parameters**: Use `GemmBlocking::auto_tuned()` for runtime tuning
//! 2. **Panel reuse**: Process multiple right-hand sides together when possible
//! 3. **Prefetching**: Enabled automatically in GEMM micro-kernels
//!
//! ## Complex Numbers
//!
//! For complex GEMM, the 3M algorithm (3 real multiplies) is used automatically
//! for matrices >= 64, providing ~25% speedup over naive 4-multiply approach.
//!
//! # Migration Guide
//!
//! ## From ndarray-linalg
//!
//! ```ignore
//! // ndarray-linalg
//! use ndarray::Array2;
//! use ndarray_linalg::Solve;
//! let a = Array2::<f64>::eye(3);
//! let b = Array2::<f64>::ones((3, 1));
//! let x = a.solve(&b).unwrap();
//!
//! // OxiBLAS equivalent
//! use oxiblas::prelude::*;
//! let a = MatBuilder::<f64>::identity(3);
//! let b = MatBuilder::<f64>::ones(3, 1);
//! let x = solve(a.as_ref(), b.as_ref()).unwrap();
//! ```
//!
//! ## From nalgebra
//!
//! ```ignore
//! // nalgebra
//! use nalgebra::{DMatrix, DVector};
//! let a = DMatrix::from_fn(3, 3, |i, j| (i + j) as f64);
//! let b = DVector::from_element(3, 1.0);
//! let lu = a.lu();
//! let x = lu.solve(&b).unwrap();
//!
//! // OxiBLAS equivalent (with nalgebra feature)
//! use oxiblas::prelude::*;
//! let a: Mat<f64> = na_matrix.to_mat(); // Convert from nalgebra
//! let b: Mat<f64> = na_vector.to_mat();
//! let x = solve(a.as_ref(), b.as_ref()).unwrap();
//! let result = x.to_dmatrix(); // Convert back if needed
//! ```
//!
//! ## From NumPy (via Rust)
//!
//! | NumPy | OxiBLAS |
//! |-------|---------|
//! | `np.dot(a, b)` | `gemm(1.0, a, b, 0.0, c)` or `a.matmul(&b)` |
//! | `np.linalg.solve(a, b)` | `solve(a, b)` |
//! | `np.linalg.svd(a)` | `Svd::compute(a)` |
//! | `np.linalg.eig(a)` | `GeneralEvd::compute(a)` |
//! | `np.linalg.qr(a)` | `Qr::compute(a)` |
//! | `np.linalg.cholesky(a)` | `Cholesky::compute(a)` |
//! | `np.linalg.inv(a)` | `inv(a)` |
//! | `np.linalg.det(a)` | `det(a)` |
//!
//! # Architecture
//!
//! ## Crate Hierarchy
//!
//! ```text
//! oxiblas (umbrella)
//! ├── oxiblas-core     # Scalar traits, SIMD, memory, parallelism
//! ├── oxiblas-matrix   # Mat/MatRef/MatMut, storage formats
//! ├── oxiblas-blas     # BLAS Level 1/2/3 operations
//! ├── oxiblas-lapack   # Decompositions, solvers, eigenvalue
//! ├── oxiblas-sparse   # Sparse formats, iterative solvers
//! └── oxiblas-ndarray  # ndarray integration
//! ```
//!
//! ## GEMM Optimization Stack
//!
//! ```text
//! User API: gemm(alpha, a, b, beta, c)
//!     │
//!     ▼
//! Size dispatch: small (<32) → unrolled, large → blocked
//!     │
//!     ▼
//! Blocking: MC×KC×NC panels (auto-tuned to cache hierarchy)
//!     │
//!     ▼
//! Packing: pack_a (MC×KC), pack_b (KC×NC) with SIMD
//!     │
//!     ▼
//! Micro-kernel: MR×NR FMA loop (8×6 f64, 8×8 f32 on NEON/AVX2)
//! ```
//!
//! ## SIMD Abstraction
//!
//! The `oxiblas-core` crate provides a unified SIMD abstraction:
//!
//! - `SimdLevel`: Runtime detection (Scalar, SSE42, AVX2, AVX512, NEON, SVE)
//! - `F64x2/F64x4`: Portable f64 vector types with fallback
//! - Architecture-specific intrinsics wrapped in safe functions
//!
//! ## Memory Model
//!
//! - **Arena allocation**: Bump allocator for GEMM temporaries (zero malloc)
//! - **Aligned vectors**: 64-byte aligned for AVX-512 compatibility
//! - **Copy-on-write**: `CowMat<T>` for lazy cloning
//! - **Memory-mapped**: `MmapMat` for large datasets (with `mmap` feature)

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::multiple_bound_locations)]

// Builder patterns for ergonomic matrix creation
pub mod builder;

// Automatic algorithm selection
pub mod auto;

// Fluent API for method chaining
pub mod fluent;

// Re-export sub-crates
pub use oxiblas_blas as blas;
pub use oxiblas_core as core;
pub use oxiblas_lapack as lapack;
pub use oxiblas_matrix as matrix;

/// Sparse matrix operations (requires `sparse` feature, enabled by default)
#[cfg(feature = "sparse")]
pub use oxiblas_sparse as sparse;

/// ndarray integration (requires `ndarray` feature)
#[cfg(feature = "ndarray")]
pub use oxiblas_ndarray as ndarray;

// Re-export commonly used types from core
pub use oxiblas_core::{
    AlignedVec, Field, MemStack, Par, ParThreshold, Real, Scalar, SimdLevel, StackReq,
    detect_simd_level, detect_simd_level_raw,
};

// Re-export scalar specialization traits for performance-critical code
pub use oxiblas_core::{
    ExtendedPrecision, HasFastFma, KBKSum, KahanSum, ScalarBatch, ScalarClass, ScalarClassify,
    SimdCompatible, UnrollHints, pairwise_sum,
};

// Re-export complex number ergonomics
pub use oxiblas_core::{C32, C64, ComplexExt, I32, I64, ToComplex, c32, c64};

// Re-export matrix types
pub use oxiblas_matrix::{DiagRef, Mat, MatMut, MatRef, TransposeRef};

// Re-export memory-mapped matrix types (requires `mmap` feature)
#[cfg(feature = "mmap")]
pub use oxiblas_matrix::{
    MmapBuilder, MmapError, MmapMat, MmapMatMut, read_dimensions, write_mmap,
};

// Re-export nalgebra conversions (requires `nalgebra` feature)
#[cfg(feature = "nalgebra")]
pub use oxiblas_matrix::{
    DMatrixOxiblasExt, MatNalgebraExt, dmatrix_to_mat, dmatrix_view_to_mat, dvector_to_mat,
    mat_ref_to_dmatrix, mat_to_dmatrix, mat_to_dvector,
};

// Re-export lazy evaluation types
pub use oxiblas_matrix::{
    ComplexExpr, ComplexScalar, Expr, ExprAdd, ExprConj, ExprFma, ExprGemm, ExprHermitian,
    ExprLeaf, ExprMul, ExprNeg, ExprScale, ExprSub, ExprTranspose, LazyExt,
};

// Re-export lazy evaluation helper functions
pub use oxiblas_matrix::lazy::{fma as lazy_fma, gemm as lazy_gemm};

// Re-export BLAS operations
pub use oxiblas_blas::level3::{GemmBlocking, GemmKernel, gemm, gemm_with_par};

/// Parallel BLAS and GEMM variants (requires `parallel` feature).
///
/// These functions dispatch work to rayon thread pools and are recommended
/// for matrices with dimensions >= 256.
#[cfg(feature = "parallel")]
pub use oxiblas_core::{CustomRayonPool, RayonGlobalPool};

/// Half-precision (f16) scalar type (requires `f16` feature).
///
/// `half::f16` implements the [`Scalar`] and [`Real`] traits, making it
/// compatible with all OxiBLAS matrix and BLAS operations.
#[cfg(feature = "f16")]
pub use oxiblas_core::f16;

/// Quad-precision scalar type backed by double-double arithmetic (requires `f128` feature).
///
/// [`QuadFloat`] provides ~31 decimal digits of precision using a
/// two-`f64` representation and implements all core OxiBLAS scalar traits.
#[cfg(feature = "f128")]
pub use oxiblas_core::QuadFloat;

/// Compile-time feature availability flags.
///
/// These constants let library users and downstream crates check which
/// optional features are compiled in at compile time without resorting to
/// `cfg` attributes on every call site.
///
/// # Example
///
/// ```
/// use oxiblas::features;
///
/// if features::HAS_PARALLEL {
///     // safe to use gemm_with_par with Par::Rayon
/// }
/// ```
pub mod features {
    /// `true` when the `parallel` feature (Rayon integration) is compiled in.
    pub const HAS_PARALLEL: bool = cfg!(feature = "parallel");
    /// `true` when the `sparse` feature (sparse matrix types and solvers) is compiled in.
    pub const HAS_SPARSE: bool = cfg!(feature = "sparse");
    /// `true` when the `f16` feature (half-precision scalar support) is compiled in.
    pub const HAS_F16: bool = cfg!(feature = "f16");
    /// `true` when the `f128` feature (quad-precision scalar via `QuadFloat`) is compiled in.
    pub const HAS_F128: bool = cfg!(feature = "f128");
    /// `true` when the `ndarray` feature (ndarray interop) is compiled in.
    pub const HAS_NDARRAY: bool = cfg!(feature = "ndarray");
    /// `true` when the `oxiblas-core` std feature is absent (no-std mode for the core layer).
    ///
    /// Note: check `oxiblas_core` feature flags directly for precise no-std detection.
    pub const NO_STD: bool = !cfg!(feature = "default");
}

/// Prelude module - import everything commonly needed.
///
/// The prelude provides convenient access to the most commonly used types
/// and functions in OxiBLAS:
///
/// - Matrix types: `Mat`, `MatRef`, `MatMut`
/// - Builder patterns: `MatBuilder`
/// - Complex numbers: `c32`, `c64`, `C32`, `C64`
/// - BLAS operations: `gemm`, `gemv`, `dot`, etc.
/// - LAPACK decompositions: `Lu`, `Qr`, `Svd`, `Cholesky`, etc.
/// - Sparse matrices: `CsrMatrix`, `CscMatrix`, `CooMatrix`
///
/// # Example
///
/// ```
/// use oxiblas::prelude::*;
///
/// // Create matrices using the builder
/// let a = MatBuilder::<f64>::identity(4);
/// let b = MatBuilder::<f64>::hilbert(4);
/// let mut c = MatBuilder::<f64>::zeros(4, 4);
///
/// // Compute C = A * B
/// gemm(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());
///
/// // Create a complex number
/// let z = c64(1.0, 2.0);
/// assert_eq!(z.re, 1.0);
/// ```
pub mod prelude {
    // Builder patterns
    pub use crate::builder::MatBuilder;

    // Fluent API
    pub use crate::fluent::{MatrixOps, MatrixOpsMut, VectorOps};

    // Core traits
    pub use oxiblas_core::prelude::*;

    // Complex number ergonomics
    pub use oxiblas_core::{C32, C64, ComplexExt, I32, I64, ToComplex, c32, c64};

    // Scalar specialization for power users
    pub use oxiblas_core::{
        ExtendedPrecision, KBKSum, KahanSum, ScalarBatch, SimdCompatible, pairwise_sum,
    };

    // Matrix types
    pub use oxiblas_matrix::prelude::*;

    // Memory-mapped matrices (requires `mmap` feature)
    #[cfg(feature = "mmap")]
    pub use oxiblas_matrix::{MmapBuilder, MmapMat, MmapMatMut};

    // Half-precision scalar type (requires `f16` feature)
    #[cfg(feature = "f16")]
    pub use oxiblas_core::f16;

    // Quad-precision scalar type (requires `f128` feature)
    #[cfg(feature = "f128")]
    pub use oxiblas_core::QuadFloat;

    // Parallel rayon pool types (requires `parallel` feature)
    #[cfg(feature = "parallel")]
    pub use oxiblas_core::{CustomRayonPool, RayonGlobalPool};

    // Lazy evaluation helpers
    pub use crate::{lazy_fma, lazy_gemm};

    // BLAS operations
    pub use oxiblas_blas::prelude::*;

    // Sparse matrix types (commonly used, requires `sparse` feature)
    #[cfg(feature = "sparse")]
    pub use oxiblas_sparse::{CooMatrix, CscMatrix, CsrMatrix};

    // Sparse iterative solvers (commonly used, requires `sparse` feature)
    #[cfg(feature = "sparse")]
    pub use oxiblas_sparse::linalg::{bicgstab, cg, gmres};

    // LAPACK operations - selective imports to avoid ambiguous re-exports
    // Note: Side and Trans are intentionally not imported from LAPACK to avoid
    // conflicts with BLAS versions. Use oxiblas::lapack::svd::{Side, Trans} if needed.
    #[allow(unused_imports)]
    pub use oxiblas_lapack::prelude::{
        BandCholesky, BandCholeskyError, BandLu, BandLuError, BidiagError, BidiagFactors,
        BidiagVect, Cholesky, CholeskyError, CholeskyInfo, CondError, DetError, Eigenvalue,
        EigenvalueSelector, Equilibrate, ErrorCategory, ErrorCode, EvdInfo, EvdWorkspaceQuery,
        ExpertCholeskySolveError, ExpertCholeskySolveResult, ExpertSolveError, ExpertSolveResult,
        ExpertSymmetricSolveError, ExpertSymmetricSolveResult, GeneralEvd, GeneralEvdError,
        HasInfoCode, Hessenberg, HessenbergError, INFO_SUCCESS, InfoCode, IntoLapackError,
        InvError, LapackError, LapackResult, Ldlt, LdltError, LeastSquaresResult, LstSqError, Lu,
        LuError, LuFullPiv, LuFullPivError, LuInfo, PinvResult, Qr, QrError, QrInfo, QrPivot,
        QrPivotError, RandomizedSvd, RandomizedSvdConfig, RandomizedSvdError, RankError,
        RefinementError, RefinementResult, Schur, SchurError, SelectiveSvd, SelectiveSvdError,
        SingularValueSelector, SolveError, Svd, SvdDc, SvdDcError, SvdError, SvdInfo,
        SvdWorkspaceQuery, SymmetricEvd, SymmetricEvdDc, SymmetricEvdDcError, SymmetricEvdError,
        SymmetricEvdInfo, TriangularKind, TriangularSolveError, TridiagError, TridiagEvd,
        TridiagEvdError, TridiagFactors, TridiagSPDFactors, Workspace, WorkspaceQuery,
        WorkspaceQueryWithInt, band_lower_to_dense, band_lu_workspace, band_to_dense,
        bidiag_workspace, cholesky_solve_workspace, cholesky_workspace, col_space,
        compute_cholesky_info, compute_general_evd_info, compute_lu_info, compute_qr_info,
        compute_svd_info, compute_symmetric_evd_info, cond, cond_1, cond_inf, count_eigenvalues,
        count_singular_values_above, dense_to_band, dense_to_band_lower, det, det_lu,
        eigenvalue_bounds, eigenvalues_by_index, eigenvalues_in_range, gebrd,
        general_evd_workspace, generalized_evd_workspace, hermitian_evd_workspace,
        hessenberg_workspace, inv, ldlt_workspace, least_squares_workspace, left_null_space,
        low_rank_approximation, lstsq, lu_solve_workspace, lu_workspace, norm_1, norm_2,
        norm_frobenius, norm_inf, norm_max, norm_nuclear, null_space, nullity, orgbr,
        orgqr_workspace, ormbr, ormqr_workspace, pinv, pinv_default, qr_pivot_workspace,
        qr_workspace, qz_workspace, rank, rcond, rcond_estimate, refine_solution,
        refine_solution_cholesky, refine_solution_symmetric, row_space, rsvd, rsvd_power,
        schur_workspace, singular_value_bounds, solve, solve_cholesky_expert, solve_expert,
        solve_multiple, solve_symmetric_expert, solve_triangular, solve_triangular_multiple,
        svd_dc_workspace, svd_workspace, symmetric_evd_dc_workspace, symmetric_evd_workspace,
        trace, triangular_solve_workspace, tridiag_factor, tridiag_factor_spd, tridiag_solve,
        tridiag_solve_factored, tridiag_solve_factored_spd, tridiag_solve_multiple,
        tridiag_solve_spd, tridiagonal_solve_workspace, ungbr, unmbr,
    };
}

#[cfg(test)]
mod tests {
    use super::prelude::*;

    #[test]
    fn test_basic_workflow() {
        // Create matrices
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);

        let b: Mat<f64> = Mat::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);

        let mut c: Mat<f64> = Mat::zeros(2, 2);

        // GEMM: C = A * B
        gemm(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());

        // Expected:
        // C[0,0] = 1*5 + 2*7 = 19
        // C[0,1] = 1*6 + 2*8 = 22
        // C[1,0] = 3*5 + 4*7 = 43
        // C[1,1] = 3*6 + 4*8 = 50

        assert!((c[(0, 0)] - 19.0).abs() < 1e-10);
        assert!((c[(0, 1)] - 22.0).abs() < 1e-10);
        assert!((c[(1, 0)] - 43.0).abs() < 1e-10);
        assert!((c[(1, 1)] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_simd_detection() {
        let level = detect_simd_level();
        println!("Detected SIMD level: {:?}", level);

        // Should detect at least scalar
        assert!(level >= SimdLevel::Scalar);
    }

    #[test]
    fn test_aligned_allocation() {
        let vec: AlignedVec<f64> = AlignedVec::zeros(100);
        let ptr = vec.as_ptr();

        // Should be aligned to at least 64 bytes
        assert_eq!(ptr as usize % 64, 0);
    }

    #[test]
    fn test_scalar_batch_operations() {
        use super::ScalarBatch;

        // Test ScalarBatch dot product
        let x = [1.0f64, 2.0, 3.0, 4.0];
        let y = [5.0f64, 6.0, 7.0, 8.0];
        let dot = f64::dot_batch(&x, &y);
        assert!((dot - 70.0).abs() < 1e-10); // 1*5 + 2*6 + 3*7 + 4*8 = 70
    }

    #[test]
    fn test_complex_ergonomics() {
        use super::{ComplexExt, I64, c64};

        // Test complex constructor
        let z = c64(3.0, 4.0);
        assert_eq!(z.re, 3.0);
        assert_eq!(z.im, 4.0);

        // Test imaginary unit
        let i = I64;
        assert_eq!(i.re, 0.0);
        assert_eq!(i.im, 1.0);

        // Test ComplexExt normalize
        let normalized = z.normalize();
        let mag = (normalized.re * normalized.re + normalized.im * normalized.im).sqrt();
        assert!((mag - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compensated_sum() {
        use super::{KahanSum, pairwise_sum};

        // Kahan sum for better accuracy
        let mut kahan = KahanSum::<f64>::new();
        for _ in 0..1000 {
            kahan.add(0.1);
        }
        assert!((kahan.sum() - 100.0).abs() < 1e-10);

        // Pairwise sum
        let values: Vec<f64> = (0..1000).map(|_| 0.1).collect();
        let result = pairwise_sum(&values);
        assert!((result - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_lazy_evaluation() {
        use super::{Expr, LazyExt};

        // Create matrices
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);

        // Build a lazy expression tree (no computation until eval())
        let expr = a.as_ref().lazy() + b.as_ref().lazy();

        // Evaluate the expression
        let result = expr.eval();

        assert!((result[(0, 0)] - 6.0).abs() < 1e-10); // 1 + 5
        assert!((result[(0, 1)] - 8.0).abs() < 1e-10); // 2 + 6
        assert!((result[(1, 0)] - 10.0).abs() < 1e-10); // 3 + 7
        assert!((result[(1, 1)] - 12.0).abs() < 1e-10); // 4 + 8

        // Test chained operations: (A + B) scaled by 2
        let scaled_expr = (a.as_ref().lazy() + b.as_ref().lazy()).scale(2.0);
        let result2 = scaled_expr.eval();

        assert!((result2[(0, 0)] - 12.0).abs() < 1e-10); // (1 + 5) * 2
        assert!((result2[(1, 1)] - 24.0).abs() < 1e-10); // (4 + 8) * 2
    }

    #[test]
    fn test_lazy_fma_gemm() {
        use super::{Expr, LazyExt, lazy_fma, lazy_gemm};

        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);

        // Test lazy_fma: 2*A + 3*B
        let fma_expr = lazy_fma(2.0, a.as_ref().lazy(), 3.0, b.as_ref().lazy());
        let result = fma_expr.eval();

        assert!((result[(0, 0)] - 17.0).abs() < 1e-10); // 2*1 + 3*5
        assert!((result[(1, 1)] - 32.0).abs() < 1e-10); // 2*4 + 3*8

        // Test lazy_gemm: 1*A*I + 0*A = A (using identity matrix)
        let identity: Mat<f64> = Mat::eye(2);
        let gemm_expr = lazy_gemm(
            1.0,
            a.as_ref().lazy(),
            identity.as_ref().lazy(),
            0.0,
            a.as_ref().lazy(),
        );
        let result2 = gemm_expr.eval();

        assert!((result2[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((result2[(1, 1)] - 4.0).abs() < 1e-10);
    }
}
