# Changelog

All notable changes to OxiBLAS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-12-27

### Initial Release

OxiBLAS 0.1.0 is the first public release of a pure Rust BLAS/LAPACK implementation.

#### Features

**Core Library (`oxiblas-core`)**
- SIMD layer with AVX2/FMA, AVX-512, and ARM NEON support
- Portable scalar fallback for all platforms
- Extended precision support: f16, f32, f64, f128 (quad precision)
- Complex number support: Complex32, Complex64
- Parallel execution with rayon (optional `parallel` feature)

**BLAS Operations (`oxiblas-blas`)**
- **Level 1** (11 operations): dot, axpy, nrm2, scal, iamax, swap, copy, rot, rotg, rotm, rotmg, asum
- **Level 2** (15 operations): gemv, ger, syr, spr, symv, sbmv, tbmv, tbsv, tpmv, tpsv, spmv, her, hpr, hbmv, hpmv
- **Level 3** (11 operations): gemm, syrk, trsm, syr2k, symm, hemm, herk, her2k
- Complete packed and banded matrix support
- Hermitian and symmetric operations for complex/real types

**LAPACK Operations (`oxiblas-lapack`)**
- **Factorizations**: LU (getrf), Cholesky (potrf), QR (geqrf, geqr2)
- **Decompositions**: SVD (gesvd), Eigenvalue (syev, geev), Schur (gees), Hessenberg (gehrd)
- **Linear Solvers**: Triangular (trtrs), General (gesv, getrs), Tridiagonal (gtsv, ptsv), Least squares (gels)
- **Advanced**: Condition number estimation, matrix inversion, balancing

**Sparse Linear Algebra (`oxiblas-sparse`)**
- **9 sparse formats**: CSR, CSC, COO, ELL, DIA, BSR, BSC, HYB, SELL-C-σ
- **Basic operations**: SpMV (generic + SIMD), SpMM, sparse addition, triangular solve
- **Iterative solvers** (10 variants):
  - CG, PCG (preconditioned conjugate gradient)
  - BiCGStab (stabilized bi-conjugate gradient)
  - GMRES, MINRES (minimal residual methods)
  - IDR(s), TFQMR, QMR (quasi-minimal residual variants)
  - Block-CG, Block-GMRES (multiple right-hand sides)
- **Eigenvalue solvers**: Lanczos, Arnoldi, IRAM (Implicitly Restarted Arnoldi)
- **SVD solvers**: Truncated SVD, Randomized SVD
- **Factorizations**: Sparse LU, Sparse QR, Sparse Cholesky
- **Preconditioners**: Jacobi, Gauss-Seidel, SOR, ILU0, ILUT, ILUTP, IC0, ICT, AMG, SPAI, AINV, Schwarz
- **Ordering algorithms**: RCM, AMD, Nested Dissection

**Matrix Utilities (`oxiblas-matrix`)**
- Flexible matrix types with multiple layouts (row-major, column-major)
- View and mutable view support
- Efficient memory management and slicing

**Extended Features**
- **Tensor operations**: Einstein summation (24 patterns), batched operations
- **Advanced summation**: Kahan (compensated), pairwise, superaccurate
- **C FFI**: Drop-in replacement for C BLAS/LAPACK (`oxiblas-ffi`)
- **ndarray integration**: Seamless interop with rust-ndarray (`oxiblas-ndarray`)

**Benchmarks (`oxiblas-benchmarks`)**
- **12 comprehensive benchmark suites** (13 files, 121 benchmark functions)
- Criterion.rs framework with HTML reports and statistical analysis
- OpenBLAS comparison suite (optional feature)
- Complete coverage: BLAS L1/L2/L3, LAPACK, sparse operations, iterative/eigenvalue/SVD solvers
- Performance: DGEMM achieves 79% of OpenBLAS speed on large matrices

#### Performance

**macOS (Apple M3, ARM64 NEON):**
- **GEMM**: 79% of OpenBLAS performance (1024×1024 matrices)
- **Peak**: 20.9 GFLOPS on large matrices

**Linux x86_64 (Intel Xeon E5-2623 v4, AVX2/FMA):**
- **DGEMM f64**: 80-112% of OpenBLAS performance across all sizes
  - 1024×1024: **102% of OpenBLAS** (213 vs 208 GFLOPS) - **Faster than OpenBLAS!**
  - 256×256: 95% of OpenBLAS (220 vs 232 GFLOPS) - Peak performance
  - 512×512: 80% of OpenBLAS (194 vs 244 GFLOPS) - Optimization target
- **SGEMM f32**: 94-112% of OpenBLAS performance
  - 64×64: **112% of OpenBLAS** (253 vs 225 GFLOPS) - **Faster than OpenBLAS!**
  - 128×128: 100% of OpenBLAS (328 vs 328 GFLOPS) - Identical performance
- **Cache-aware tuning**: 13-20% performance improvement with platform-specific optimization
  - Fine-tuned blocking parameters: KC=192, MC=128 (optimized for 256KB L2 cache)
  - Increased prefetch distance: 12 iterations (Intel Xeon E5-2600 latency)

**General:**
- **GEMM-based operations**: 6-15× speedup over naive implementations
- **Cache-aware**: BLIS-style blocking for optimal memory hierarchy usage
- **Platform detection**: Linux sysfs, macOS sysctl, x86_64 CPUID fallback
- **Parallel**: Efficient multi-threading with rayon
- **SIMD**: Hand-tuned kernels for AVX2/FMA and ARM NEON

#### Documentation

- Comprehensive API documentation with examples
- README with quick start guide
- TODO.md tracking development priorities
- 121 benchmark functions demonstrating all operations

#### Code Quality

- ~156,000 lines of Rust code
- Zero clippy warnings policy enforced
- Modular architecture with 9 specialized crates
- Complete test coverage for all operations
- Safe Rust with minimal unsafe blocks (only in SIMD kernels)

#### Known Limitations

- SVD performance not yet optimized (future work)
- Some LAPACK operations may be slower than commercial libraries
- No GPU acceleration (CPU-only)
- Requires Rust 1.85+ for edition 2024 features

#### Dependencies

**Required**:
- num-complex 0.4
- num-traits 0.2
- bytemuck 1.21

**Optional**:
- rayon 1.10 (parallel feature)
- ndarray 0.17 (ndarray integration)
- nalgebra 0.33 (interop)
- half 2.4 (f16 support)
- twofloat 0.8 (f128 support)

#### Crate Structure

- `oxiblas` - Main convenience crate, re-exports all modules
- `oxiblas-core` - Core types, traits, SIMD layer
- `oxiblas-matrix` - Matrix types and utilities
- `oxiblas-blas` - BLAS Level 1/2/3 implementations
- `oxiblas-lapack` - LAPACK factorizations and solvers
- `oxiblas-sparse` - Sparse matrix formats and algorithms
- `oxiblas-ffi` - C FFI bindings (CBLAS/LAPACKE compatible)
- `oxiblas-ndarray` - Integration with rust-ndarray
- `oxiblas-benchmarks` - Comprehensive performance benchmarks

---

## Release Checklist

- [x] All tests pass
- [x] Zero clippy warnings
- [x] Documentation complete
- [x] Benchmarks comprehensive
- [x] LICENSE files (MIT + Apache-2.0)
- [x] README up to date
- [x] CHANGELOG created
- [ ] Version 0.1.0 in all Cargo.toml
- [ ] Examples functional
- [ ] cargo publish --dry-run succeeds
- [ ] Git tags created

---

[Unreleased]: https://github.com/cool-japan/oxiblas/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cool-japan/oxiblas/releases/tag/v0.1.0
