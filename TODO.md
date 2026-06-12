# OxiBLAS TODO

## Stubs to implement (added 2026-06-12 by /cooljapan-stub-check)

- [ ] `oxiblas-blas`: `crates/oxiblas-blas/src/level3/gemm_packing.rs:284` — add AVX-512 streaming stores for GEMM packing once `std::arch::x86_64::_mm512_stream_ps` is stabilized
  - Priority: P2 | Scope: small | Hint: none

Production-grade pure Rust BLAS/LAPACK implementation.

## Project Status (v0.2.1 Release - Updated 2026-03-16)

- **Tests:** 2,922 tests passing (100% success rate) + 287 doctests
- **Code:** ~223,935 lines of Rust across 371 files
- **Documentation:** ~16,163 lines of comments, 12 comprehensive examples
- **Benchmarks:** 14 criterion suites (+ size_variations, precision_bench), 121+ benchmarks
- **Coverage:** Full BLAS/LAPACK feature parity + modern extensions + sparse operations
- **no_std:** oxiblas-core and oxiblas-matrix support `#![no_std]` with alloc
- **Performance:**
  - **macOS (Apple M3):** DGEMM 25.6 GFLOPS (matches OpenBLAS 25.4!), rectangular 2.6-3.6× faster
  - **Linux x86_64 (Intel Xeon E5-2623 v4):** DGEMM 220 GFLOPS (256×256), **102% of OpenBLAS on 1024×1024**, **112% on f32 small matrices**
  - **Cholesky n=500: 9.75× speedup** (1.65 → 16.06 Gelem/s) with blocked algorithm
  - **5 LAPACK operations optimized:** Cholesky (9.75×), LU (~7×), QR (✅ complete), Bidiag (~3×), Hessenberg (~3×)
  - **OpenBLAS parity achieved:** 80-112% performance (sometimes faster on Linux!)
  - **All operations have `compute_auto()` for automatic optimization**
- **Zero warnings:** ✅ clippy clean + rustdoc clean
- **Zero unwrap():** ✅ All production code free of unwrap() calls
- **Refactoring:** 16 files (51,890 lines) → 113 modules, all <2000 lines
- **Files >2000:** 0 (100% compliant)
- **v0.2.0 New:**
  - Fixed blocked QR factorization (WY representation) - T→T^T bug fixed, 3-7× speedup
  - Recursive cache-oblivious factorizations: Cholesky, LU, QR (`compute_recursive()`)
  - Parallel blocked factorizations: Cholesky, LU (`compute_blocked_par()`)
  - Complex bidiagonal reduction (`ComplexBidiagFactors` for Complex64/Complex32)
  - Runtime auto-tuning infrastructure (`RuntimeAutoTuner`, `gemm_auto_tuned()`)
  - Multifrontal sparse factorizations (`MultifrontalCholesky`, `MultifrontalLU`)
  - ndarray parallel GEMM (`matmul_par`) and sparse integration (`array2_to_csr`, `spmv_ndarray`, `sparse_solve_ndarray`)
  - Batched BLAS operations (`gemm_batched`, `gemm_strided_batched`, `axpy_batched`, `gemv_batched` + parallel variants)
  - Runtime SIMD dispatch infrastructure (`SimdCapabilities`, `SimdDispatcher`, `KernelSelector`, `simd_dispatch!`)
  - Feature-gated imports + `features` module in main crate + library comparison docs
  - Benchmark size variations (tiny/nonpow2/rectangular/large) + FLOPS reporting + precision benchmarks
  - Performance comparison tables and Algorithm Selection Guide in README
  - Mixed-precision iterative refinement: LU, Cholesky, symmetric, QR (`mixed_precision_solve_qr`)
  - Advanced sparse LU pivoting: threshold, static (SuperLU-style), Bunch-Kaufman LDL^T (`SparseLdlt`)
  - Standard test matrix generators (`laplacian_2d`, `laplacian_3d`, `random_spd`, etc.)
  - LAPACK integration test suite (61 tests: LU, Cholesky, QR, SVD, EVD, Solve)
  - Memory usage tests for sparse operations (27 tests)
  - Refactored scalar.rs (2846 lines → 8 modules under scalar/)
  - Retired oxiblas-ffi (Pure Rust ecosystem)
  - no_std support for oxiblas-core and oxiblas-matrix
  - SSE4.2 intermediate GEMM micro-kernels (F64x2Sse 4×2, F32x4Sse 4×4)
  - NUMA-aware allocation (`NumaVec<T>`, `MatNuma<T>`, Linux real NUMA)
  - Thread pool customization (`set_global_thread_pool`, `OxiblasThreadConfig`)
  - Performance regression framework (`PerfBaseline`, `RegressionChecker`, JSON) + `regress` CLI binary
  - Multilevel graph partitioning (METIS-equivalent pure Rust, HEM + KL refinement)
  - Out-of-core sparse factorization (`OutOfCoreLu`, `OutOfCoreCholesky` with block I/O)
  - Thick-Restart Lanczos (TRL) - Wu-Simon algorithm for sparse eigenvalue
  - LOBPCG - Knyazev 2001, block preconditioned CG eigensolver
  - Sparse QR - COLAMD + Givens rotations, `SparseQr` with `solve_least_squares`
  - Randomized EVD - Halko-Martinsson-Tropp for dense symmetric matrices, f64/f32
  - Stochastic trace/diagonal - Hutchinson, Hutch++, XTrace, Bekas diagonal, log-det

### LAPACK Performance Optimization (Session 19 - Continued)

#### Problem Analysis

**BLAS Level 3 Microkernels: Production Grade** ✅
- Hand-optimized SIMD intrinsics (AVX2/AVX-512/NEON)
- FMA instructions, optimal register blocking (8×6 YMM tiles)
- 4-way loop unrolling, software prefetching
- **Assessment: OpenBLAS/BLIS quality implementation**

**Critical Bottleneck Identified:**
- LAPACK `compute()` methods used unblocked Level 2 BLAS
- Performance: ~2-3 Gelem/s despite having 51 GFLOPS GEMM kernels
- Blocked algorithms existed but weren't auto-selected
- **Impact: Users achieved only 10-15% of potential performance**

#### Solution Implemented

Added `compute_auto()` methods for f64/f32 with automatic algorithm selection:
- **Blocked algorithm** (n ≥ 128): Level 3 BLAS (GEMM/TRSM) for cache efficiency
- **Unblocked algorithm** (n < 128): Level 2 BLAS with lower overhead

#### Performance Results (Measured)

| Operation | Size | Before (µs) | After (µs) | **Speedup** | Throughput Improvement |
|-----------|------|-------------|------------|-------------|------------------------|
| Cholesky  | n=500 | 25,250 (1.65 Gelem/s) | 2,671 (15.60 Gelem/s) | **9.45×** | +845% throughput |
| Cholesky  | n=200 | 1,148 (2.32 Gelem/s) | 254.6 (10.47 Gelem/s) | **4.51×** | +351% throughput |
| Cholesky  | n=100 | 100.1 (3.33 Gelem/s) | 50.59 (6.59 Gelem/s) | **1.98×** | +97.9% throughput |
| LU        | n≥128 | ~3 Gelem/s | ~15 Gelem/s (est.) | **~5×** | Expected similar gains |

**Why Blocking Works:**
- Cache hierarchy exploitation: 4000 flops/byte vs 0.5 flops/byte
- Block size 64×64 = 32KB fits L1 cache perfectly
- Level 3 BLAS operations dominate (O(n³) work in GEMM)

#### API Changes

**Non-breaking - Existing methods preserved:**
- `Cholesky::compute()`, `Lu::compute()` - Unchanged behavior
- `Cholesky::compute_blocked()`, `Lu::compute_blocked()` - Explicit control

**New convenience methods:**
```rust
Cholesky::compute_auto(a) -> Result<Self, CholeskyError>  // f64/f32
Lu::compute_auto(a) -> Result<Self, LuError>              // f64/f32
Qr::compute_auto(a) -> Result<Self, QrError>              // Fixed in v0.2.0
Qr::compute_blocked(a, nb) -> Result<Self, QrError>       // Fixed in v0.2.0
Qr::compute_recursive(a) -> Result<Self, QrError>         // New in v0.2.0
Cholesky::compute_recursive(a) -> Result<Self, ...>       // New in v0.2.0
Lu::compute_recursive(a) -> Result<Self, ...>             // New in v0.2.0
Cholesky::compute_blocked_par(a) -> Result<Self, ...>     // New in v0.2.0 (parallel feature)
Lu::compute_blocked_par(a) -> Result<Self, ...>           // New in v0.2.0 (parallel feature)
```

#### Files Modified

**Cholesky optimization:**
- `crates/oxiblas-lapack/src/cholesky/llt.rs` (lines 500-541)
  - Added `compute_auto()` with auto-selection logic
  - Threshold: n ≥ 128 → blocked, n < 128 → unblocked

**LU optimization:**
- `crates/oxiblas-lapack/src/lu/partial_piv.rs` (lines 766-806)
  - Added `compute_auto()` following same pattern
  - Maintains partial pivoting for numerical stability

**QR optimization (COMPLETE ✅):**
- `crates/oxiblas-lapack/src/qr/householder.rs` (lines 252-333)
  - Implemented blocked QR with automatic algorithm selection
  - Added `compute_auto()` and `compute_blocked()` methods
  - Uses standard unblocked algorithm within blocks for correctness
  - All 4 new tests passing (2832 total tests now passing)
  - Performance: Expected 2-3× speedup for large matrices (benchmarking...)

**SVD bidiagonalization optimization:**
- `crates/oxiblas-lapack/src/svd/bidiag_reduce.rs` (lines 136-171)
  - Added `compute_auto()` with auto-selection logic
  - Threshold: min(m,n) ≥ 64 → blocked, otherwise → unblocked
  - Expected 2-4× speedup for large matrices

**Eigenvalue Hessenberg optimization:**
- `crates/oxiblas-lapack/src/evd/hessenberg.rs` (lines 229-262)
  - Added `compute_auto()` with auto-selection logic
  - Threshold: n ≥ 96 → blocked, n < 96 → unblocked
  - Expected 2-4× speedup for large matrices

#### Technical Achievement

**Algorithm complexity analysis:**
- Unblocked: O(n³/3) flops, O(n³/B) cache misses
- Blocked: O(n³/3) flops, O(n³/(B√M)) cache misses
- Speedup factor: ~√M/B ≈ 3-10× (verified experimentally)

**Quality metrics:**
- ✅ 2,582 tests + 271 doc tests passing (100% success rate)
- ✅ Zero clippy warnings across workspace
- ✅ Zero unwrap() in production code
- ✅ All blocked algorithms verified via reconstruction tests
- ✅ Numerical stability maintained (partial pivoting in LU)

#### Recommendations for Users

**For best performance in v0.1.0, migrate to `compute_auto()` methods:**
```rust
// 4 operations optimized and ready:
let chol = Cholesky::compute_auto(a)?;         // 9.75× faster (measured!)
let lu = Lu::compute_auto(a)?;                 // ~7× faster (expected)
let bidiag = BidiagFactors::compute_auto(a)?;  // ~3× faster (expected)
let hess = Hessenberg::compute_auto(a)?;       // ~3× faster (expected)
```

**Benefits:**
- Transparent optimization (no code changes needed beyond method name)
- Optimal performance for all matrix sizes (no performance cliffs)
- Cache-aware algorithm selection based on matrix size

---

## TIER 1 - CRITICAL (Blocks Production Use)

### Complex FFI Bindings (RETIRED v0.2.0 - Pure Rust ecosystem)
- [x] ~~CGEMV, ZGEMV (complex GEMV)~~
- [x] ~~CTRSM, ZTRSM (complex triangular solve)~~
- [x] ~~CGETRF, ZGETRF (complex LU)~~
- [x] ~~CPOTRF, ZPOTRF (complex Cholesky)~~
- [x] ~~CGEQRF, ZGEQRF (complex QR)~~
- [x] ~~CGESVD, ZGESVD (complex SVD)~~
- [x] ~~CHEEV, ZHEEV (Hermitian eigenvalues)~~
- [x] ~~CHEEVD, ZHEEVD (Hermitian eigenvalues D&C)~~
- [x] ~~CGEEV, ZGEEV (Complex general eigenvalues)~~
- **Note:** oxiblas-ffi has been retired. The COOLJAPAN ecosystem is Pure Rust.

### BLAS Level 2 (Complete)
- [x] `symv` - Symmetric matrix-vector multiply
- [x] `hemv` - Hermitian matrix-vector multiply
- [x] `syr2` - Symmetric rank-2 update
- [x] `her2` - Hermitian rank-2 update

### Sparse Eigenvalue Solvers (Complete)
- [x] Lanczos iteration for symmetric matrices
- [x] Arnoldi iteration for general matrices
- [x] Shift-and-invert spectral transformation
- [x] Implicit restart (IRAM)

### Iterative Refinement (Complete)
- [x] Post-factorization refinement for LU (sgerfs, dgerfs)
- [x] Post-factorization refinement for Cholesky (sporfs, dporfs)
- [x] Symmetric system refinement (ssyrfs, dsyrfs)
- [x] Mixed precision refinement (f32 factor, f64 residual)

### Band Matrix Support (Complete)
- [x] `gbmv` - General banded matrix-vector
- [x] `gbtrf` - General banded LU factorization
- [x] `gbtrs` - General banded triangular solve
- [x] `gbsv` - General banded system solve

---

## TIER 2 - HIGH (Complete BLAS/LAPACK Coverage)

### BLAS Level 2 Packed/Banded (Complete)
- [x] `sbmv` - Symmetric banded matrix-vector
- [x] `hbmv` - Hermitian banded matrix-vector
- [x] `spmv` - Symmetric packed matrix-vector
- [x] `hpmv` - Hermitian packed matrix-vector
- [x] `tbmv` - Triangular banded matrix-vector
- [x] `tpmv` - Triangular packed matrix-vector
- [x] `tbsv` - Triangular banded solve
- [x] `tpsv` - Triangular packed solve

### LAPACK Complex Variants (FFI) (RETIRED v0.2.0 - Pure Rust ecosystem)
- [x] ~~Complete complex GETRF/GETRS~~
- [x] ~~Complex GEQRF/UNGQR~~
- [x] ~~Complex GESVD~~
- [x] ~~Complex HEEV/HEEVD~~
- [x] ~~Complex GEEV~~
- **Note:** oxiblas-ffi has been retired. The COOLJAPAN ecosystem is Pure Rust.

### Orthogonal Transformation Functions (Complete)
- [x] `orgqr` / `ungqr` - Generate Q from QR
- [x] `ormqr` / `unmqr` - Multiply by Q
- [x] `ormbr` / `unmbr` - Multiply by bidiagonal transforms
- [x] `orgbr` / `ungbr` - Generate bidiagonal transforms

### Tridiagonal Solvers (Complete)
- [x] `gtsv` - General tridiagonal solve
- [x] `gttrf` - General tridiagonal factorization
- [x] `gttrs` - General tridiagonal solve from factorization
- [x] `ptsv` - Positive definite tridiagonal solve
- [x] `pttrf` - Positive definite tridiagonal factorization
- [x] `pttrs` - Positive definite tridiagonal solve from factorization

### Generalized Eigenvalue (Complete)
- [x] Full `ggev` support (general eigenvalue)
- [x] `sygv` / `hegv` (symmetric/Hermitian generalized)
- [x] `gges` (generalized Schur)

### Balancing Algorithms (Complete)
- [x] `gebal` - Balance a general matrix
- [x] `gebak` - Back-transform eigenvectors

---

## TIER 3 - MEDIUM (Production Library Features)

### Extended Features
- [x] f16 (half precision) support via `half` crate
- [x] Extended precision dot products (sdsdot, dsdot, dot_kahan, dot_pairwise)
- [x] Auto-tuning utilities for block sizes (TuningConfig, AutoTuner)
- [x] Tensor contraction operations (einsum, batched matmul, outer product)
- [x] f128 / quad precision support via `twofloat` crate (QuadFloat newtype)
- [x] Mixed precision algorithms - `mixed_precision_solve`, `mixed_precision_solve_cholesky`, `mixed_precision_solve_symmetric` (f32 factorization + f64 refinement)

### Advanced Sparse Preconditioners
- [x] ILUT (incomplete LU with threshold)
- [x] ILUTP (ILUT with pivoting)
- [x] IC (incomplete Cholesky) - IC0, ICT implemented
- [x] Jacobi / Block Jacobi
- [x] Gauss-Seidel / SOR / SSOR
- [x] AMG (algebraic multigrid) - classical Ruge-Stüben with V/W-cycle
- [x] SPAI (sparse approximate inverse) - least-squares column computation
- [x] AINV (approximate inverse) - factored sparse approximate inverse
- [x] Additive Schwarz (domain decomposition) - overlapping subdomains with ILU/Jacobi
- [x] Polynomial preconditioners (Neumann series, Chebyshev)

### Block Iterative Solvers
- [x] GMRES with restart (includes preconditioned variant)
- [x] Block-CG (with Block-PCG preconditioned variant)
- [x] MINRES (minimum residual) - includes pminres (preconditioned)
- [x] QMR (quasi-minimal residual)
- [x] TFQMR (transpose-free quasi-minimal residual)

### Reordering Algorithms
- [x] RCM (reverse Cuthill-McKee)
- [x] AMD (approximate minimum degree)
- [x] Nested dissection (level-set based)
- [x] METIS-equivalent pure Rust multilevel nested dissection - `MultilevelPartitioner`, HEM coarsening, KL refinement (v0.2.0)

### Extended Precision
- [x] f16 (half precision) support
- [x] Extended precision dot products (sdsdot, dsdot)
- [x] f128 / quad precision support (QuadFloat via twofloat)
- [x] Mixed precision algorithms (f32 factorization + f64 refinement)

### Sparse Advanced
- [x] Sparse SVD (truncated) - Lanczos-based truncated SVD, Randomized SVD, Incremental SVD complete
- [x] Sparse eigenvalue (beyond Lanczos) - Shift-invert, IRAM, Block Lanczos, Block Arnoldi, Interval eigenvalue, Polynomial filtering complete
- [x] Sparse-sparse multiply (A*B both sparse) - `spmm_sparse()`
- [x] Additional sparse formats (ELL, DIA, BSR) - All three formats implemented with full conversion support

### Infrastructure
- [x] Auto-tuning for block sizes - TuningConfig with architecture-specific heuristics
- [x] Workspace size query functions (lwork) - Full LAPACK-style workspace queries
- [x] Detailed info structures for factorizations - LU/Cholesky/QR/SVD/EVD info
- [x] Error code standardization - LAPACK INFO codes, unified error types

### Performance Benchmarks
- [x] Dedicated benchmarks subcrate (oxiblas-benchmarks)
- [x] BLAS Level 1 benchmarks (dot, axpy, scal, nrm2, asum, iamax)
- [x] BLAS Level 2 benchmarks (gemv, ger)
- [x] BLAS Level 3 benchmarks (gemm, gemm3m, trmm) - includes rectangular and complex variants
- [x] LAPACK QR benchmarks (Qr, QrPivot, Lq, Rq, CompleteOrthogonalDecomp)
- [x] LAPACK SVD benchmarks (Svd, SvdDc) - includes algorithm comparison
- [x] New features benchmarks (extended precision, einsum, tensor operations, outer product)
- [x] Criterion-based with HTML reports and statistical analysis
- [x] Comparison against OpenBLAS (optional feature: compare-openblas)
  - [x] GEMM (square and rectangular matrices)
  - [x] GEMV (matrix-vector multiply)
  - [x] DOT, AXPY, NRM2 (vector operations)
  - [x] Comprehensive documentation and usage guide
- [x] Continuous benchmark regression tracking - `PerfBaseline`, `RegressionChecker`, JSON storage (v0.2.0)
- [ ] Cross-platform performance comparison (x86-64, ARM, Apple Silicon)
- [ ] Comparison against MKL, BLIS, Accelerate (future)

---

## TIER 4 - OPTIONAL (Differentiators)

### Micro-Optimizations for Apple Silicon ✅ Completed (2025-12-26)
- [x] 128-byte cache line alignment (vs 64-byte on x86_64)
- [x] Optimized prefetch distances (10 iterations for micro-kernel, 6 cache lines for packing)
- [x] Tuned KC parameter (448 for f64, 896 for f32 - 17% larger for 4MB L2)
- **Expected gain**: 4-9% toward 90% OpenBLAS target
- **Commit**: a27d8eb "perf: TIER 4 optimizations for Apple Silicon"

### Advanced SIMD
- [x] Runtime SIMD dispatch (already implemented via is_x86_feature_detected!)
- [x] AVX-512 support (f64 16×6, f32 16×16 kernels implemented)
- [ ] SVE support (ARM Scalable Vector Extension - requires nightly)
- [x] SSE4.2 intermediate kernels - `gemm_kernel_sse42.rs` F64x2Sse/F32x4Sse 4×4 micro-kernels (v0.2.0)

### Specialized Algorithms
- [x] Divide-and-conquer SVD variants - SvdDc + SelectiveSvd (GESVDX-style)
- [x] QR iteration variants - Bisection + inverse iteration for tridiagonal EVD (TridiagEvd)
- [x] Randomized algorithms (rSVD) - RandomizedSvd with power iteration, low-rank approximation

### Custom Threading
- [x] NUMA-aware allocation (full infrastructure in memory/numa.rs - Linux only)
- [x] Thread pool customization - `set_global_thread_pool`, `OxiblasThreadConfig`, `with_thread_count` (v0.2.0)
- [ ] Work-stealing scheduler tuning (Rayon handles this)

### Tensor Operations
- [x] BLAS-like tensor contractions (Tensor3, contract_2d, contract_3d_2d)
- [x] Einsum-style operations (einsum with 24 patterns including advanced contractions)
- [x] Batched matrix multiplication
- [x] Outer product operations
- [x] More einsum patterns (advanced contractions: trace, dot product, 3D transposes, tensor-matrix contraction, axis sums)
- [x] Tensor transpose and permutation (3 transpose variants: ijk->ikj, ijk->jik, ijk->kji)
- [x] N-dimensional tensor support (NdTensor with dynamic shape, reshape, transpose, permute, matmul, contract, outer, diagonal, trace)

---

## Performance Targets

**Platform:** Apple M3 (ARM64 NEON), tested 2025-12-23

### Core BLAS/LAPACK Operations

| Operation | Current (vs OpenBLAS) | Target | Status | Notes |
|-----------|----------------------|--------|--------|-------|
| DGEMM (64×64) | 59% (13.9 Gf/s) | 90% | 🟡 In Progress | 4×6 NEON kernel, +74% improvement |
| DGEMM (256×256) | 72% (19.3 Gf/s) | 90% | 🟡 In Progress | 4×6 NEON kernel, +111% improvement |
| DGEMM (512×512) | 76% (20.6 Gf/s) | 90% | 🟡 In Progress | 4×6 NEON kernel, +159% improvement |
| DGEMM (1024×1024) | 79% (20.9 Gf/s) | 90% | 🟢 Nearly There | 4×6 NEON kernel, +175% improvement |
| DGEMV (500×500) | 3.9-5.3 Gelem/s | 85% | 🟢 Good | SIMD + cache blocking implemented |
| LU (1024×1024) | ~7.0 Gf/s | 85% | 🟡 In Progress | Blocked algorithm, 20× vs unblocked |
| Cholesky (1024×1024) | ~14.7 Gf/s | 85% | 🟡 In Progress | Blocked algorithm, 10× vs unblocked |
| QR (500×500) | 1.5 ms (167 Melem/s) | 80% | 🟢 Good | Householder with blocking |
| SVD D&C (200×200) | 33 ms (1.2 Melem/s) | 75% | 🟢 Good | 2-3× faster than standard |

### GEMM-Based Operations (Optimized via GEMM Kernel)

| Operation | Size | Naive → Optimized | Speedup | Status |
|-----------|------|-------------------|---------|--------|
| SYRK (f64) | 128×128 | 4.20 → 26.32 Gf/s | **6.27×** | ✅ Complete |
| SYRK (f64) | 256×256 | 3.90 → 37.16 Gf/s | **9.52×** | ✅ Complete |
| SYRK (f64) | 512×512 | 3.46 → 39.71 Gf/s | **11.49×** | ✅ Complete |
| SYRK (f64) | 1024×1024 | 3.21 → 40.24 Gf/s | **12.53×** | ✅ Complete |
| SYR2K (f64) | 128×128 | 4.29 → 25.75 Gf/s | **6.00×** | ✅ Complete |
| SYR2K (f64) | 256×256 | 3.80 → 37.31 Gf/s | **9.82×** | ✅ Complete |
| SYR2K (f64) | 512×512 | 3.28 → 39.08 Gf/s | **11.91×** | ✅ Complete |
| SYR2K (f64) | 1024×1024 | 2.78 → 40.99 Gf/s | **14.76×** | ✅ Complete |
| SYMM (f64) | 512×512 | TBD | 1.1-1.8× | ✅ Complete |
| HEMM (c64) | - | - | Disabled | 3M overhead too high |
| HERK (f64) | 1024×1024 | - | 6-12× (same as SYRK) | ✅ Complete |
| HER2K (f64) | 1024×1024 | - | 6-15× (same as SYR2K) | ✅ Complete |
| TRMM (f64) | 128×128 | 3.20 → 22.07 Gf/s | **6.89×** | ✅ Complete |
| TRMM (f64) | 256×256 | 4.60 → 37.35 Gf/s | **8.11×** | ✅ Complete |
| TRMM (f64) | 512×512 | 4.15 → 39.48 Gf/s | **9.51×** | ✅ Complete |
| TRMM (f64) | 1024×1024 | 3.76 → 40.60 Gf/s | **10.79×** | ✅ Complete |
| TRSM (f64) | 128×128 | 3.12 → 7.68 Gf/s | **2.46×** | ✅ Complete |
| TRSM (f64) | 256×256 | 2.72 → 12.35 Gf/s | **4.54×** | ✅ Complete |
| TRSM (f64) | 512×512 | 2.28 → 16.47 Gf/s | **7.21×** | ✅ Complete |
| TRSM (f64) | 1024×1024 | 1.93 → 19.96 Gf/s | **10.32×** | ✅ Complete |
| Complex GEMM 3M | 1024×1024 | - | 40.72 Gf/s | ✅ Complete |
| Parallel GEMM | 1024×1024 | - | 130.81 Gf/s | ✅ Complete |

### LAPACK Blocked Factorizations (Optimized via GEMM/TRSM)

| Operation | Size | Unblocked → Blocked | Speedup | Status |
|-----------|------|---------------------|---------|--------|
| LU (f64) | 256×256 | 0.62 → 9.05 Gf/s | **14.50×** | ✅ Complete |
| LU (f64) | 512×512 | 0.45 → 6.65 Gf/s | **14.86×** | ✅ Complete |
| LU (f64) | 768×768 | 0.60 → 13.84 Gf/s | **23.01×** | ✅ Complete |
| LU (f64) | 1024×1024 | 0.35 → 6.99 Gf/s | **19.74×** | ✅ Complete |
| Cholesky (f64) | 256×256 | 2.16 → 13.01 Gf/s | **6.03×** | ✅ Complete |
| Cholesky (f64) | 512×512 | 1.65 → 14.97 Gf/s | **9.06×** | ✅ Complete |
| Cholesky (f64) | 768×768 | 1.84 → 14.76 Gf/s | **8.03×** | ✅ Complete |
| Cholesky (f64) | 1024×1024 | 1.44 → 14.73 Gf/s | **10.20×** | ✅ Complete |

**Recent Achievement (Session 17):** Blocked LU (**14-23× speedup**) and Blocked Cholesky (**6-10× speedup**) via GEMM/TRSM.

**Optimization Progress:**
1. ✅ GEMM: 79% of OpenBLAS (4×6 NEON + cache blocking + prefetching)
2. ✅ Parallel GEMM: 2D decomposition, 130.81 Gf/s (f64), 324.31 Gf/s (f32)
3. ✅ Complex GEMM: 3M method, 40.72 Gf/s (c64), 88.95 Gf/s (c32)
4. ✅ SYMM: GEMM-based, 1.1-1.8× speedup
5. ✅ SYRK/SYR2K: GEMM-based, 6-15× speedup
6. ✅ HERK/HER2K: GEMM-based, 6-15× speedup (same as SYRK/SYR2K for real types)
7. ✅ TRMM: GEMM-based, 7-11× speedup (expand triangular to full matrix)
8. ✅ TRSM: Blocked algorithm, 2.5-10× speedup (GEMM for off-diagonal updates)
9. ✅ LU: Blocked algorithm, 14-23× speedup (panel factorization + GEMM updates)
10. ✅ Cholesky: Blocked algorithm, 6-10× speedup (GEMM for symmetric rank-k updates)

**Next optimizations for 90% GEMM target:**
- [x] SIMD-optimized packing (+5-8%) - Implemented with AVX2 (x86_64) and NEON (aarch64) support
- [x] Arena-based allocation for temporary matrices - Bump allocator reduces malloc overhead
- [x] Cache block tuning (+3-5%) - Improved auto-tuning with L2-aware KC, cache-line alignment, aspect-ratio adaptation
- [x] Kernel optimization - Current kernels: f64 8×6 (24 NEON/12 AVX2 registers), f32 8×8 (16 NEON/8 AVX2 registers) with 2-4 way loop unrolling and software prefetching

**Current kernel implementations:**
- f64 NEON: 8×6 with 24 accumulator registers, 2-way unrolling, aggressive prefetching
- f64 AVX2: 8×6 with 12 ymm registers, 2-way unrolling, software prefetching
- f64 AVX-512: 16×6 with 12 zmm registers
- f32 NEON: 8×8 with 16 accumulator registers, 4-way unrolling
- f32 AVX2: 8×8 with 8 ymm registers, 4-way unrolling
- f32 AVX-512: 16×16 with 16 zmm registers

---

## Testing Requirements

- [x] All new functions must have unit tests - 2,811 tests + 195 doctests
- [x] Numerical accuracy tests against reference implementations - LAPACK compat suite (61 tests)
- [x] Performance regression tests - `quick_gemm_f64`, `quick_gemm_f32`, `PerfBaseline` JSON tracking (v0.2.0)
- [x] Edge case tests (empty, 1x1, non-square, etc.) - covered in LAPACK compat + unit tests
- [x] Complex number tests for all complex variants - complex bidiag, complex SVD, complex EVD

---

## Documentation Requirements

- [x] Rustdoc for all public APIs
  - [x] All BLAS Level 1/2/3 functions documented with examples
  - [x] All LAPACK decompositions (LU, QR, Cholesky, SVD, EVD) documented
  - [x] Core types (Mat, MatRef, MatMut) fully documented
  - [x] Scalar traits and extended precision documented
  - [x] Tensor operations documented with 24 einsum patterns
  - [x] Fix remaining rustdoc "unresolved link" warnings in oxiblas main crate (27→0 warnings)
  - [x] 258 passing doctests across workspace
- [x] Examples for common use cases
  - [x] basic_blas.rs - BLAS Level 1/2/3 operations
  - [x] lapack_decompositions.rs - LU, QR, Cholesky, SVD, EVD
  - [x] extended_precision.rs - f128, Kahan, pairwise, mixed precision
  - [x] tensor_operations.rs - einsum (24 patterns), batched matmul
  - [x] sparse_matrices.rs - CSR/CSC/COO, CG/GMRES, preconditioners
- [x] Performance guide - Added to lib.rs with algorithm selection, memory layout, parallelization tips
- [x] Migration guide from other libraries (from NumPy, ndarray-linalg, nalgebra) - Added to lib.rs
- [x] Architecture documentation (SIMD kernels, cache blocking) - Added to lib.rs with crate hierarchy, GEMM stack, SIMD abstraction

---

## Release Milestones

### v0.1.0 - Initial Release (Current) ✅
- All BLAS Level 1/2/3 operations
- ~~Complex number support in FFI~~ (RETIRED v0.2.0)
- Full LAPACK coverage (LU, Cholesky, QR, SVD, EVD, banded, packed)
- Band matrix and tridiagonal solvers
- Iterative refinement and expert drivers
- Orthogonal transformations
- Sparse eigenvalue solvers (Lanczos, Arnoldi, shift-invert)
- Advanced preconditioners (AMG, SPAI, AINV, Additive Schwarz)
- Reordering algorithms (AMD, RCM, MMD, COLAMD)
- Extended precision operations (f16, f128, dsdot, dot_kahan, dot_pairwise)
- Tensor operations (einsum with 24 patterns, batched matmul, outer product)
- GEMM 3M for complex matrices
- Complete orthogonal decomposition
- RQ/LQ/QL factorizations
- Comprehensive criterion benchmarks (12 suites, 121 benchmarks)
- 3,055 tests passing, zero clippy/rustdoc warnings

### v0.1.x - Performance Focus (Ongoing)
- [x] GEMM micro-kernel optimizations (4-way unrolling, interleaved loads/FMAs)
- [x] AVX-512 f64/f32 kernels with prefetching
- [x] AVX2 f64/f32 kernels with 4-way unrolling
- [x] NEON f64/f32 kernels optimized for Apple Silicon
- Current GEMM performance: ~85% OpenBLAS (DGEMM 51.4 Gf/s, SGEMM 109 Gf/s)
- Rectangular matrix speedup: 2.6-3.6× faster
- [x] Runtime SIMD dispatch (function multi-versioning) - Centralized detection & caching
- [x] Florida/SuiteSparse test matrices support - 6 standard patterns (Laplacian, tridiagonal, etc.)
- [x] Memory usage tests for sparse operations - 10 tests, no-leak verification
- [x] CI/CD integration for performance tracking - 3 workflows (CI, benchmarks, release)
- [x] Winograd algorithm for GEMM - Classic 2×2 algorithm with blocked variant
- [x] Cache-oblivious GEMM algorithm - Recursive divide-and-conquer with auto cache adaptation
- [x] Multifrontal methods for sparse factorization - SupernodalCholesky, SupernodalLU with BLAS-3
- [ ] SVE (ARM) support - for Graviton/A64FX (requires nightly)
- [x] NUMA-aware memory allocation - `NumaVec<T>`, `MatNuma<T>`, Linux NUMA topology (v0.2.0)
- [x] Complex bidiagonal reduction - `ComplexBidiagFactors` for Complex64/Complex32 (v0.2.0)
- [x] LAPACK test suite compatibility (61 integration tests in lapack_compat.rs) (v0.2.0)

### v1.0.0 - Production Ready
- Full BLAS/LAPACK coverage ✅
- Performance within 90% of MKL
- Comprehensive documentation ✅
- Stable API

### v0.2.0+ - Future Enhancements

**oxiblas-lapack (Performance Optimizations)**
- [x] Blocked QR factorization - Complete WY representation (Fixed in v0.2.0)
  - Code implemented in `qr/householder.rs` lines 252-494
  - T matrix construction and block reflector application debugged and fixed
  - Achieved 3-7× speedup for large matrices
- [x] Recursive factorizations (Cholesky, LU) - Cache-oblivious variants (Fixed in v0.2.0)
- [x] Recursive QR factorization - Cache-oblivious variant (Fixed in v0.2.0)
- [x] Parallel blocked factorizations for very large matrices (Fixed in v0.2.0)
- [x] Mixed-precision iterative refinement (f32 factor + f64 residual) - LU, Cholesky, symmetric, QR (v0.2.0)
- [x] Auto-tuning infrastructure (runtime block size optimization) - `RuntimeAutoTuner` (v0.2.0)

**oxiblas-core (SIMD Extensions)**
- [ ] RISC-V Vector (RVV) support (requires RISC-V hardware)
- [ ] PowerPC VSX support (requires PowerPC hardware)

**oxiblas-sparse (Large-Scale Extensions)**
- [x] Out-of-core factorization - `OutOfCoreLu`, `OutOfCoreCholesky` with block I/O + RAII temp files (v0.2.0)

---

## Code Quality Improvements

### Refactoring Status (Files >2000 lines)

**✅ Successfully Refactored (15 files → 105 modules)**:

| File | Original | Now | Status |
|------|----------|-----|--------|
| blas1.rs | 2,252 | 2 modules | ✅ Complete |
| blas2.rs | 5,481 | 5 modules | ✅ Complete |
| blas3.rs | 3,565 | 3 modules | ✅ Complete |
| cblas.rs | 2,300 | 3 modules | ✅ Complete |
| memory.rs | 2,417 | 6 modules | ✅ Complete |
| lapack/solve.rs | 3,639 | 3 modules | ✅ Complete |
| ops.rs | 2,186 | 3 modules | ✅ Complete |
| svd.rs | 2,186 | 6 modules | ✅ Complete |
| ordering.rs | 2,186 | 4 modules | ✅ Complete |
| graph.rs | 2,158 | 3 modules | ✅ Complete |
| matfun.rs | 2,113 | 4 modules | ✅ Complete |
| precond.rs | 5,904 | 10 modules | ✅ Complete |
| trsm.rs | 2,678 | 5 modules | ✅ Complete |
| iterative.rs | 5,571 | 13 modules | ✅ Complete |
| eigenvalue.rs | 9,880 | 11 modules | ✅ Complete |

**Summary**: 49,043 lines refactored into 105 modules across 3 sessions

**v0.2.0 Additional Refactoring:**

| File | Original | Now | Status |
|------|----------|-----|--------|
| scalar.rs | 2,846 | 8 modules (scalar/) | ✅ Complete |

**All files now under 2000 lines!** Largest file: simd/x86_64.rs at 1,989 lines.


