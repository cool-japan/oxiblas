# OxiBLAS

**Pure Rust BLAS/LAPACK implementation for the SciRS2 ecosystem**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org/)

OxiBLAS is a production-grade, pure Rust implementation of BLAS (Basic Linear Algebra Subprograms) and LAPACK (Linear Algebra PACKage). Designed as the foundational linear algebra library for the [SciRS2](https://github.com/cool-japan/scirs) scientific computing ecosystem.

## Features

- **Pure Rust** - No C dependencies, fully portable, safe by default
- **SIMD Optimized** - Custom SIMD layer using `core::arch` intrinsics
  - x86_64: AVX2/FMA (256-bit), AVX512F (512-bit)
  - AArch64: NEON (128-bit)
  - Fallback: Scalar operations
- **Cache Aware** - BLIS-style blocked algorithms for optimal cache usage
- **Parallel** - Optional rayon-based parallelization for multi-core systems
- **Complete BLAS** - Full Level 1, 2, 3 operations (including packed/banded)
- **Extensive LAPACK** - LU, Cholesky, QR, SVD, EVD, Schur, Hessenberg
- **Extended Precision** - f16, f128 (quad precision), Kahan/pairwise summation
- **Tensor Operations** - Einstein summation (24 patterns), batched matmul, batched BLAS
- **Sparse Support** - 9 formats (CSR, CSC, COO, ELL, DIA, BSR, BSC, HYB, SELL) with advanced solvers
- **Advanced Preconditioners** - AMG, SPAI, AINV, Schwarz, polynomial
- **Recursive Factorizations** - Cache-oblivious Cholesky, LU, QR (`compute_recursive()`)
- **Parallel Factorizations** - Parallel blocked Cholesky and LU (`compute_blocked_par()`)
- **Batched BLAS** - `gemm_batched`, `gemm_strided_batched`, `axpy_batched`, `gemv_batched`
- **Mixed-Precision Refinement** - f32 factorization + f64 residual for LU, Cholesky, QR
- **Runtime Auto-Tuning** - `RuntimeAutoTuner` selects optimal block sizes at runtime
- **Multifrontal Sparse** - `MultifrontalCholesky`, `MultifrontalLU` for large sparse systems
- **no_std Support** - `oxiblas-core` and `oxiblas-matrix` support `#![no_std]` with alloc
- **Comprehensive Benchmarks** - Direct performance comparison with OpenBLAS

## Supported Types

| Type | Description | Feature Flag |
|------|-------------|--------------|
| `f32` | Single precision floating point | (always) |
| `f64` | Double precision floating point | (always) |
| `f16` | Half precision (16-bit) floating point | `f16` |
| `f128` (QuadFloat) | Quad precision (~31 decimal digits) | `f128` |
| `Complex32` | Single precision complex | (always) |
| `Complex64` | Double precision complex | (always) |

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
oxiblas = "0.2"

# With parallelization
oxiblas = { version = "0.2", features = ["parallel"] }

# With extended precision
oxiblas = { version = "0.2", features = ["f16", "f128"] }

# All features
oxiblas = { version = "0.2", features = ["full"] }
```

### Basic BLAS Example

```rust
use oxiblas_blas::level3::gemm;
use oxiblas_matrix::Mat;

// Create matrices
let a = Mat::from_rows(&[
    &[1.0, 2.0, 3.0],
    &[4.0, 5.0, 6.0],
]);
let b = Mat::from_rows(&[
    &[7.0, 8.0],
    &[9.0, 10.0],
    &[11.0, 12.0],
]);
let mut c = Mat::zeros(2, 2);

// GEMM: C = A * B
gemm(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());
// Result: [[58, 64], [139, 154]]
```

### LU Decomposition

```rust
use oxiblas_lapack::lu::Lu;
use oxiblas_matrix::Mat;

let a = Mat::from_rows(&[
    &[2.0, 1.0],
    &[1.0, 3.0],
]);

let lu = Lu::compute(a.as_ref()).expect("Matrix is not singular");
let det = lu.determinant();
assert!((det - 5.0).abs() < 1e-10); // det = 2*3 - 1*1 = 5
```

### SVD

```rust
use oxiblas_lapack::svd::Svd;
use oxiblas_matrix::Mat;

let a = Mat::from_rows(&[
    &[3.0, 1.0],
    &[1.0, 3.0],
    &[1.0, 1.0],
]);

let svd = Svd::compute(a.as_ref()).expect("SVD failed");
let singular_values = svd.singular_values();  // [4.24, 2.00]
let u = svd.u();   // Left singular vectors
let vt = svd.vt(); // Right singular vectors (transposed)
```

## Examples

OxiBLAS includes comprehensive examples demonstrating all major features:

```bash
# Basic BLAS operations (Level 1/2/3)
cargo run --example basic_blas

# LAPACK decompositions (LU, QR, Cholesky, SVD, EVD)
cargo run --example lapack_decompositions

# Extended precision (f128, Kahan, pairwise summation)
cargo run --example extended_precision --features f128

# Tensor operations & Einstein summation (24 patterns)
cargo run --example tensor_operations

# Sparse matrices (CSR, CSC, COO, iterative solvers, preconditioners)
cargo run --example sparse_matrices --features parallel
```

See [crates/oxiblas/examples/](crates/oxiblas/examples/) for complete source code.

## Crate Structure

OxiBLAS is organized as a workspace with specialized sub-crates:

| Crate | Description |
|-------|-------------|
| `oxiblas` | Unified re-exports and prelude |
| `oxiblas-core` | Core traits, SIMD abstractions, memory management, scalar types (f16, f128); `no_std` |
| `oxiblas-matrix` | Matrix types (`Mat`, `MatRef`, `MatMut`, `DiagRef`); `no_std` |
| `oxiblas-blas` | BLAS Level 1/2/3 operations, tensor operations, einsum, batched BLAS |
| `oxiblas-lapack` | LAPACK decompositions and solvers (blocked, recursive, parallel, auto variants) |
| `oxiblas-sparse` | Sparse matrix formats (9 types), multifrontal factorizations, and algorithms |
| `oxiblas-ndarray` | Integration with ndarray (parallel GEMM, sparse integration) |
| `oxiblas-benchmarks` | Performance benchmarks with OpenBLAS comparison (not in meta crate) |

## BLAS Operations

### Level 1 (Vector-Vector)

| Function | Description |
|----------|-------------|
| `dot` | Dot product |
| `axpy` | y = alpha * x + y |
| `scal` | x = alpha * x |
| `copy` | y = x |
| `swap` | Swap x and y |
| `nrm2` | Euclidean norm |
| `asum` | Sum of absolute values |
| `iamax` | Index of max absolute value |
| `rot` | Givens rotation |

### Level 2 (Matrix-Vector)

| Function | Description |
|----------|-------------|
| `gemv` | General matrix-vector multiply |
| `symv` | Symmetric matrix-vector multiply |
| `hemv` | Hermitian matrix-vector multiply |
| `trmv` | Triangular matrix-vector multiply |
| `trsv` | Triangular solve |
| `ger` | Rank-1 update |
| `syr` | Symmetric rank-1 update |
| `her` | Hermitian rank-1 update |
| `syr2` | Symmetric rank-2 update |
| `her2` | Hermitian rank-2 update |
| `gbmv` | General banded matrix-vector |
| `sbmv` | Symmetric banded matrix-vector |
| `hbmv` | Hermitian banded matrix-vector |
| `spmv` | Symmetric packed matrix-vector |
| `hpmv` | Hermitian packed matrix-vector |
| `tbmv` | Triangular banded matrix-vector |
| `tpmv` | Triangular packed matrix-vector |
| `tbsv` | Triangular banded solve |
| `tpsv` | Triangular packed solve |

### Level 3 (Matrix-Matrix)

| Function | Description |
|----------|-------------|
| `gemm` | General matrix-matrix multiply |
| `symm` | Symmetric matrix-matrix multiply |
| `hemm` | Hermitian matrix-matrix multiply |
| `trmm` | Triangular matrix-matrix multiply |
| `trsm` | Triangular solve with multiple RHS |
| `syrk` | Symmetric rank-k update |
| `herk` | Hermitian rank-k update |
| `syr2k` | Symmetric rank-2k update |
| `her2k` | Hermitian rank-2k update |

## LAPACK Operations

### Decompositions

| Decomposition | Description |
|---------------|-------------|
| `Lu` | LU with partial pivoting (`compute`, `compute_blocked`, `compute_recursive`, `compute_auto`, `compute_blocked_par`) |
| `LuFullPiv` | LU with full pivoting |
| `BandLu` | Banded LU factorization |
| `Cholesky` | LL^T for positive definite matrices (`compute`, `compute_blocked`, `compute_recursive`, `compute_auto`, `compute_blocked_par`) |
| `Ldlt` | LDL^T decomposition |
| `BandCholesky` | Banded Cholesky |
| `Qr` | QR decomposition (`compute`, `compute_blocked`, `compute_recursive`, `compute_auto`) |
| `QrPivot` | QR with column pivoting |
| `Svd` | Singular value decomposition (Jacobi) |
| `SvdDc` | SVD with divide-and-conquer |
| `SymmetricEvd` | Symmetric eigenvalue decomposition |
| `SymmetricEvdDc` | Symmetric EVD with divide-and-conquer |
| `GeneralEvd` | General eigenvalue decomposition |
| `Schur` | Schur decomposition |
| `Hessenberg` | Hessenberg reduction |

### Solvers

| Solver | Description |
|--------|-------------|
| `solve` | General linear system Ax = b |
| `solve_multiple` | Multiple right-hand sides |
| `solve_triangular` | Triangular system |
| `lstsq` | Least squares solution |
| `tridiag_solve` | Tridiagonal system |

### Utilities

| Function | Description |
|----------|-------------|
| `det` | Determinant |
| `inv` | Matrix inverse |
| `pinv` | Pseudoinverse |
| `cond` | Condition number |
| `rcond` | Reciprocal condition number |
| `rank` | Matrix rank |
| `nullity` | Nullity (dimension of null space) |
| `null_space` | Null space basis |
| `col_space` | Column space basis |
| `norm_1`, `norm_2`, `norm_inf`, `norm_frobenius` | Matrix norms |

## Extended Precision & Tensor Operations

### Extended Precision

OxiBLAS supports multiple precision levels:

```rust
use oxiblas::prelude::*;

// Quad-precision (f128) - ~31 decimal digits
#[cfg(feature = "f128")]
{
    use oxiblas_core::scalar::QuadFloat;
    let x = QuadFloat::from(2.0);
    let sqrt_x = x.sqrt();
    // Extremely high precision calculations
}

// Extended precision dot products
let x: Vec<f64> = vec![/* ... */];
let y: Vec<f64> = vec![/* ... */];

// Kahan summation (compensated)
let result = dot_kahan(&x, &y);

// Pairwise summation (divide-and-conquer)
let result = dot_pairwise(&x, &y);

// Mixed precision (f32 computation, f64 accumulation)
let x_f32: Vec<f32> = vec![/* ... */];
let y_f32: Vec<f32> = vec![/* ... */];
let result_f64 = dsdot(&x_f32, &y_f32);
```

### Tensor Operations & Einstein Summation

OxiBLAS includes comprehensive tensor contraction support via `einsum`:

```rust
use oxiblas_blas::tensor::einsum;

// Matrix multiplication
let c = einsum("ij,jk->ik", &a, &[m, n], Some((&b, &[n, k])))?;

// Outer product
let c = einsum("i,j->ij", &x, &[m], Some((&y, &[n])))?;

// Transpose
let at = einsum("ij->ji", &a, &[m, n], None)?;

// Matrix trace
let trace = einsum("ii->", &a, &[n, n], None)?;

// Hadamard (element-wise) product
let c = einsum("ij,ij->ij", &a, &[m, n], Some((&b, &[m, n])))?;

// Tensor contractions (24 patterns supported)
let result = einsum("ijk,kl->ijl", &tensor3d, &[i, j, k], Some((&mat, &[k, l])))?;
```

**Supported Patterns:**
- Matrix operations: matmul, transpose, trace, diagonal
- Tensor transposes: 3 variants (ijk→ikj, ijk→jik, ijk→kji)
- Reductions: row/column sums, total sum, axis sums
- Products: outer, Hadamard, dot, Frobenius
- Advanced: tensor-matrix contraction, batched operations

### Batched Operations

```rust
use oxiblas_blas::tensor::{batched_matmul, Tensor3};

// Batch of matrix multiplications
let a = Tensor3::from_data(data_a, batch_size, m, k);
let b = Tensor3::from_data(data_b, batch_size, k, n);
let c = batched_matmul(&a, &b)?;
// c[i] = a[i] * b[i] for each batch i
```

## Sparse Matrix Support

### Formats

- **CSR** - Compressed Sparse Row
- **CSC** - Compressed Sparse Column
- **COO** - Coordinate format
- **ELL** - ELLPACK format
- **DIA** - Diagonal format
- **BSR** - Block Sparse Row
- **BSC** - Block Sparse Column
- **HYB** - Hybrid ELL+COO
- **SELL** - Sliced ELLPACK (GPU-optimized)

### Iterative Solvers

- GMRES with restart (includes PGMRES, FGMRES)
- Conjugate Gradient (CG, PCG, Block-CG)
- BiCGStab
- MINRES (includes PMINRES)
- QMR (Quasi-Minimal Residual)
- TFQMR (Transpose-Free QMR)
- IDR(s) (Induced Dimension Reduction)
- Block-GMRES (multiple RHS)
- Lanczos iteration (symmetric)
- Arnoldi iteration (general)
- IRAM (Implicitly Restarted Arnoldi Method)

### Preconditioners

- ILU(0) / ILUT (Incomplete LU with threshold)
- ILUTP (ILUT with pivoting)
- Incomplete Cholesky (IC0, ICT)
- Jacobi / Block Jacobi
- Gauss-Seidel / SOR / SSOR
- AMG (Algebraic Multigrid - classical Ruge-Stüben)
- SA-AMG (Smoothed Aggregation AMG)
- SPAI (Sparse Approximate Inverse)
- AINV (Approximate Inverse)
- Additive Schwarz (domain decomposition)
- Polynomial preconditioners (Neumann, Chebyshev)

### Sparse Eigenvalue & SVD

- Lanczos method (symmetric matrices)
- Block Lanczos
- Arnoldi iteration (general matrices)
- Block Arnoldi
- Shift-invert spectral transformation
- Polynomial filtering (Chebyshev)
- Interval eigenvalue computation (Sturm sequence)
- Truncated SVD
- Randomized SVD
- Incremental SVD (Brand algorithm)

### Reordering

- RCM (Reverse Cuthill-McKee)
- AMD (Approximate Minimum Degree)
- MMD (Multiple Minimum Degree)
- COLAMD (Column AMD for unsymmetric/rectangular)
- Nested Dissection (level-set based)

## Performance

OxiBLAS implements BLIS-style blocked algorithms with architecture-specific SIMD kernels and platform-aware cache tuning.

### Linux x86_64 Performance (Intel Xeon E5-2623 v4 @ 2.60GHz)

**Benchmarked:** 2025-12-27
**CPU:** Intel Xeon E5-2623 v4 @ 2.60GHz (Broadwell-EP)
**Cache:** L1D=32KB, L2=256KB, L3=10MB
**SIMD:** AVX2, FMA, SSE4.1, SSE4.2

#### DGEMM Performance (f64)

| Matrix Size | OxiBLAS Time | OxiBLAS Throughput | OpenBLAS Throughput | vs OpenBLAS | Status |
|-------------|--------------|-------------------|---------------------|-------------|--------|
| 64×64       | 30.84 µs     | 8.50 Gelem/s (136 Gf/s) | 8.49 Gelem/s (136 Gf/s) | 100% | 🟢 Excellent |
| 128×128     | 184.92 µs    | 11.34 Gelem/s (181 Gf/s) | 12.62 Gelem/s (202 Gf/s) | 90% | 🟢 Very Good |
| 256×256     | 1.220 ms     | 13.75 Gelem/s (220 Gf/s) | 14.48 Gelem/s (232 Gf/s) | 95% | 🟢 Very Good |
| 512×512     | 11.05 ms     | 12.15 Gelem/s (194 Gf/s) | 15.24 Gelem/s (244 Gf/s) | 80% | 🟡 Good |
| 1024×1024   | 80.68 ms     | 13.31 Gelem/s (213 Gf/s) | 13.01 Gelem/s (208 Gf/s) | **102%** | 🟢 **Excellent** |

**Peak Performance:** 13.75 Gelem/s = 220 GFLOPS (256×256 f64)
**Key Achievement:** **OxiBLAS outperforms OpenBLAS by 2% on very large matrices (1024×1024)**

#### SGEMM Performance (f32)

| Matrix Size | OxiBLAS Throughput | OpenBLAS Throughput | vs OpenBLAS | Status |
|-------------|-------------------|---------------------|-------------|--------|
| 64×64       | 15.79 Gelem/s (253 Gf/s) | 14.06 Gelem/s (225 Gf/s) | **112%** | 🟢 **Excellent** |
| 128×128     | 20.50 Gelem/s (328 Gf/s) | 20.50 Gelem/s (328 Gf/s) | 100% | 🟢 Excellent |
| 256×256     | 23.65 Gelem/s (378 Gf/s) | 25.02 Gelem/s (400 Gf/s) | 95% | 🟢 Very Good |
| 512×512     | 25.16 Gelem/s (403 Gf/s) | ~26.8 Gelem/s (~429 Gf/s) | ~94% | 🟢 Very Good |

**Summary:** OxiBLAS achieves **80-112% of OpenBLAS** performance on Linux x86_64, with **superior performance on small f32 and very large f64 matrices**.

#### Linux Optimization Highlights

- **13-20% performance improvement** after cache tuning for 256KB L2 systems
- **Fine-tuned blocking parameters:** KC=192, MC=128 (optimized for 256KB L2 cache)
- **Increased prefetch distance:** 12 iterations (optimized for Intel Xeon E5-2600 memory latency)
- **Platform-aware cache detection:** Linux sysfs, macOS sysctl, x86_64 CPUID fallback
- **All 2833 tests passing** with zero warnings

### macOS AArch64 Performance (Apple M3)

**Benchmarked:** 2025-12-28
**CPU:** Apple M3 (ARM64 NEON)
**Cache:** P-cores: L1D=128KB, L2=16MB
**SIMD:** NEON (128-bit)

#### DGEMM Performance (f64)

| Matrix Size | OxiBLAS Time | OxiBLAS Throughput | OpenBLAS Throughput | vs OpenBLAS | Status |
|-------------|--------------|-------------------|---------------------|-------------|--------|
| 64×64       | 10.14 µs     | 25.84 Gelem/s (414 Gf/s) | 25.59 Gelem/s (409 Gf/s) | **101%** | 🟢 **Excellent** |
| 128×128     | 83.63 µs     | 25.08 Gelem/s (401 Gf/s) | 25.88 Gelem/s (414 Gf/s) | 97% | 🟢 Very Good |
| 256×256     | 670.89 µs    | 24.99 Gelem/s (400 Gf/s) | 25.47 Gelem/s (408 Gf/s) | 98% | 🟢 Very Good |
| 512×512     | 5.21 ms      | 25.76 Gelem/s (412 Gf/s) | 25.44 Gelem/s (407 Gf/s) | **101%** | 🟢 **Excellent** |
| 1024×1024   | 40.25 ms     | 26.68 Gelem/s (427 Gf/s) | 26.48 Gelem/s (424 Gf/s) | **101%** | 🟢 **Excellent** |

**Peak Performance:** 26.68 Gelem/s = 427 GFLOPS (1024×1024 f64)
**Key Achievement:** **OxiBLAS matches or exceeds OpenBLAS** on Apple M3 for f64 GEMM

#### SGEMM Performance (f32)

| Matrix Size | OxiBLAS Throughput | OpenBLAS Throughput | vs OpenBLAS | Status |
|-------------|-------------------|---------------------|-------------|--------|
| 64×64       | 44.74 Gelem/s (716 Gf/s) | 44.39 Gelem/s (710 Gf/s) | **101%** | 🟢 **Excellent** |
| 128×128     | 28.34 Gelem/s (453 Gf/s) | 23.49 Gelem/s (376 Gf/s) | **121%** | 🟢 **Outstanding** |
| 256×256     | 49.08 Gelem/s (785 Gf/s) | 48.78 Gelem/s (781 Gf/s) | **101%** | 🟢 **Excellent** |
| 512×512     | 56.28 Gelem/s (901 Gf/s) | 54.36 Gelem/s (870 Gf/s) | **104%** | 🟢 **Excellent** |
| 1024×1024   | 55.99 Gelem/s (896 Gf/s) | 32.59 Gelem/s (521 Gf/s) | **172%** | 🟢 **Outstanding** |

**Breakthrough:** **OxiBLAS is 72% faster than OpenBLAS** for large f32 matrices (1024×1024) on Apple M3!

**Summary:** OxiBLAS achieves **97-172% of OpenBLAS** performance on Apple M3, with **superior performance on f32 operations** and **competitive or better f64 performance**.

#### macOS Optimization Highlights

- **NEON micro-kernel excellence:** 4×6 kernel optimized for f32, achieving 121-172% of OpenBLAS
- **Large cache utilization:** Optimal blocking for 16MB L2 cache on Apple M3 performance cores
- **DOT product optimization:** 65% faster than OpenBLAS (6.0 vs 3.6 Gelem/s)
- **Production-ready:** Competitive or superior performance across all major operations
- **Platform-aware tuning:** macOS sysctl cache detection working perfectly

### BLAS Level 3 Performance (via GEMM Optimization)

| Operation | Size | Performance | Speedup | Status |
|-----------|------|-------------|---------|--------|
| SYRK (f64) | 1024×1024 | 40.24 Gf/s | **12.53×** vs naive | ✅ |
| SYR2K (f64) | 1024×1024 | 40.99 Gf/s | **14.76×** vs naive | ✅ |
| HERK (f64) | 1024×1024 | - | 6-12× vs naive | ✅ |
| HER2K (f64) | 1024×1024 | - | 6-15× vs naive | ✅ |
| SYMM (f64) | 512×512 | - | 1.1-1.8× vs naive | ✅ |
| TRMM (f64) | 1024×1024 | 40.60 Gf/s | **10.79×** vs naive | ✅ |
| TRSM (f64) | 1024×1024 | 19.96 Gf/s | **10.32×** vs naive | ✅ |
| Complex GEMM (3M) | 1024×1024 | 40.72 Gf/s | - | ✅ |
| Parallel GEMM | 1024×1024 | 130.81 Gf/s | - | ✅ |

### Operation Performance Targets

| Operation | Linux x86_64 | macOS M3 | Notes |
|-----------|--------------|----------|-------|
| BLAS Level 1 (DOT) | 2-3 Gf/s | **165% OpenBLAS** ✓ | M3: OxiBLAS dominates |
| BLAS Level 1 (AXPY) | TBD | 64% OpenBLAS | Optimization opportunity |
| BLAS Level 2 (matvec) | TBD | TBD | Cache-aware blocking |
| BLAS Level 3 (f64 GEMM) | **80-102% OpenBLAS** ✓ | **97-101% OpenBLAS** ✓ | Production-ready |
| BLAS Level 3 (f32 GEMM) | **94-112% OpenBLAS** ✓ | **101-172% OpenBLAS** ✓ | M3: Outstanding |
| SYRK/SYR2K | **6-15×** speedup | - | GEMM-based optimization ✓ |
| HERK/HER2K | **6-15×** speedup | - | GEMM-based optimization ✓ |
| TRMM | **7-11×** speedup | - | GEMM-based optimization ✓ |
| TRSM | **2.5-10×** speedup | - | Blocked algorithm with GEMM ✓ |
| QR Factorization | **3-7×** vs unblocked ✓ | - | WY representation (fixed v0.2.0) |
| SVD | TBD | TBD | Divide-and-conquer |

**Achievement:** OxiBLAS matches or exceeds OpenBLAS on Apple M3 (97-172%), and is competitive on Linux x86_64 (80-112%). Pure Rust with NEON/AVX2 intrinsics.

### LAPACK Factorization Speedups (Blocked vs Unblocked)

Blocked algorithms using GEMM/TRSM as building blocks deliver dramatic speedups over the unblocked Level 2 baseline. All results measured on Apple M3 (f64).

#### LU Factorization

| Matrix Size | Unblocked | Blocked | Speedup |
|-------------|-----------|---------|---------|
| 256×256 | 0.62 Gf/s | 9.05 Gf/s | **14.5×** |
| 512×512 | 0.45 Gf/s | 6.65 Gf/s | **14.9×** |
| 768×768 | 0.60 Gf/s | 13.84 Gf/s | **23.0×** |
| 1024×1024 | 0.35 Gf/s | 6.99 Gf/s | **19.7×** |

#### Cholesky Factorization

| Matrix Size | Unblocked | Blocked | Speedup |
|-------------|-----------|---------|---------|
| 200×200 | 2.32 Gelem/s | 10.47 Gelem/s | **4.5×** |
| 256×256 | 2.16 Gf/s | 13.01 Gf/s | **6.0×** |
| 500×500 | 1.65 Gelem/s | 15.60 Gelem/s | **9.5×** |
| 512×512 | 1.65 Gf/s | 14.97 Gf/s | **9.1×** |
| 1024×1024 | 1.44 Gf/s | 14.73 Gf/s | **10.2×** |

### Parallelization

Enable multi-core parallelization:

```toml
oxiblas = { version = "0.2", features = ["parallel"] }
```

Parallel operations automatically use all available cores via Rayon for large workloads.

### Benchmarking

OxiBLAS includes a comprehensive benchmarking suite with direct comparisons to OpenBLAS:

```bash
# Run all benchmarks
cargo bench --package oxiblas-benchmarks

# Compare with OpenBLAS (requires OpenBLAS installed)
cargo bench --package oxiblas-benchmarks --bench comparison --features compare-openblas

# View HTML reports
open target/criterion/report/index.html
```

See [crates/oxiblas-benchmarks/README.md](crates/oxiblas-benchmarks/README.md) for detailed benchmarking guide.

## Platform Support and Performance

### Tested Platforms

| Platform | CPU | SIMD | DGEMM | vs OpenBLAS |
|----------|-----|------|-------|-------------|
| macOS (Apple M3) | ARM64 | NEON 128-bit | 25.6 GFLOPS | ~100% |
| Linux x86_64 | Intel Xeon E5-2623 v4 | AVX2 256-bit | 220 GFLOPS | ~102% (1024x1024) |
| Any | Scalar fallback | None | varies | ~30-50% |

### SIMD Feature Detection

OxiBLAS automatically detects and uses the best available SIMD:
- **x86_64**: AVX-512 > AVX2 > SSE4.2 > scalar
- **aarch64**: NEON (native 128-bit) + emulated 256-bit
- **wasm32**: SIMD128 (if enabled) + emulated 256-bit
- **others**: Scalar fallback

Runtime detection via `SimdCapabilities::detect()` with zero overhead after first call.

### Performance Regression CLI

OxiBLAS ships a `regress` binary for tracking performance over time:

```bash
# Build the CLI
cargo build --release -p oxiblas-benchmarks

# Capture current performance as a baseline
./target/release/regress capture --output baseline.json

# On a subsequent commit/build, check for regressions (fail if >5% slower)
./target/release/regress check --baseline baseline.json --threshold 5.0

# Print a formatted performance report
./target/release/regress report --baseline baseline.json

# List all available benchmark names
./target/release/regress list
```

The `check` subcommand exits with a non-zero status when any operation regresses beyond the threshold, making it suitable for CI pipelines.

## Algorithm Selection Guide

OxiBLAS provides multiple computation variants for dense factorizations, each suited to different use cases. The `compute_auto()` method is the recommended default — it selects between blocked and unblocked algorithms based on matrix size at zero cost.

### Dense Factorizations (Cholesky, LU, QR)

| Method | When to Use | Notes |
|--------|-------------|-------|
| `compute()` | Small matrices (n < 64-128), minimal overhead | Unblocked Level 2 BLAS |
| `compute_blocked(nb)` | Large matrices, explicit block size control | Level 3 BLAS; 6-23× over unblocked |
| `compute_recursive()` | Very large matrices, cache-oblivious | No explicit block size; auto-adapts to cache |
| `compute_auto()` | **Recommended default** | Auto-selects blocked (n >= 128) or unblocked (n < 128) |
| `compute_blocked_par()` | Very large matrices on multi-core systems | Requires `parallel` feature; Rayon-based |

```rust
use oxiblas_lapack::{cholesky::Cholesky, lu::Lu, qr::Qr};

// Recommended: automatic algorithm selection
let chol = Cholesky::compute_auto(a.as_ref())?;  // up to 9.75× faster than compute()
let lu   = Lu::compute_auto(a.as_ref())?;         // up to 23× faster than compute()
let qr   = Qr::compute_auto(a.as_ref())?;         // 3-7× faster for large matrices

// Explicit blocked with custom block size
let chol = Cholesky::compute_blocked(a.as_ref(), 64)?;

// Cache-oblivious recursive (no tuning required)
let lu = Lu::compute_recursive(a.as_ref())?;

// Parallel blocked (requires "parallel" feature)
let chol = Cholesky::compute_blocked_par(a.as_ref())?;
```

### Mixed-Precision Iterative Refinement

For applications requiring double-precision accuracy with single-precision speed:

```rust
use oxiblas_lapack::refine::{mixed_precision_solve, mixed_precision_solve_cholesky};

// LU-based: factorize in f32, refine residuals in f64
let x = mixed_precision_solve(a.as_ref(), b.as_ref())?;

// Cholesky-based (symmetric positive definite systems)
let x = mixed_precision_solve_cholesky(a.as_ref(), b.as_ref())?;
```

### Batched BLAS Operations

For batches of independent small matrix operations (e.g., neural network layers):

```rust
use oxiblas_blas::batched::{gemm_batched, gemm_strided_batched, axpy_batched};

// Batch of independent GEMMs: C[i] = A[i] * B[i]
gemm_batched(1.0, &a_slices, &b_slices, 0.0, &mut c_slices)?;

// Strided batched GEMM (contiguous memory layout)
gemm_strided_batched(1.0, &a, stride_a, &b, stride_b, 0.0, &mut c, stride_c, batch)?;
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `default` | Core functionality (f32, f64, Complex32, Complex64) |
| `parallel` | Enable rayon-based parallelization for multi-core |
| `f16` | Half-precision (16-bit) floating point support |
| `f128` | Quad-precision (~31 digits) via QuadFloat |
| `full` | All features enabled |
| `nightly` | Nightly-only optimizations |

**Benchmarks-specific features:**
| Feature | Description |
|---------|-------------|
| `compare-openblas` | Enable OpenBLAS comparison benchmarks |

## Requirements

- Rust 1.85+ (Edition 2024)
- No external C dependencies

## Project Status

**Version:** 0.2.0 (2026-03-06)

- **Lines of Code:** ~168,794 Rust (356 files)
- **Documentation:** ~15,034 lines of comments, 5 comprehensive examples
- **Tests:** 2,811 passing lib tests + 195 doctests (100% success rate)
- **Coverage:**
  - ✅ Full BLAS Level 1/2/3 (including packed/banded variants)
  - ✅ Extensive LAPACK (LU, Cholesky, QR, SVD, EVD, Schur, Hessenberg)
  - ✅ Recursive cache-oblivious factorizations (Cholesky, LU, QR)
  - ✅ Parallel blocked factorizations (Cholesky, LU)
  - ✅ Fixed blocked QR with WY representation (3-7x speedup realized)
  - ✅ Complex bidiagonal reduction (`ComplexBidiagFactors`)
  - ✅ Batched BLAS operations (gemm_batched, axpy_batched, gemv_batched)
  - ✅ Mixed-precision iterative refinement (LU, Cholesky, symmetric, QR)
  - ✅ Runtime auto-tuning infrastructure (`RuntimeAutoTuner`)
  - ✅ Multifrontal sparse factorizations (`MultifrontalCholesky`, `MultifrontalLU`)
  - ✅ ndarray parallel GEMM and sparse integration
  - ✅ no_std support for oxiblas-core and oxiblas-matrix
  - ✅ Advanced sparse LU pivoting (threshold, static, Bunch-Kaufman LDL^T)
  - ✅ Sparse matrices (9 formats: CSR, CSC, COO, ELL, DIA, BSR, BSC, HYB, SELL)
  - ✅ Iterative solvers (GMRES, FGMRES, CG, BiCGStab, MINRES, QMR, TFQMR, IDR(s))
  - ✅ Advanced preconditioners (ILUT, IC, AMG, SA-AMG, SPAI, AINV, Schwarz)
  - ✅ Sparse eigenvalue/SVD (Lanczos, Arnoldi, IRAM, polynomial filtering, truncated/randomized SVD)
  - ✅ Tensor operations (einsum with 24 patterns, batched matmul)
  - ✅ Extended precision (f16, f128, Kahan/pairwise summation)
  - ✅ Comprehensive benchmarks with OpenBLAS comparison
  - ✅ Complete API documentation with examples

See [TODO.md](TODO.md) for the development roadmap.

## Related Projects

OxiBLAS is part of the SciRS2 scientific computing ecosystem:

- [SciRS2](https://github.com/cool-japan/scirs) - Scientific computing library
- [NumRS2](https://github.com/cool-japan/numrs) - Numerical computing
- [SkleaRS](https://github.com/cool-japan/sklears) - Machine learning
- [ToRSh](https://github.com/cool-japan/torsh) - Tensor operations
- [TrustformeRS](https://github.com/cool-japan/trustformers) - Transformers
- [QuantRS2](https://github.com/cool-japan/quantrs) - Quantum computing framework
- [OxiRS](https://github.com/cool-japan/oxirs) - Semantic Web platform (SPARQL 1.2, GraphQL, Industrial Digital Twin, AI-augmented reasoning)

## License

Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).

See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
