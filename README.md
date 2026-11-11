# OxiBLAS

**Pure Rust BLAS/LAPACK implementation for the SciRS2 ecosystem**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.89%2B-orange.svg)](https://www.rust-lang.org/)

OxiBLAS is a production-grade, pure Rust implementation of BLAS (Basic Linear Algebra Subprograms) and LAPACK (Linear Algebra PACKage). Designed as the foundational linear algebra library for the [SciRS2](https://github.com/cool-japan/scirs) scientific computing ecosystem.

## Features

- **Pure Rust** - No C, C++ or Fortran in the libraries or their normal dependencies (two dev-dependencies of the benches and tests compile C; see [Requirements](#requirements)), fully portable, safe by default
- **SIMD Optimized** - Custom SIMD layer using `core::arch` intrinsics
  - x86_64: SSE4.2, AVX2/FMA (256-bit), AVX-512F (512-bit)
  - AArch64: NEON (128-bit). `oxiblas-core`'s nightly-only `sve-nightly` feature adds SVE length queries and stand-alone `sve_dot_f64`, `sve_dot_f32` and `sve_axpy_f64` kernels, which the BLAS routines do not call yet. `oxiblas-core` declares the `stdarch_aarch64_sve` feature gate for that build; it is compile-checked on nightly with `-C target-feature=+sve`, but has not run on SVE hardware
  - wasm32: SIMD128 (when enabled at compile time)
  - Fallback: Scalar operations
- **Cache Aware** - BLIS-style blocked algorithms for optimal cache usage
- **Parallel** - Optional rayon-based parallelization, opt-in per call (`gemm_with_par`, `trsm_with_par`, the `compute_with_par` of the symmetric EVDs and `SvdDc`); large GEMMs run on a 2D driver whose results are bitwise identical to the sequential path
- **Complete BLAS** - Full Level 1, 2, 3 operations (including packed/banded)
- **CBLAS Interface** - 152 `extern "C"` entry points: every routine in netlib `cblas.h` 3.12.1 except `cblas_xerbla`
- **Extensive LAPACK** - LU, Cholesky, QR, SVD, EVD, Schur, Hessenberg, QZ (generalized Schur)
- **Extended Precision** - f16, f128 (quad precision), Kahan/pairwise summation
- **Tensor Operations** - Einstein summation (22 patterns), batched matmul, batched BLAS
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
| `f128` (QuadFloat) | Double-double: about 31 decimal digits for `+ − × ÷` and `sqrt`; the elementary functions are less accurate (12 to 23 digits in spot checks: `ln` about 12-14, `exp` about 18) | `f128` |
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
assert!((det - 5.0_f64).abs() < 1e-10); // det = 2*3 - 1*1 = 5
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
cargo run -p oxiblas --example basic_blas

# LAPACK decompositions (LU, QR, Cholesky, SVD, EVD)
cargo run -p oxiblas --example lapack_decompositions

# Extended precision (f128, Kahan, pairwise summation)
cargo run -p oxiblas --example extended_precision --features f128

# Tensor operations & Einstein summation (22 patterns)
cargo run -p oxiblas --example tensor_operations

# Sparse matrices (CSR, CSC, COO, CG and GMRES, a Jacobi preconditioner)
cargo run -p oxiblas --example sparse_matrices
```

The timing examples `lu_bench`, `cholesky_bench`, `trmm_trsm_bench`, `syrk_syr2k_bench`, `symm_hemm_bench`, `complex_gemm_bench`, `evd_svd_bench` and `parallel_bench` (which needs `--features parallel`) are meant to run with `--release`; `evd_svd_bench` takes `-- [--reps N] [SIZE ...]` and also prints residual and orthogonality errors.

The workspace root is a virtual manifest (no `[package]`), so `-p oxiblas` (or
running from `crates/oxiblas/`) is required — plain `cargo run --example NAME`
from the repo root will fail with "no such example".

See [crates/oxiblas/examples/](crates/oxiblas/examples/) for complete source code.

## Crate Structure

OxiBLAS is organized as a workspace with specialized sub-crates:

| Crate | Description |
|-------|-------------|
| `oxiblas` | Unified re-exports and prelude |
| `oxiblas-core` | Core traits, SIMD abstractions, memory management, scalar types (f16, f128); `no_std` |
| `oxiblas-matrix` | Matrix types (`Mat`, `MatRef`, `MatMut`, `DiagRef`); `no_std` |
| `oxiblas-blas` | BLAS Level 1/2/3 operations, CBLAS interface, tensor operations, einsum, batched BLAS |
| `oxiblas-lapack` | LAPACK decompositions and solvers (blocked, recursive, parallel, auto variants) |
| `oxiblas-sparse` | Sparse matrix formats (9 types), multifrontal factorizations, and algorithms |
| `oxiblas-ndarray` | Integration with ndarray (parallel GEMM, sparse integration) |
| `oxiblas-benchmarks` | Performance benchmarks with OpenBLAS comparison (not in meta crate) |

`crates/oxiblas-ffi`, the former C FFI bindings, was retired in v0.2.0 and is excluded from the workspace.

## BLAS Operations

### Level 1 (Vector-Vector)

| Function | Description |
|----------|-------------|
| `dot`, `dotc`, `dotu_c32`, `dotu_c64` | Dot product (`dotc` conjugates x) |
| `dot_kahan`, `dot_pairwise` | Compensated and pairwise-summation f64 dot products |
| `dsdot`, `sdsdot` | f32 dot products accumulated in f64 |
| `axpy` | y = alpha * x + y |
| `scal`, `csscal`, `zdscal` | x = alpha * x (`csscal`/`zdscal`: real alpha, complex x) |
| `copy` | y = x |
| `swap` | Swap x and y |
| `nrm2`, `cnrm2`, `scnrm2`, `dznrm2` | Euclidean norm |
| `asum`, `casum`, `scasum`, `dzasum` | Sum of absolute values |
| `cabs1`, `scabs1`, `dcabs1` | \|re\| + \|im\| of a complex value |
| `iamax`, `iamin` | Index of max/min absolute value |
| `rot`, `zdrot` | Apply a Givens rotation |
| `rotg`, `crotg` | Generate a Givens rotation |
| `rotm`, `rotmg` | Apply/generate a modified Givens rotation |

`*_strided` variants (`dot_strided`, `axpy_strided`, `nrm2_strided`, `rotm_strided`, ...) take vectors with an increment; `dot_par`, `axpy_par`, `scal_par`, `nrm2_par` and `asum_par` run on Rayon.

### Level 2 (Matrix-Vector)

| Function | Description |
|----------|-------------|
| `gemv` | General matrix-vector multiply |
| `symv` | Symmetric matrix-vector multiply |
| `hemv` | Hermitian matrix-vector multiply |
| `trmv` | Triangular matrix-vector multiply |
| `trsv` | Triangular solve |
| `ger`, `gerc` | Rank-1 update (`gerc` conjugates y) |
| `syr`, `spr` | Symmetric rank-1 update (full, packed) |
| `her`, `hpr` | Hermitian rank-1 update (full, packed) |
| `syr2`, `spr2` | Symmetric rank-2 update (full, packed) |
| `her2`, `hpr2` | Hermitian rank-2 update (full, packed) |
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
| `gemm` | General matrix-matrix multiply (sequential) |
| `gemm_with_par` | `gemm` with an explicit `Par`, for multi-threaded runs |
| `gemm3m_c32`, `gemm3m_c64` | Complex GEMM with the 3M method |
| `gemm_strassen`, `gemm_winograd`, `gemm_cache_oblivious` | Alternative GEMM algorithms |
| `gemmt` | GEMM updating one triangle of C |
| `gemm_batched` | Batch of independent GEMMs |
| `symm` | Symmetric matrix-matrix multiply |
| `hemm` | Hermitian matrix-matrix multiply |
| `trmm` | Triangular matrix-matrix multiply |
| `trsm` | Triangular solve with multiple RHS (sequential; `trsm_with_par` takes a `Par`) |
| `syrk` | Symmetric rank-k update |
| `herk` | Hermitian rank-k update |
| `syr2k` | Symmetric rank-2k update |
| `her2k` | Hermitian rank-2k update |

## LAPACK Operations

### Decompositions

| Decomposition | Description |
|---------------|-------------|
| `Lu` | LU with partial pivoting (`compute`, `compute_blocked`, `compute_with_block_size`, `compute_recursive`, `compute_auto`, `compute_blocked_par`) |
| `LuFullPiv` | LU with full pivoting |
| `BandLu` | Banded LU factorization |
| `Cholesky` | LL^T for positive definite matrices (`compute`, `compute_blocked`, `compute_with_block_size`, `compute_recursive`, `compute_auto`, `compute_blocked_par`) |
| `Ldlt` | LDL^T decomposition |
| `BandCholesky` | Banded Cholesky |
| `Qr` | QR decomposition (`compute`, `compute_blocked`, `compute_recursive`, `compute_auto`) |
| `QrPivot` | QR with column pivoting |
| `Svd` | Singular value decomposition (Jacobi) |
| `SvdDc` | SVD with divide-and-conquer (LAPACK `dgesdd` scheme; `compute_with_par`) |
| `SymmetricEvd` | Symmetric eigenvalue decomposition (tridiagonal reduction + implicit QR; `compute_with_par`) |
| `SymmetricEvdDc` | Symmetric EVD with divide-and-conquer (LAPACK `DSYEVD` scheme; `compute_with_par`) |
| `HermitianEvd`, `HermitianEvdDc` | Hermitian eigenvalue decomposition (implicit QR; divide-and-conquer as LAPACK `ZHEEVD`) |
| `GeneralEvd`, `ComplexGeneralEvd` | General eigenvalue decomposition |
| `GeneralizedEvd` | Generalized eigenproblem Ax = λBx, via LU of B |
| `Qz` | Generalized Schur decomposition of (A, B), equivalent to LAPACK `DGGES` (`compute`, `compute_sorted`, `reorder`) |
| `Schur`, `ComplexSchur` | Schur decomposition |
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

## CBLAS Interface

`oxiblas_blas::cblas` provides 152 `extern "C"` entry points: every routine declared in netlib `cblas.h` 3.12.1 except the `cblas_xerbla` error handler. Both layouts are supported, and valid calls follow reference CBLAS semantics.

```rust
use oxiblas_blas::cblas::{CblasLayout, CblasTranspose, cblas_dgemm};

let a = [1.0, 2.0, 3.0, 4.0];
let b = [5.0, 6.0, 7.0, 8.0];
let mut c = [0.0; 4];

// SAFETY: all three buffers hold a 2×2 matrix with leading dimension 2.
unsafe {
    cblas_dgemm(
        CblasLayout::RowMajor,
        CblasTranspose::NoTrans,
        CblasTranspose::NoTrans,
        2,
        2,
        2,
        1.0,
        a.as_ptr(),
        2,
        b.as_ptr(),
        2,
        0.0,
        c.as_mut_ptr(),
        2,
    );
}
assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
```

Differences from reference CBLAS:

- Invalid arguments are a silent no-op. There is no `cblas_xerbla`, which in reference CBLAS reports the error and terminates the process.
- The layout, transpose, uplo, diag and side arguments are `#[repr(C)]` Rust enums with the `cblas.h` values, so a C caller that passes any other value triggers undefined behaviour.
- `cblas_i?amax` return a 32-bit `int` rather than `size_t`, and there is no ILP64 (`_64`) API.
- The symbols live in the `oxiblas-blas` rlib; this workspace does not build a C-linkable library.
- The symbols are `#[no_mangle]`, so every `cblas_*` reference in a binary that links `oxiblas-blas` binds to OxiBLAS, even when another BLAS is linked too.

The enum and `i?amax` changes break public signatures and are planned for 0.3.0. The remaining NaN/Inf differences in the triangular routines are planned for 0.2.4; see [TODO.md](TODO.md).

## Extended Precision & Tensor Operations

### Extended Precision

OxiBLAS supports multiple precision levels:

```rust
use oxiblas_blas::level1::{dot_kahan, dot_pairwise, dsdot};
use oxiblas_core::scalar::{QuadFloat, Real}; // QuadFloat requires the `f128` feature

// Quad precision (f128, double-double): + - * / and sqrt to about 31 decimal digits;
// exp, ln and the other elementary functions are less accurate
let x = QuadFloat::from(2.0);
let sqrt_x = x.sqrt();

// Extended precision dot products
let x = vec![1.0_f64, 2.0, 3.0];
let y = vec![4.0_f64, 5.0, 6.0];

// Kahan summation (compensated)
let result = dot_kahan(&x, &y);

// Pairwise summation (divide-and-conquer)
let result = dot_pairwise(&x, &y);

// Mixed precision (f32 input, f64 accumulation)
let x_f32 = vec![1.0_f32, 2.0, 3.0];
let y_f32 = vec![4.0_f32, 5.0, 6.0];
let result_f64 = dsdot(&x_f32, &y_f32);
```

### Tensor Operations & Einstein Summation

OxiBLAS includes comprehensive tensor contraction support via `einsum`:

```rust
use oxiblas_blas::tensor::einsum;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (m, n, k) = (2, 3, 2);
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // m×n
    let b = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // n×k
    let s = vec![1.0, 2.0, 3.0, 4.0]; // k×k
    let (x, y) = (vec![1.0, 2.0], vec![3.0, 4.0, 5.0]);

    // Matrix multiplication
    let c = einsum("ij,jk->ik", &a, &[m, n], Some((&b, &[n, k])))?;

    // Outer product
    let outer = einsum("i,j->ij", &x, &[m], Some((&y, &[n])))?;

    // Transpose
    let at = einsum("ij->ji", &a, &[m, n], None)?;

    // Matrix trace
    let trace = einsum("ii->", &s, &[k, k], None)?;

    // Hadamard (element-wise) product
    let hadamard = einsum("ij,ij->ij", &a, &[m, n], Some((&a, &[m, n])))?;

    // Tensor contractions (22 patterns supported)
    let tensor3d = vec![1.0; 2 * 3 * 2];
    let result = einsum("ijk,kl->ijl", &tensor3d, &[2, 3, 2], Some((&s, &[k, k])))?;
    Ok(())
}
```

**Supported Patterns:**
- Matrix operations: matmul, transpose, trace, diagonal
- Tensor transposes: 3 variants (ijk→ikj, ijk→jik, ijk→kji)
- Reductions: row/column sums, total sum, axis sums
- Products: outer, Hadamard, dot, Frobenius
- Advanced: tensor-matrix contraction, batched operations

### Batched Operations

```rust
use oxiblas_blas::tensor::{Tensor3, batched_matmul};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (batch_size, m, k, n) = (4, 2, 3, 2);

    // Batch of matrix multiplications
    let a = Tensor3::from_data(vec![1.0_f64; batch_size * m * k], batch_size, m, k);
    let b = Tensor3::from_data(vec![2.0_f64; batch_size * k * n], batch_size, k, n);
    let c = batched_matmul(&a, &b)?;
    // c[i] = a[i] * b[i] for each batch i
    Ok(())
}
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

OxiBLAS implements BLIS-style blocked algorithms with architecture-specific SIMD kernels and cache-aware blocking.

Disregard figures published before 0.2.3. Their GFLOPS were 8 times too high, and their OpenBLAS columns measured OxiBLAS's own `cblas_*` layer. See the [CHANGELOG](CHANGELOG.md).

### Apple M3

Measured on 2026-11-09 on an Apple M3 (MacBook Pro) with 4 performance and 4 efficiency cores, macOS 27.2, using rustc 1.95.0 and Homebrew OpenBLAS 0.3.30 (`neoversen1` kernels, OpenMP). The [`comparison` benchmark](crates/oxiblas-benchmarks/README.md#comparison-benchmarks-with-openblas) reproduces the OpenBLAS tables.

- Both libraries run on the same `Mat` buffers.
- GFLOPS for real GEMM is 2n³/t.
- Ratio is OxiBLAS's speed as a percentage of OpenBLAS's, so above 100% means OxiBLAS is faster.
- Each value is the best Criterion median across separate runs: 5 runs on one thread, 3 on 4 threads and 5 on 8 threads.
- On one thread, both libraries ran on one core throughout: CPU time over each measurement window was 0.99-1.01 cores.
- No other benchmark or build ran, but a macOS background service used about one core during most runs. Taking the best run filters that out on one thread; the 8-thread figures are probably lower than on an idle machine.

#### GEMM on one thread (GFLOPS)

| n | f64 OxiBLAS | f64 OpenBLAS | f64 ratio | f32 OxiBLAS | f32 OpenBLAS | f32 ratio |
|---|---|---|---|---|---|---|
| 64 | 45.0 | 49.1 | 92% | 103.0 | 85.8 | 120% |
| 128 | 49.9 | 51.9 | 96% | 106.7 | 103.7 | 103% |
| 256 | 53.9 | 54.6 | 99% | 113.1 | 110.2 | 103% |
| 512 | 54.7 | 56.9 | 96% | 116.3 | 114.1 | 102% |
| 1024 | 55.0 | 56.8 | 97% | 116.8 | 115.8 | 101% |

f64 GEMM runs at 92-99% of OpenBLAS and f32 GEMM at 101-120%. OpenBLAS's speed at small sizes changed from one process to the next: in 4 of the 5 runs it reached about 25 GFLOPS (f64) and 50 GFLOPS (f32) at n = 64, and about 42 and 48 GFLOPS (f64) at n = 128 and 256; in the fifth, the faster values the table shows. The cause is not known. The benchmark thread already ran at the user-interactive QoS class, and requesting it again changed nothing; in a small C program that calls OpenBLAS directly, `sgemm` at n = 64 ran at 82-87 or 50-52 GFLOPS depending on where its operands were allocated. OxiBLAS varied by at most 3.3% between runs.

#### GEMM on 4 and 8 threads (GFLOPS)

| n | Threads | f64 OxiBLAS | f64 OpenBLAS | f64 ratio | f32 OxiBLAS | f32 OpenBLAS | f32 ratio |
|---|---|---|---|---|---|---|---|
| 512 | 4 | 163.3 | 158.6 | 103% | 336.5 | 353.2 | 95% |
| 1024 | 4 | 166.5 | 170.9 | 97% | 347.9 | 365.3 | 95% |
| 2048 | 4 | 170.5 | 170.7 | 100% | 360.5 | 361.6 | 100% |
| 512 | 8 | 228.1 | 134.7 | 169% | 436.3 | 326.2 | 134% |
| 1024 | 8 | 223.2 | 153.0 | 146% | 469.9 | 324.8 | 145% |
| 2048 | 8 | 223.9 | 195.2 | 115% | 474.1 | 387.8 | 122% |

On 4 threads the two libraries are within 5% of each other. OxiBLAS gains 30-40% from the efficiency cores, while OpenBLAS is 8-15% slower on 8 threads than on 4 at n = 512 and 1024, and 7-14% faster at n = 2048. With each library at its better thread count (OxiBLAS on 8; OpenBLAS on 4 at n = 512 and 1024, on 8 at n = 2048), OxiBLAS is at 115-144% of OpenBLAS. Expect different ratios on CPUs with a single core type. The 8-thread figures varied by up to 19% between runs, the 4-thread ones by at most 2.5%. An earlier, quieter run reached 236.5 GFLOPS (f64) and 496.6 GFLOPS (f32) at n = 2048 on 8 threads ([#16](https://github.com/cool-japan/oxiblas/issues/16)).

#### Other routines on one thread

| Routine | Size | OxiBLAS | OpenBLAS | Ratio |
|---------|------|---------|----------|-------|
| DOT | 1K | 92.3 ns | 127 ns | 137% |
| DOT | 100K | 13.8 µs | 15.3 µs | 112% |
| DOT | 1M | 143 µs | 156 µs | 109% |
| AXPY | 1K | 108 ns | 104 ns | 96% |
| AXPY | 100K | 17.3 µs | 19.3 µs | 112% |
| AXPY | 1M | 176 µs | 194 µs | 110% |
| NRM2 | 1K | 124 ns | 1.80 µs | 1443% |
| NRM2 | 100K | 11.4 µs | 185 µs | 1629% |
| NRM2 | 1M | 115 µs | 1.86 ms | 1610% |
| GEMV | 100×100 | 550 ns | 1.22 µs | 222% |
| GEMV | 1000×1000 | 83.7 µs | 129 µs | 154% |
| GEMV | 5000×5000 | 3.64 ms | 3.53 ms | 97% |
| GEMV | 10000×10000 | 13.4 ms | 19.2 ms | 143% |
| TRSM | 64×64 | 9.83 µs | 19.7 µs | 200% |
| TRSM | 128×128 | 57.7 µs | 96.8 µs | 168% |
| TRSM | 256×256 | 386 µs | 538 µs | 139% |
| TRSM | 512×512 | 2.83 ms | 3.30 ms | 117% |
| SYRK | 64×64 | 7.14 µs | 7.10 µs | 99% |
| SYRK | 256×256 | 339 µs | 366 µs | 108% |
| SYRK | 512×512 | 2.57 ms | 2.59 ms | 101% |
| GEMM | 128×256×128 | 165 µs | 162 µs | 98% |
| GEMM | 256×512×256 | 1.25 ms | 1.22 ms | 97% |
| GEMM | 512×256×512 | 2.45 ms | 2.36 ms | 96% |

OpenBLAS's DOT at 1K also changed speed from one process to the next (127-132 ns in 2 of the 5 runs, 309-332 ns in the other 3); the table shows the faster level. OpenBLAS's `dnrm2` in this build takes about 1.8 ns per element at every size and OxiBLAS's about 0.12 ns; both stay overflow- and underflow-safe on the benchmark's vector scaled by 1e300 and by 1e-300. GEMV at 5000×5000 is the one GEMV size where OpenBLAS is ahead. The DOT, NRM2 and GEMV kernels behind these rows are aarch64 (NEON) only; x86_64 keeps the older kernels. SYRK computes only the micro-tiles of the referenced triangle, with a triangle-only blocked kernel on GEMM's packers and micro-kernel; TRSM packs diagonal blocks of order 128 with their reciprocals and updates the rest with in-place GEMMs. Rectangular GEMM sizes read m×k×n.

#### Level 3 routines against naive loops

OxiBLAS only, at n = 1024, best of 3 runs. Flop counts are n²k for SYRK, 2n²k for SYR2K, n³ for TRMM and TRSM, 2n³ for SYMM and 8n³ for the complex routines; these run the 3M method, which does about 6n³ real flops, so their GFLOPS are effective rates. Speedup is the best optimized rate over the best naive rate.

| Routine | Type | GFLOPS | Speedup |
|---------|------|--------|---------|
| SYMM | `f64` | 50.6 | 22.0× |
| SYRK | `f64` | 54.0 | 33.1× |
| SYR2K | `f64` | 53.3 | 35.3× |
| TRMM | `f64` | 50.8 | 25.9× |
| TRSM | `f64` | 50.5 | 24.1× |
| HEMM | `Complex64` | 44.6 | 5.9× |
| GEMM (3M) | `Complex64` | 51.4 | 7.6× |
| GEMM (3M) | `Complex32` | 98.7 | 13.2× |

#### LAPACK factorizations

Blocked against unblocked, `f64`, best of 3 runs. Flop counts are ⅔n³ for LU and ⅓n³ for Cholesky. The matrices of `lu_bench` are diagonally dominant, so partial pivoting never swaps rows.

| Factorization | n | Unblocked (GFLOPS) | Blocked (GFLOPS) | Speedup |
|---------------|---|--------------------|------------------|---------|
| LU | 256 | 9.27 | 20.51 | 2.2× |
| LU | 512 | 10.07 | 28.96 | 2.9× |
| LU | 768 | 10.25 | 34.04 | 3.3× |
| LU | 1024 | 10.70 | 37.91 | 3.5× |
| Cholesky | 256 | 11.55 | 29.25 | 2.5× |
| Cholesky | 512 | 10.84 | 36.37 | 3.4× |
| Cholesky | 768 | 11.40 | 41.33 | 3.6× |
| Cholesky | 1024 | 11.49 | 40.88 | 3.6× |

Unblocked LU (`Lu::compute`) eliminates column by column. Blocked LU works in place: it factors each panel recursively (`?getrf2`), solves the block row of U with TRSM and updates the trailing matrix with one GEMM. Blocked Cholesky updates the off-diagonal blocks of the lower triangle with GEMM and the diagonal blocks with the triangle-only SYRK. Both forms got faster in 0.2.3. Before that work (measured on 2026-11-02), unblocked LU ran at 0.38-0.62 GFLOPS and blocked LU at 3.8-15.9, unblocked Cholesky at 1.6-2.2 and blocked Cholesky at 14.0-19.1. The speedups are smaller than they were because the unblocked forms gained the most.

`Qr::compute_blocked` with block size 64 takes 10.2 ms at n = 512 and 58.3 ms at n = 1024, which is 3.7× and 5.3× faster than `Qr::compute` (38.0 ms and 306.4 ms).

On random matrices, `SvdDc` (the `dgesdd` scheme) is 13-32× faster than the one-sided Jacobi `Svd`, and `SymmetricEvdDc` (the `DSYEVD` scheme) 1.6-2.2× faster than `SymmetricEvd`, at every size measured from n = 200 to 2000 on one thread. `cargo run --release -p oxiblas --example evd_svd_bench` prints the times and accuracies; the times below are the median of 3 calls, best of 2 runs.

| n | `SymmetricEvd` | `SymmetricEvdDc` | `Svd` (Jacobi) | `SvdDc` |
|---|---|---|---|---|
| 200 | 3.5 ms | 2.2 ms | 51.5 ms | 4.0 ms |
| 500 | 40.7 ms | 21.2 ms | 0.91 s | 41.1 ms |
| 1000 | 297 ms | 138 ms | 7.77 s | 271 ms |
| 2000 | 2.09 s | 0.96 s | 68.8 s | 2.13 s |

The divide-and-conquer drivers are also the more accurate ones here: at n = 2000 the relative residual is 4.34e-15 for `SymmetricEvdDc` against 3.17e-14 for `SymmetricEvd`, and 5.30e-15 for `SvdDc` against 6.65e-13 for `Svd`.

### Linux x86_64

Not re-measured since 0.1.0: 2025-12-27, OxiBLAS only, on an Intel Xeon E5-2623 v4 @ 2.60GHz with AVX2 and FMA. These figures predate balanced blocking and the 2D parallel driver, and the OpenBLAS columns once published for this machine are dropped.

| n | f64 GEMM (GFLOPS) | f32 GEMM (GFLOPS) |
|---|-------------------|-------------------|
| 64 | 17.0 | 31.6 |
| 128 | 22.7 | 41.0 |
| 256 | 27.5 | 47.3 |
| 512 | 24.3 | 50.3 |
| 1024 | 26.6 | not recorded |

### Parallelization

Enable multi-core parallelization:

```toml
oxiblas = { version = "0.2", features = ["parallel"] }
```

`gemm()` itself stays sequential, with or without the feature. Pass `Par::Rayon` to use Rayon's thread pool:

```rust
use oxiblas_blas::level3::gemm_with_par;
use oxiblas_core::parallel::Par;
use oxiblas_matrix::Mat;

let n = 512;
let a = Mat::<f64>::filled(n, n, 1.0);
let b = Mat::<f64>::filled(n, n, 2.0);
let mut c = Mat::<f64>::zeros(n, n);

gemm_with_par(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut(), Par::Rayon);
```

GEMM goes parallel from 128³ multiply-adds, with at least 80³ per thread, so from n = 128 on 4 threads and n = 160 on 8; smaller problems run sequentially. Tile edges follow the micro-kernel grid, so the result is bitwise identical to `Par::Seq` for any thread count.

`Par::RayonWith(n)` partitions the work for `n` threads, but the tasks still run on the ambient Rayon pool. To bound the thread count, run the call inside `rayon::ThreadPool::install` or set `RAYON_NUM_THREADS`. Running on an `n`-thread pool is planned for 0.2.4.

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

| Platform | CPU | SIMD | f64 GEMM, one thread, n = 1024 | vs OpenBLAS |
|----------|-----|------|--------------------------------|-------------|
| macOS | Apple M3 | NEON 128-bit | 55.0 GFLOPS | 97% |
| Linux x86_64 | Intel Xeon E5-2623 v4 | AVX2 256-bit | 26.6 GFLOPS (0.1.0) | not measured |

With the `force-scalar` feature the M3 reaches 29.8 GFLOPS for f64 and 34.7 for f32, which is 54% and 30% of its NEON speed. That feature replaces the GEMM micro-kernels and, on aarch64, also keeps the generic `dot`, `nrm2` and `cnrm2`, `gemv`, the unit-stride `cblas_?gemv`, the TRSM/TRMM tile kernels and the GEMM B-panel packers on portable code; the typed `*_f64`/`*_f32` Level-1 kernels and the x86_64 Level-1 and `gemv` kernels do not consult it.

### SIMD Feature Detection

OxiBLAS automatically detects and uses the best available SIMD:
- **x86_64**: AVX-512 > AVX2 > SSE4.2 > scalar
- **aarch64**: NEON (native 128-bit) + emulated 256-bit
- **wasm32**: SIMD128 (if enabled) + emulated 256-bit
- **others**: Scalar fallback

Runtime detection via `SimdCapabilities::detect()`; with `std`, the result is cached after the first call. Level 1 and `gemv` kernels are chosen per architecture at compile time or by their own checks, and only GEMM goes through `KernelSelector`.

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
| `compute_blocked()` | Large matrices | Level 3 BLAS; picks the block size for Cholesky/LU, takes `nb` for QR |
| `compute_with_block_size(nb)` | Explicit block size control | Cholesky and LU |
| `compute_recursive()` | Very large matrices, cache-oblivious | No explicit block size; auto-adapts to cache |
| `compute_auto()` | **Recommended default** | Auto-selects blocked (n >= 128) or unblocked (n < 128) |
| `compute_blocked_par()` | Very large matrices on multi-core systems | Cholesky and LU; requires `parallel` feature; Rayon-based |

```rust
use oxiblas_lapack::{cholesky::Cholesky, lu::Lu, qr::Qr};
use oxiblas_matrix::Mat;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Symmetric and diagonally dominant, so also positive definite
    let n = 256;
    let mut a = Mat::<f64>::filled(n, n, 0.5);
    for i in 0..n {
        a[(i, i)] = n as f64;
    }

    // Recommended: automatic algorithm selection
    let chol = Cholesky::compute_auto(a.as_ref())?;
    let lu = Lu::compute_auto(a.as_ref())?;
    let qr = Qr::compute_auto(a.as_ref())?;

    // Explicit block size
    let chol = Cholesky::compute_with_block_size(a.as_ref(), 64)?;
    let lu = Lu::compute_with_block_size(a.as_ref(), 64)?;
    let qr = Qr::compute_blocked(a.as_ref(), 64)?;

    // Cache-oblivious recursive (no tuning required)
    let lu = Lu::compute_recursive(a.as_ref())?;

    // Parallel blocked (requires "parallel" feature)
    let chol = Cholesky::compute_blocked_par(a.as_ref())?;
    Ok(())
}
```

### Mixed-Precision Iterative Refinement

For applications requiring double-precision accuracy with single-precision speed:

```rust
use oxiblas_lapack::solve::{mixed_precision_solve, mixed_precision_solve_cholesky};
use oxiblas_matrix::Mat;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Mat::from_rows(&[&[4.0, 1.0], &[1.0, 3.0]]);
    let b = Mat::from_rows(&[&[1.0], &[2.0]]);

    // LU-based: factorize in f32, refine residuals in f64
    let result = mixed_precision_solve(a.as_ref(), b.as_ref())?;
    let x = result.solution;

    // Cholesky-based (symmetric positive definite systems)
    let result = mixed_precision_solve_cholesky(a.as_ref(), b.as_ref())?;
    let x = result.solution;
    Ok(())
}
```

### Batched BLAS Operations

For batches of independent small matrix operations (e.g., neural network layers):

```rust
use oxiblas_blas::level3::batched::{Transpose, gemm_batched};
use oxiblas_matrix::{Mat, MatMut, MatRef};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a_mats = vec![Mat::<f64>::filled(4, 4, 1.0); 8];
    let b_mats = vec![Mat::<f64>::filled(4, 4, 2.0); 8];
    let mut c_mats = vec![Mat::<f64>::zeros(4, 4); 8];

    // Batch of independent GEMMs: C[i] = alpha * A[i] * B[i] + beta * C[i]
    let a_refs: Vec<MatRef<'_, f64>> = a_mats.iter().map(Mat::as_ref).collect();
    let b_refs: Vec<MatRef<'_, f64>> = b_mats.iter().map(Mat::as_ref).collect();
    let mut c_muts: Vec<MatMut<'_, f64>> = c_mats.iter_mut().map(Mat::as_mut).collect();

    gemm_batched(
        Transpose::NoTrans,
        Transpose::NoTrans,
        1.0,
        &a_refs,
        &b_refs,
        0.0,
        &mut c_muts,
    )?;
    Ok(())
}
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `default` | `std` and `sparse`: f32, f64, Complex32, Complex64 and the sparse crate |
| `std` | Standard library support in the core and matrix crates |
| `parallel` | Enable rayon-based parallelization for multi-core |
| `f16` | Half-precision (16-bit) floating point support |
| `f128` | Quad precision via the double-double `QuadFloat` (about 31 digits for the arithmetic and `sqrt`) |
| `sparse` | Re-export `oxiblas-sparse` |
| `ndarray` | Re-export `oxiblas-ndarray` |
| `serde` | Serialization for matrix types |
| `mmap` | Memory-mapped matrices |
| `nalgebra` | Conversions to and from nalgebra types |
| `full` | All of the above (not the SIMD limits below) |
| `force-scalar` | Use the portable scalar GEMM kernel and report no SIMD level; on aarch64 also keep the generic DOT/NRM2, GEMV and TRSM/TRMM tile kernels on portable code |
| `max-simd-128` | Cap the GEMM kernel at 128-bit SIMD |
| `max-simd-256` | Cap the GEMM kernel at 256-bit SIMD |

**Crate-specific features:**
| Feature | Crate | Description |
|---------|-------|-------------|
| `runtime-tuning` | `oxiblas-blas` | Runtime cache-parameter tuning |
| `sve-nightly` | `oxiblas-core` | SVE length queries and stand-alone dot/axpy kernels, not used by the BLAS routines; with `-C target-feature=+sve` it enables the `stdarch_aarch64_sve` feature gate and needs a nightly toolchain (compile-checked, not run on SVE hardware) |
| `compare-openblas` | `oxiblas-benchmarks` | Enable OpenBLAS comparison benchmarks |

## Requirements

- Rust 1.89+ (Edition 2024)
- No C, C++ or Fortran in the normal or default dependencies of the libraries. Two dev-dependencies do compile C or assembly when the benches and tests are built: `criterion` (through `alloca`) and, in `oxiblas-ndarray`, `faer` (through `psm` and `generator`). The optional `compare-openblas` bench loads a system OpenBLAS at run time

## Project Status

**Version:** 0.2.4 (2026-11-12)

- **Lines of Code:** ~283,000 lines of Rust, of which 228,258 are code (467 files, excluding the retired `oxiblas-ffi` crate; `tokei`)
- **Documentation:** 13 examples
- **Tests:** 4,437 passing lib/integration tests and 4 release-only ones skipped in debug builds (`cargo nextest run --workspace --exclude oxiblas-benchmarks`) + 358 passing doctests (`cargo test --doc --workspace --exclude oxiblas-benchmarks`)
- **Coverage:**
  - ✅ Full BLAS Level 1/2/3 (including packed/banded variants)
  - ✅ CBLAS interface (every routine in netlib `cblas.h` 3.12.1 except `cblas_xerbla`)
  - ✅ Extensive LAPACK (LU, Cholesky, QR, SVD, EVD, Schur, Hessenberg)
  - ✅ Generalized Schur decomposition (`Qz`, equivalent to `DGGES`, with reordering)
  - ✅ Recursive cache-oblivious factorizations (Cholesky, LU, QR)
  - ✅ Parallel blocked factorizations (Cholesky, LU)
  - ✅ Fixed blocked QR with WY representation (3.7× faster than unblocked at n = 512, 5.3× at n = 1024)
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
  - ✅ Tensor operations (einsum with 22 patterns, batched matmul)
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

## Sponsorship

OxiBLAS is developed and maintained by **COOLJAPAN OU (Team Kitasan)**.

If you find OxiBLAS useful, please consider sponsoring the project to support continued development of the Pure Rust ecosystem.

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-red?logo=github)](https://github.com/sponsors/cool-japan)

**[https://github.com/sponsors/cool-japan](https://github.com/sponsors/cool-japan)**

Your sponsorship helps us:
- Maintain and improve the COOLJAPAN ecosystem
- Keep the entire ecosystem (OxiBLAS, OxiFFT, SciRS2, etc.) 100% Pure Rust
- Provide long-term support and security updates

## License

Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).

See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) before submitting issues and pull requests.
