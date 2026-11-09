# oxiblas

**Unified API for OxiBLAS - Pure Rust BLAS/LAPACK implementation**

[![Crates.io](https://img.shields.io/crates/v/oxiblas.svg)](https://crates.io/crates/oxiblas)
[![Documentation](https://docs.rs/oxiblas/badge.svg)](https://docs.rs/oxiblas)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../../LICENSE)

## Overview

`oxiblas` is the meta-crate that re-exports all OxiBLAS functionality through a unified, convenient API. This is the recommended way to use OxiBLAS for most users.

## Features

- **Complete BLAS** - Level 1, 2, 3 operations
- **Extensive LAPACK** - LU, QR, SVD, Cholesky, EVD, and more
- **Sparse matrices** - 9 formats with iterative solvers
- **Tensor operations** - Einstein summation, batched operations
- **Extended precision** - f16, f128 support
- **High performance** - BLIS-style blocked algorithms and SIMD kernels; see [Performance](#performance)
- **Pure Rust** - No C, C++ or Fortran in the libraries or their normal dependencies, easy cross-compilation

## Quick Start

```toml
[dependencies]
oxiblas = "0.2"

# With all features
oxiblas = { version = "0.2", features = ["full"] }
```

## Usage

### Matrix Operations

```rust
use oxiblas::prelude::*;

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

// Matrix multiplication
let mut c = Mat::zeros(2, 2);
gemm(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());
// c = [[58, 64], [139, 154]]
```

### Builders and Lazy Evaluation

```rust
use oxiblas::prelude::*;

let a = MatBuilder::<f64>::identity(3);
let b = MatBuilder::<f64>::ones(3, 3);

// Expressions are only evaluated when `.eval()` is called
let c = (a.as_ref().lazy() + b.as_ref().lazy()).scale(2.0).eval();
```

### Parallel GEMM

With the `parallel` feature, the Rayon path is opt-in per call. Plain `gemm` always runs sequentially; pass `Par::Rayon` to `gemm_with_par` to parallelize:

```rust
use oxiblas::prelude::*;

let a = Mat::<f64>::filled(256, 256, 1.0);
let b = Mat::<f64>::filled(256, 256, 2.0);
let mut c = Mat::<f64>::zeros(256, 256);

// Requires the `parallel` feature
gemm_with_par(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut(), Par::Rayon);
```

### Linear Algebra

```rust
use oxiblas::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Mat::from_rows(&[
        &[2.0, 1.0, 1.0],
        &[4.0, 3.0, 3.0],
        &[8.0, 7.0, 9.0],
    ]);

    // LU decomposition
    let lu = Lu::compute(a.as_ref())?;
    let det = lu.determinant();
    let inv = lu.inverse()?;

    // QR decomposition
    let qr = Qr::compute(a.as_ref())?;
    let q = qr.q();
    let r = qr.r();

    // SVD
    let svd = Svd::compute(a.as_ref())?;
    let singular_values = svd.singular_values();

    // Solve Ax = b (b is an n x 1 matrix, i.e. a column vector)
    let b = Mat::from_rows(&[&[4.0], &[10.0], &[24.0]]);
    let x = lu.solve(b.as_ref())?;
    Ok(())
}
```

### Sparse Matrices

```rust
use oxiblas::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 1000 x 1000 sparse matrix (tridiagonal)
    let n = 1000;
    let mut coo = CooMatrix::<f64>::new_empty(n, n);
    for i in 0..n {
        coo.push(i, i, 4.0);
        if i + 1 < n {
            coo.push(i, i + 1, -1.0);
            coo.push(i + 1, i, -1.0);
        }
    }

    let csr = coo.to_csr();

    // Solve sparse system with GMRES
    let b = vec![1.0; n];
    let x0 = vec![0.0; n]; // initial guess
    let result = gmres(&csr, &b, &x0, 30, 1e-10, 100)?;
    let x = result.x;
    Ok(())
}
```

### Tensor Operations

```rust
// Tensor operations live in `oxiblas-blas`'s `tensor` module and are not
// part of `oxiblas::prelude` - import them from `oxiblas::blas::tensor`.
use oxiblas::blas::tensor::{Tensor3, batched_matmul, einsum};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Einstein summation: matrix multiplication (A: 2x3, B: 3x2)
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let c = einsum("ij,jk->ik", &a, &[2, 3], Some((&b, &[3, 2])))?;

    // Batched matrix multiplication
    let (batch, m, k, n) = (4, 2, 3, 2);
    let data_a = vec![1.0; batch * m * k];
    let data_b = vec![1.0; batch * k * n];
    let a_batch = Tensor3::from_data(data_a, batch, m, k);
    let b_batch = Tensor3::from_data(data_b, batch, k, n);
    let c_batch = batched_matmul(&a_batch, &b_batch)?;
    Ok(())
}
```

## Module Structure

The `oxiblas` crate re-exports from these sub-crates:

| Module | Re-exported from | Description |
|--------|------------------|-------------|
| `oxiblas::core` | `oxiblas-core` | Core traits, SIMD, scalar types |
| `oxiblas::matrix` | `oxiblas-matrix` | Matrix types and views |
| `oxiblas::blas` | `oxiblas-blas` | BLAS operations |
| `oxiblas::lapack` | `oxiblas-lapack` | LAPACK decompositions |
| `oxiblas::sparse` | `oxiblas-sparse` | Sparse matrices and solvers (`sparse` feature, on by default) |
| `oxiblas::ndarray` | `oxiblas-ndarray` | ndarray integration (`ndarray` feature) |
| | `oxiblas-ffi` | **RETIRED** (v0.2.0) - C FFI bindings |

The crate also defines `oxiblas::builder` (`MatBuilder`), `oxiblas::fluent` (method-chaining traits such as `MatrixOps`), `oxiblas::auto` (size- and structure-based algorithm selection, e.g. `auto_matmul`) and `oxiblas::features` (compile-time constants such as `HAS_PARALLEL`).

## Prelude

The `oxiblas::prelude` module provides convenient imports:

```rust
use oxiblas::prelude::*;

// Now you have access to:
// - Mat, MatRef, MatMut (matrix types)
// - gemm, gemv, dot, axpy, etc. (BLAS operations)
// - Lu, Qr, Svd, Cholesky, etc. (LAPACK decompositions)
// - solve, inv, det, lstsq, etc. (LAPACK solvers and utilities)
// - CsrMatrix, CooMatrix, etc. (sparse matrices)
// - gmres, cg, pcg, bicgstab (sparse solvers; the others are in oxiblas::sparse::linalg)
```

## Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `default` | `std` and `sparse` (f32, f64 and complex are always available) | ✓ |
| `std` | Standard library support (forwarded to `oxiblas-core` and `oxiblas-matrix`) | ✓ |
| `parallel` | Rayon-based parallelization, opt-in per call with `Par::Rayon` (see [Parallel GEMM](#parallel-gemm)); implies `std` | |
| `f16` | Half-precision (16-bit) floating point | |
| `f128` | Quad precision with the double-double `QuadFloat`: about 31 digits for the arithmetic and `sqrt`, fewer for `exp`, `ln` and the other elementary functions | |
| `sparse` | Sparse matrix operations | ✓ |
| `ndarray` | ndarray integration | |
| `serde` | Serialization support for matrix types | |
| `mmap` | Memory-mapped matrices for large datasets | |
| `nalgebra` | nalgebra type conversions | |
| `full` | `parallel`, `f16`, `f128`, `ndarray`, `sparse`, `serde`, `mmap` and `nalgebra` (not the SIMD flags below) | |
| `force-scalar` | Use the portable scalar GEMM kernel and report no SIMD level; on aarch64 also the portable DOT, NRM2, GEMV and TRSM/TRMM code | |
| `max-simd-128` | Cap the GEMM kernel at 128-bit SIMD (SSE4.2/NEON) | |
| `max-simd-256` | Cap the GEMM kernel at 256-bit SIMD (AVX2) | |

The three SIMD flags are forwarded to `oxiblas-core`, `oxiblas-blas` and `oxiblas-lapack`. They take effect in the capability detection and in GEMM kernel selection. On aarch64, `force-scalar` also keeps the generic `dot`, `nrm2` and `cnrm2`, `gemv`, the unit-stride `cblas_?gemv` and the TRSM/TRMM tile kernels on portable code; the typed `*_f64`/`*_f32` Level-1 kernels and the x86_64 Level-1 and `gemv` kernels pick AVX2 or NEON on their own and ignore the flags.

> Note: `oxiblas-ffi` (C FFI bindings) was **retired** in v0.2.0 and is no longer
> a feature of this crate - see the [Module Structure](#module-structure) table above.

### Examples

```toml
# Minimal (dense matrices only, no sparse support)
oxiblas = { version = "0.2", default-features = false, features = ["std"] }

# With parallelization
oxiblas = { version = "0.2", features = ["parallel"] }

# With extended precision
oxiblas = { version = "0.2", features = ["f16", "f128"] }

# With ndarray support
oxiblas = { version = "0.2", features = ["ndarray"] }

# All features
oxiblas = { version = "0.2", features = ["full"] }
```

## Examples

The repository includes comprehensive examples (in `crates/oxiblas/examples/`):

```bash
# Basic BLAS operations
cargo run -p oxiblas --example basic_blas

# LAPACK decompositions
cargo run -p oxiblas --example lapack_decompositions

# Extended precision
cargo run -p oxiblas --example extended_precision --features f128

# Tensor operations
cargo run -p oxiblas --example tensor_operations

# Sparse matrices
cargo run -p oxiblas --example sparse_matrices
```

The timing examples (`lu_bench`, `cholesky_bench`, `trmm_trsm_bench`, `syrk_syr2k_bench`, `symm_hemm_bench`, `complex_gemm_bench`, `evd_svd_bench`, and `parallel_bench` with `--features parallel`) are meant to run with `--release`, for example `cargo run --release -p oxiblas --example evd_svd_bench -- --reps 3 500 1000`.

## Performance

OxiBLAS uses BLIS-style blocked algorithms and hand-tuned SIMD micro-kernels. On an
Apple M3 on one thread, f64 GEMM runs at 92-99% of OpenBLAS and f32 GEMM at 101-120%;
`Par::Rayon` GEMM is level with OpenBLAS on 4 threads (95-103%) and ahead on 8
(115-169%). DOT, NRM2, GEMV, TRSM and SYRK were reworked in 0.2.3 (NEON Level-1/2
kernels on aarch64, a blocked TRSM, a triangle-only SYRK) and now run at 99% of
OpenBLAS or faster on one thread, except GEMV at 5000×5000 (97%). See
[Performance](../../README.md#performance) in the root README for the figures.

Actual performance depends heavily on operation, matrix size, and hardware (SIMD width,
cache hierarchy, core count), so no single ratio is representative. Run the benchmark
suite yourself for numbers reproducible on your own hardware:

```bash
# OxiBLAS-only benchmarks
cargo bench --package oxiblas-benchmarks

# Direct comparison against OpenBLAS (requires OpenBLAS installed)
cargo bench --package oxiblas-benchmarks --bench comparison --features compare-openblas
```

See the [oxiblas-benchmarks README](../oxiblas-benchmarks/README.md) for details on
the available benchmark suites.

## Documentation

- **[Main README](../../README.md)** - Project overview and features
- **[API Documentation](https://docs.rs/oxiblas)** - Complete API reference
- **[Examples](examples/)** - Usage examples
- **[Benchmarks README](../oxiblas-benchmarks/README.md)** - Performance benchmarking guide

## Sub-Crate Documentation

For more detailed documentation on specific components:

- **[oxiblas-core](../oxiblas-core/README.md)** - Core traits and SIMD
- **[oxiblas-matrix](../oxiblas-matrix/README.md)** - Matrix types
- **[oxiblas-blas](../oxiblas-blas/README.md)** - BLAS operations
- **[oxiblas-lapack](../oxiblas-lapack/README.md)** - LAPACK decompositions
- **[oxiblas-sparse](../oxiblas-sparse/README.md)** - Sparse matrices
- **[oxiblas-ndarray](../oxiblas-ndarray/README.md)** - ndarray integration
- **[oxiblas-benchmarks](../oxiblas-benchmarks/README.md)** - Benchmarking suite

## Ecosystem

OxiBLAS is part of the SciRS2 scientific computing ecosystem:

- [SciRS2](https://github.com/cool-japan/scirs) - Scientific computing library
- [NumRS2](https://github.com/cool-japan/numrs) - Numerical computing
- [SkleaRS](https://github.com/cool-japan/sklears) - Machine learning
- [ToRSh](https://github.com/cool-japan/torsh) - Tensor operations
- [TrustformeRS](https://github.com/cool-japan/trustformers) - Transformers
- [QuantRS2](https://github.com/cool-japan/quantrs) - Quantum computing framework
- [OxiRS](https://github.com/cool-japan/oxirs) - Semantic Web platform

## Requirements

- **Rust**: 1.89+ (Edition 2024)
- **No C, C++ or Fortran in the normal or default dependencies**; only the `criterion` (through `alloca`) and `faer` (through `psm` and `generator`) dev-dependencies of the benches and tests compile C or assembly
- **Supported platforms**: x86_64, AArch64 (Linux, macOS, Windows)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## Citation

If you use OxiBLAS in your research, please cite:

```bibtex
@software{oxiblas2026,
  author = {OxiBLAS Contributors},
  title = {OxiBLAS: Pure Rust BLAS/LAPACK Implementation},
  year = {2026},
  url = {https://github.com/cool-japan/oxiblas}
}
```
