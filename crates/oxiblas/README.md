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
- **High performance** - 80-172% of OpenBLAS depending on operation
- **Pure Rust** - No C dependencies, easy cross-compilation

## Quick Start

```toml
[dependencies]
oxiblas = "0.1"

# With all features
oxiblas = { version = "0.1", features = ["full"] }
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

### Linear Algebra

```rust
use oxiblas::prelude::*;

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

// Solve Ax = b
let b = vec![4.0, 10.0, 24.0];
let x = lu.solve(&b)?;
```

### Sparse Matrices

```rust
use oxiblas::prelude::*;

// Create sparse matrix
let mut coo = CooMatrix::<f64>::new(1000, 1000);
coo.push(0, 0, 4.0);
coo.push(0, 1, -1.0);
// ... add more elements

let csr = CsrMatrix::from_coo(&coo);

// Solve sparse system with GMRES
let b = vec![/*...*/];
let result = gmres(&csr, &b, 1e-10, 100, 30, None)?;
```

### Tensor Operations

```rust
use oxiblas::prelude::*;

// Einstein summation
let c = einsum("ij,jk->ik", &a, &[m, n], Some((&b, &[n, k])))?;

// Batched matrix multiplication
let a_batch = Tensor3::from_data(data_a, batch, m, k);
let b_batch = Tensor3::from_data(data_b, batch, k, n);
let c_batch = batched_matmul(&a_batch, &b_batch)?;
```

## Module Structure

The `oxiblas` crate re-exports from these sub-crates:

| Module | Re-exported from | Description |
|--------|------------------|-------------|
| `oxiblas::core` | `oxiblas-core` | Core traits, SIMD, scalar types |
| `oxiblas::matrix` | `oxiblas-matrix` | Matrix types and views |
| `oxiblas::blas` | `oxiblas-blas` | BLAS operations |
| `oxiblas::lapack` | `oxiblas-lapack` | LAPACK decompositions |
| `oxiblas::sparse` | `oxiblas-sparse` | Sparse matrices and solvers |
| `oxiblas::ndarray` | `oxiblas-ndarray` | ndarray integration (optional) |
| | `oxiblas-ffi` | **RETIRED** (v0.2.0) - C FFI bindings |

## Prelude

The `oxiblas::prelude` module provides convenient imports:

```rust
use oxiblas::prelude::*;

// Now you have access to:
// - Mat, MatRef, MatMut (matrix types)
// - gemm, gemv, dot, axpy, etc. (BLAS operations)
// - Lu, Qr, Svd, Cholesky, etc. (LAPACK decompositions)
// - CsrMatrix, CooMatrix, etc. (sparse matrices)
// - gmres, cg, bicgstab, etc. (sparse solvers)
```

## Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `default` | Core functionality (f32, f64, complex) | ✓ |
| `parallel` | Rayon-based parallelization | |
| `f16` | Half-precision (16-bit) floating point | |
| `f128` | Quad-precision (~31 digits) | |
| `sparse` | Sparse matrix operations | ✓ |
| `ndarray` | ndarray integration | |
| `ffi` | C FFI bindings | |
| `full` | All features enabled | |
| `nightly` | Nightly-only optimizations | |

### Examples

```toml
# Minimal (dense matrices only)
oxiblas = "0.1"

# With parallelization
oxiblas = { version = "0.1", features = ["parallel"] }

# With extended precision
oxiblas = { version = "0.1", features = ["f16", "f128"] }

# With ndarray support
oxiblas = { version = "0.1", features = ["ndarray"] }

# All features
oxiblas = { version = "0.1", features = ["full"] }
```

## Examples

The repository includes comprehensive examples:

```bash
# Basic BLAS operations
cargo run --example basic_blas

# LAPACK decompositions
cargo run --example lapack_decompositions

# Extended precision
cargo run --example extended_precision --features f128

# Tensor operations
cargo run --example tensor_operations

# Sparse matrices
cargo run --example sparse_matrices --features parallel
```

## Performance

OxiBLAS provides competitive performance with industry-standard libraries:

### macOS M3 (Apple Silicon)

| Operation | OxiBLAS | OpenBLAS | Ratio |
|-----------|---------|----------|-------|
| DGEMM 1024×1024 | 40.25 ms | 40.54 ms | **101%** |
| SGEMM 1024×1024 | 19.18 ms | 32.94 ms | **172%** |
| DOT 1M elements | 167 µs | 279 µs | **165%** |

### Linux x86_64 (Intel Xeon)

| Operation | OxiBLAS | OpenBLAS | Ratio |
|-----------|---------|----------|-------|
| DGEMM 1024×1024 | 80.68 ms | 82.51 ms | **102%** |
| SGEMM 64×64 | 16.60 µs | 18.64 µs | **112%** |
| DGEMM 256×256 | 1.220 ms | 1.159 ms | 95% |

**Summary**: OxiBLAS achieves **80-172% of OpenBLAS performance** across different platforms and operations.

## Documentation

- **[Main README](../../README.md)** - Project overview and features
- **[API Documentation](https://docs.rs/oxiblas)** - Complete API reference
- **[Examples](../../examples/)** - Usage examples
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

- **Rust**: 1.85+ (Edition 2024)
- **No external C dependencies**
- **Supported platforms**: x86_64, AArch64 (Linux, macOS, Windows)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## Citation

If you use OxiBLAS in your research, please cite:

```bibtex
@software{oxiblas2025,
  author = {OxiBLAS Contributors},
  title = {OxiBLAS: Pure Rust BLAS/LAPACK Implementation},
  year = {2025},
  url = {https://github.com/cool-japan/oxiblas}
}
```
