# oxiblas-blas

**BLAS (Basic Linear Algebra Subprograms) operations for OxiBLAS**

[![Crates.io](https://img.shields.io/crates/v/oxiblas-blas.svg)](https://crates.io/crates/oxiblas-blas)
[![Documentation](https://docs.rs/oxiblas-blas/badge.svg)](https://docs.rs/oxiblas-blas)

## Overview

`oxiblas-blas` implements the complete BLAS API in pure Rust with SIMD optimization and optional parallelization. It provides Level 1 (vector-vector), Level 2 (matrix-vector), and Level 3 (matrix-matrix) operations with performance competitive with OpenBLAS.

## Features

### Complete BLAS Implementation

- **Level 1**: 15+ vector-vector operations
- **Level 2**: 19+ matrix-vector operations (including banded and packed variants)
- **Level 3**: 9+ matrix-matrix operations
- **Extended Operations**: Tensor operations, Einstein summation, batched operations

### High Performance

- **SIMD-optimized** kernels for x86_64 (AVX2/AVX-512) and AArch64 (NEON)
- **Cache-aware blocking** for optimal L1/L2/L3 utilization
- **Parallel execution** via Rayon for large matrices (with `parallel` feature)
- **Competitive with OpenBLAS**: 80-172% performance depending on operation and platform

## Installation

```toml
[dependencies]
oxiblas-blas = "0.1"

# With parallelization
oxiblas-blas = { version = "0.1", features = ["parallel"] }
```

## BLAS Level 1 (Vector-Vector)

| Function | Description | Complexity |
|----------|-------------|-----------|
| `dot` | Dot product: x · y | O(n) |
| `axpy` | y = α×x + y | O(n) |
| `scal` | x = α×x | O(n) |
| `copy` | y = x | O(n) |
| `swap` | Swap x and y | O(n) |
| `nrm2` | Euclidean norm: ‖x‖₂ | O(n) |
| `asum` | Sum of absolute values: Σ\|xᵢ\| | O(n) |
| `iamax` | Index of max absolute value | O(n) |
| `rot` | Apply Givens rotation | O(n) |
| `rotg` | Generate Givens rotation | O(1) |
| `rotm` | Apply modified Givens rotation | O(n) |
| `rotmg` | Generate modified Givens rotation | O(1) |

### Usage Example

```rust
use oxiblas_blas::level1::{dot, axpy, nrm2};

let x = vec![1.0, 2.0, 3.0, 4.0];
let y = vec![5.0, 6.0, 7.0, 8.0];

// Dot product
let result = dot(&x, &y);  // 70.0

// AXPY: y = 2.5*x + y
let mut y = vec![1.0, 2.0, 3.0, 4.0];
axpy(2.5, &x, &mut y);
// y = [3.5, 7.0, 10.5, 14.0]

// Euclidean norm
let norm = nrm2(&x);  // sqrt(30) ≈ 5.477
```

## BLAS Level 2 (Matrix-Vector)

| Function | Description | Complexity |
|----------|-------------|-----------|
| `gemv` | General matrix-vector: y = α×A×x + β×y | O(mn) |
| `symv` | Symmetric matrix-vector | O(n²) |
| `hemv` | Hermitian matrix-vector (complex) | O(n²) |
| `trmv` | Triangular matrix-vector | O(n²) |
| `trsv` | Triangular solve: x = A⁻¹×b | O(n²) |
| `ger` | Rank-1 update: A = α×x×yᵀ + A | O(mn) |
| `syr` | Symmetric rank-1: A = α×x×xᵀ + A | O(n²) |
| `her` | Hermitian rank-1 (complex) | O(n²) |
| `syr2` | Symmetric rank-2 | O(n²) |
| `her2` | Hermitian rank-2 (complex) | O(n²) |

**Banded & Packed variants**: `gbmv`, `sbmv`, `hbmv`, `tbmv`, `tbsv`, `spmv`, `hpmv`, `tpmv`, `tpsv`

### Usage Example

```rust
use oxiblas_blas::level2::{gemv, GemvTrans};
use oxiblas_matrix::Mat;

let a = Mat::from_rows(&[
    &[1.0, 2.0, 3.0],
    &[4.0, 5.0, 6.0],
]);
let x = vec![1.0, 2.0, 3.0];
let mut y = vec![0.0, 0.0];

// y = A × x
gemv(GemvTrans::NoTrans, 1.0, a.as_ref(), &x, 0.0, &mut y);
// y = [14.0, 32.0]
```

## BLAS Level 3 (Matrix-Matrix)

| Function | Description | Complexity | Performance |
|----------|-------------|-----------|-------------|
| `gemm` | General matrix multiply: C = α×A×B + β×C | O(n³) | **79-172% of OpenBLAS** |
| `symm` | Symmetric matrix multiply | O(n³) | Optimized |
| `hemm` | Hermitian matrix multiply (complex) | O(n³) | Optimized |
| `trmm` | Triangular matrix multiply | O(n³) | **7-11× vs naive** |
| `trsm` | Triangular solve multiple RHS | O(n³) | **10× vs naive** |
| `syrk` | Symmetric rank-k: C = α×A×Aᵀ + β×C | O(n²k) | **6-15× vs naive** |
| `herk` | Hermitian rank-k (complex) | O(n²k) | **6-12× vs naive** |
| `syr2k` | Symmetric rank-2k | O(n²k) | **6-15× vs naive** |
| `her2k` | Hermitian rank-2k (complex) | O(n²k) | **6-15× vs naive** |

### GEMM Example

```rust
use oxiblas_blas::level3::gemm;
use oxiblas_matrix::Mat;

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

// C = A × B
gemm(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut());
// C = [[58, 64], [139, 154]]
```

## Tensor Operations

### Einstein Summation

Supports 24 tensor contraction patterns:

```rust
use oxiblas_blas::tensor::einsum;

// Matrix multiplication: C = A × B
let c = einsum("ij,jk->ik", &a, &[m, n], Some((&b, &[n, k])))?;

// Outer product: C = x ⊗ y
let c = einsum("i,j->ij", &x, &[m], Some((&y, &[n])))?;

// Transpose: Aᵀ
let at = einsum("ij->ji", &a, &[m, n], None)?;

// Matrix trace: tr(A)
let trace = einsum("ii->", &a, &[n, n], None)?;

// Hadamard (element-wise) product: C = A ⊙ B
let c = einsum("ij,ij->ij", &a, &[m, n], Some((&b, &[m, n])))?;
```

**Supported patterns**: matmul, transpose, trace, diagonal, tensor transposes, reductions, products, batched operations

### Batched Operations

```rust
use oxiblas_blas::tensor::{batched_matmul, Tensor3};

// Batch of matrix multiplications
let a = Tensor3::from_data(data_a, batch_size, m, k);
let b = Tensor3::from_data(data_b, batch_size, k, n);

let c = batched_matmul(&a, &b)?;
// c[i] = a[i] × b[i] for each batch i
```

## Extended Precision

```rust
use oxiblas_blas::level1::{dot_kahan, dot_pairwise, dsdot};

// Kahan (compensated) summation for accuracy
let result_kahan = dot_kahan(&x, &y);

// Pairwise summation (divide-and-conquer)
let result_pairwise = dot_pairwise(&x, &y);

// Mixed precision: f32 computation, f64 accumulation
let x_f32: Vec<f32> = vec![/*...*/];
let y_f32: Vec<f32> = vec![/*...*/];
let result_f64 = dsdot(&x_f32, &y_f32);
```

## Performance

### macOS M3 (Apple Silicon NEON)

| Operation | Size | OxiBLAS | OpenBLAS | Ratio |
|-----------|------|---------|----------|-------|
| **DGEMM (f64)** | 1024×1024 | 40.25 ms (427 Gf/s) | 40.54 ms (424 Gf/s) | **101%** 🟢 |
| **SGEMM (f32)** | 1024×1024 | 19.18 ms (896 Gf/s) | 32.94 ms (521 Gf/s) | **172%** 🟢 |
| **DOT (f64)** | 1M | 167 µs (6.0 Gelem/s) | 279 µs (3.6 Gelem/s) | **165%** 🟢 |

### Linux x86_64 (Intel Xeon AVX2)

| Operation | Size | OxiBLAS | OpenBLAS | Ratio |
|-----------|------|---------|----------|-------|
| **DGEMM (f64)** | 1024×1024 | 80.68 ms (213 Gf/s) | 82.51 ms (208 Gf/s) | **102%** 🟢 |
| **SGEMM (f32)** | 64×64 | 16.60 µs (253 Gf/s) | 18.64 µs (225 Gf/s) | **112%** 🟢 |
| **DGEMM (f64)** | 256×256 | 1.220 ms (220 Gf/s) | 1.159 ms (232 Gf/s) | 95% 🟢 |

## Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `default` | Core BLAS operations | ✓ |
| `parallel` | Rayon-based parallelization | |

## Architecture

```
oxiblas-blas/
├── level1/            # Vector-vector operations
│   ├── dot.rs
│   ├── axpy.rs
│   ├── scal.rs
│   └── ...
├── level2/            # Matrix-vector operations
│   ├── gemv.rs
│   ├── ger.rs
│   ├── symv.rs
│   └── ...
├── level3/            # Matrix-matrix operations
│   ├── gemm.rs        # General matrix multiply
│   ├── gemm_kernel.rs # SIMD micro-kernels
│   ├── autotune.rs    # Cache-aware blocking
│   ├── syrk.rs
│   ├── trmm.rs
│   └── ...
└── tensor/            # Tensor operations
    ├── einsum.rs      # Einstein summation
    ├── batched.rs     # Batched operations
    └── tensor3.rs     # 3D tensor type
```

## Examples

```bash
# Run BLAS examples
cargo run --example basic_blas
cargo run --example tensor_operations
```

## Benchmarks

```bash
# Level 1 benchmarks
cargo bench --package oxiblas-benchmarks --bench blas_level1

# Level 2 benchmarks
cargo bench --package oxiblas-benchmarks --bench blas_level2

# Level 3 benchmarks
cargo bench --package oxiblas-benchmarks --bench blas_level3

# Compare with OpenBLAS
cargo bench --package oxiblas-benchmarks --bench comparison --features compare-openblas
```

## Related Crates

- [`oxiblas-core`](../oxiblas-core/) - Core traits and SIMD
- [`oxiblas-matrix`](../oxiblas-matrix/) - Matrix types
- [`oxiblas-lapack`](../oxiblas-lapack/) - LAPACK decompositions
- [`oxiblas`](../oxiblas/) - Meta-crate

## License

Licensed under MIT or Apache-2.0 at your option.
