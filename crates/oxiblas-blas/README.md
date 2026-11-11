# oxiblas-blas

**BLAS (Basic Linear Algebra Subprograms) operations for OxiBLAS**

[![Crates.io](https://img.shields.io/crates/v/oxiblas-blas.svg)](https://crates.io/crates/oxiblas-blas)
[![Documentation](https://docs.rs/oxiblas-blas/badge.svg)](https://docs.rs/oxiblas-blas)

## Overview

`oxiblas-blas` implements the complete BLAS API in pure Rust with SIMD optimization and optional parallelization. It provides Level 1 (vector-vector), Level 2 (matrix-vector), and Level 3 (matrix-matrix) operations with performance competitive with OpenBLAS. It also exports a CBLAS-compatible C interface (`oxiblas_blas::cblas`, see [CBLAS Interface](#cblas-interface)).

## Features

### Complete BLAS Implementation

- **Level 1**: 15+ vector-vector operations (real and complex, with strided variants)
- **Level 2**: 24 matrix-vector operations (including banded and packed variants)
- **Level 3**: 9 matrix-matrix operations, plus `gemmt` and a 3M complex GEMM
- **CBLAS**: 152 `extern "C"` entry points, every routine of netlib `cblas.h` 3.12.1 except `cblas_xerbla`
- **Extended Operations**: Tensor operations, Einstein summation, batched operations

### High Performance

- **SIMD-optimized** kernels for x86_64 (AVX2/AVX-512) and AArch64 (NEON)
- **Cache-aware blocking** for optimal L1/L2/L3 utilization
- **Parallel execution** via Rayon for large matrices (with `parallel` feature); opt in per call, for example `gemm_with_par(.., Par::Rayon)`, because plain `gemm` is always sequential
- **Close to OpenBLAS or ahead of it** on one thread on an Apple M3: f64 GEMM at 92-99% and f32 GEMM at 101-120% of OpenBLAS; [Performance](#performance) has the other routines

## Installation

```toml
[dependencies]
oxiblas-blas = "0.2"

# With parallelization
oxiblas-blas = { version = "0.2", features = ["parallel"] }
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
| `iamin` | Index of min absolute value | O(n) |
| `rot` | Apply Givens rotation | O(n) |
| `rotg` | Generate Givens rotation | O(1) |
| `rotm` | Apply modified Givens rotation | O(n) |
| `rotmg` | Generate modified Givens rotation | O(1) |

**Complex, mixed-precision, strided and parallel variants** (all in `oxiblas_blas::level1`):
- Complex: `dotc`, `dotu_c32`/`dotu_c64`, `casum`/`scasum`/`dzasum`, `cnrm2`/`scnrm2`/`dznrm2`, `csscal`/`zdscal`/`scal_real`, `crotg`/`zdrot`, `cabs1`/`scabs1`/`dcabs1`
- Mixed precision: `dsdot`, `sdsdot`
- Strided views (`StridedSlice`, `StridedSliceMut`) with Netlib increment semantics, negative and zero increments included: `dot_strided`, `axpy_strided`, `nrm2_strided`, `rotm_strided`, and so on
- Parallel (`Par` argument): `dot_par`, `axpy_par`, `nrm2_par`, `asum_par`, `scal_par`

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
| `gerc` | Conjugated rank-1 update (complex): A = α×x×yᴴ + A | O(mn) |
| `syr` | Symmetric rank-1: A = α×x×xᵀ + A | O(n²) |
| `her` | Hermitian rank-1 (complex) | O(n²) |
| `syr2` | Symmetric rank-2 | O(n²) |
| `her2` | Hermitian rank-2 (complex) | O(n²) |

**Banded & Packed variants**: `gbmv`, `sbmv`, `hbmv`, `tbmv`, `tbsv`, `spmv`, `hpmv`, `tpmv`, `tpsv`, and the packed rank updates `spr`, `spr2`, `hpr`, `hpr2`

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

| Function | Description | Complexity |
|----------|-------------|-----------|
| `gemm` | General matrix multiply: C = α×A×B + β×C | O(n³) |
| `symm` | Symmetric matrix multiply | O(n³) |
| `hemm` | Hermitian matrix multiply (complex) | O(n³) |
| `trmm` | Triangular matrix multiply | O(n³) |
| `trsm` | Triangular solve multiple RHS | O(n³) |
| `syrk` | Symmetric rank-k: C = α×A×Aᵀ + β×C | O(n²k) |
| `herk` | Hermitian rank-k (complex) | O(n²k) |
| `syr2k` | Symmetric rank-2k | O(n²k) |
| `her2k` | Hermitian rank-2k (complex) | O(n²k) |

**Also in `oxiblas_blas::level3`**: `trsm_in_place`; `trsm_with_par` and `trsm_in_place_with_par`, which take a `Par` (`trsm` itself runs sequentially); `gemmt` (GEMM that updates only one triangle of C), `gemm3m_c32`/`gemm3m_c64` (complex GEMM from three real multiplications), `gemm_batched`/`gemm_strided_batched` and the other batched routines, and the `gemm_strassen`, `gemm_winograd` and `gemm_cache_oblivious` variants.

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

### Parallel GEMM

`gemm` always runs on the calling thread, even with the `parallel` feature. To use Rayon, call `gemm_with_par` with `Par::Rayon` (`Par` lives in `oxiblas_core::parallel`):

```rust
use oxiblas_blas::level3::gemm_with_par;
use oxiblas_core::parallel::Par;
use oxiblas_matrix::Mat;

let a: Mat<f64> = Mat::filled(256, 256, 1.0);
let b: Mat<f64> = Mat::filled(256, 256, 2.0);
let mut c: Mat<f64> = Mat::zeros(256, 256);

// Requires the `parallel` feature
gemm_with_par(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut(), Par::Rayon);
```

Below a size gate (fewer than 128³ multiply-adds, `m*n*k`, in total or fewer than 80³ per thread) `Par::Rayon` falls back to the sequential path. Above it, a 2D driver splits `C` into tiles, one Rayon task per tile, and its results are bitwise identical to `Par::Seq`. `Par::RayonWith(n)` sizes the gate and the tile grid for `n` threads, but the tasks still run on the ambient Rayon pool (a pool of exactly `n` threads is planned for 0.2.4); bound the thread count with `rayon::ThreadPool::install` or `RAYON_NUM_THREADS`.

## Tensor Operations

### Einstein Summation

Supports 22 tensor contraction patterns:

```rust
use oxiblas_blas::tensor::einsum;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Row-major data: a is 2x3, b is 3x2, x has 2 elements, y has 3
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let x = vec![1.0, 2.0];
    let y = vec![3.0, 4.0, 5.0];

    // Matrix multiplication: C = A × B
    let c = einsum("ij,jk->ik", &a, &[2, 3], Some((&b, &[3, 2])))?;
    // c = [58.0, 64.0, 139.0, 154.0]

    // Outer product: C = x ⊗ y
    let c = einsum("i,j->ij", &x, &[2], Some((&y, &[3])))?;

    // Transpose: Aᵀ
    let at = einsum("ij->ji", &a, &[2, 3], None)?;

    // Matrix trace: tr(S) of a square matrix S
    let s = vec![1.0, 2.0, 3.0, 4.0];
    let trace = einsum("ii->", &s, &[2, 2], None)?;
    // trace = [5.0]

    // Hadamard (element-wise) product: C = A ⊙ A
    let c = einsum("ij,ij->ij", &a, &[2, 3], Some((&a, &[2, 3])))?;
    Ok(())
}
```

**Supported patterns**: matrix-matrix and matrix-vector products (`ij,jk->ik`, `ik,kj->ij`, `ij,j->i`), outer products (`i,j->ij`, `ij,ik->ijk`), diagonal and trace (`ii->i`, `ii->`), 2D and 3D transposes, element-wise and inner products (`ij,ij->ij`, `i,i->i`, `ij,ij->`, `i,i->`, `ijk,ijk->`), the 3D contraction `ijk,kl->ijl`, and axis sums (`ij->i`, `ij->j`, `ij->`, `ijk->ij`, `ijk->jk`). Any other notation returns `TensorError::Unsupported`.

### Batched Operations

```rust
use oxiblas_blas::tensor::{batched_matmul, Tensor3};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (batch_size, m, k, n) = (4, 2, 3, 2);
    let data_a = vec![1.0; batch_size * m * k];
    let data_b = vec![1.0; batch_size * k * n];

    // Batch of matrix multiplications
    let a = Tensor3::from_data(data_a, batch_size, m, k);
    let b = Tensor3::from_data(data_b, batch_size, k, n);

    let c = batched_matmul(&a, &b)?;
    // c[i] = a[i] × b[i] for each batch i
    Ok(())
}
```

### N-dimensional Tensors

`oxiblas_blas::ndtensor::NdTensor` is an N-dimensional array with dynamic shape and strides (row-major or column-major `Order`), reshaping, transposition and axis permutation, reductions, `matmul`, `contract`, `outer` and `trace`.

## Extended Precision

```rust
use oxiblas_blas::level1::{dot_kahan, dot_pairwise, dsdot};

let x = vec![1.0, 2.0, 3.0, 4.0];
let y = vec![5.0, 6.0, 7.0, 8.0];

// Kahan (compensated) summation for accuracy
let result_kahan = dot_kahan(&x, &y);

// Pairwise summation (divide-and-conquer)
let result_pairwise = dot_pairwise(&x, &y);

// Mixed precision: f32 inputs, f64 accumulation and result
let x_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
let y_f32: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
let result_f64 = dsdot(&x_f32, &y_f32);
```

## CBLAS Interface

`oxiblas_blas::cblas` exports 152 `#[no_mangle] extern "C"` entry points named `cblas_*`: every routine declared in netlib `cblas.h` 3.12.1 except the `cblas_xerbla` error handler. Row-major and column-major layouts are supported. The `layout`, `trans`, `uplo`, `side` and `diag` arguments are the `#[repr(C)]` enums `CblasLayout`, `CblasTranspose`, `CblasUplo`, `CblasDiag` and `CblasSide`.

```rust
use oxiblas_blas::cblas::{CblasLayout, CblasTranspose, cblas_dgemm};

// C = A × B with column-major A (2x3) and B (3x2)
let a = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
let b = [7.0, 9.0, 11.0, 8.0, 10.0, 12.0];
let mut c = [0.0; 4];

unsafe {
    cblas_dgemm(
        CblasLayout::ColMajor,
        CblasTranspose::NoTrans,
        CblasTranspose::NoTrans,
        2, 2, 3,           // m, n, k
        1.0,               // alpha
        a.as_ptr(), 2,     // a, lda
        b.as_ptr(), 3,     // b, ldb
        0.0,               // beta
        c.as_mut_ptr(), 2, // c, ldc
    );
}
// c = [58.0, 139.0, 64.0, 154.0] (column-major)
```

Known deviations from reference CBLAS:

- There is no `cblas_xerbla`: an invalid argument or a null pointer makes the call a silent no-op.
- The enum arguments are Rust enums, so a C caller that passes an out-of-range value causes undefined behaviour, where reference CBLAS reports it through `cblas_xerbla`.
- `cblas_i{s,d,c,z}amax` return a 32-bit `int`, not `CBLAS_INDEX` (`size_t`).
- There is no ILP64 (`_64`) API.
- The symbols are `#[no_mangle]` functions in the `oxiblas-blas` rlib; this workspace builds no C-linkable library (`cdylib` or `staticlib`). Any binary that links `oxiblas-blas` gets its `cblas_*` references bound to OxiBLAS, even when another BLAS is linked as well.

## Performance

Measured on 2026-11-09 on an Apple M3 against OpenBLAS 0.3.30 on one thread. Ratio is OxiBLAS's speed as a percentage of OpenBLAS's, so above 100% means OxiBLAS is faster.

| Operation | Size | OxiBLAS | OpenBLAS | Ratio |
|-----------|------|---------|----------|-------|
| DGEMM | 1024×1024 | 39.0 ms (55.0 GFLOPS) | 37.8 ms (56.8 GFLOPS) | 97% |
| SGEMM | 1024×1024 | 18.4 ms (116.8 GFLOPS) | 18.5 ms (115.8 GFLOPS) | 101% |
| DGEMV | 1000×1000 | 83.7 µs | 129 µs | 154% |
| DTRSM | 512×512 | 2.83 ms | 3.30 ms | 117% |
| DSYRK | 512×512 | 2.57 ms | 2.59 ms | 101% |
| DDOT | 1M | 143 µs | 156 µs | 109% |
| DAXPY | 1M | 176 µs | 194 µs | 110% |
| DNRM2 | 1M | 115 µs | 1.86 ms | 1610% |

The DDOT, DNRM2 and DGEMV kernels behind these rows are aarch64 (NEON) only; x86_64 keeps the older kernels. `gemm_with_par(.., Par::Rayon)` reaches 95-103% of OpenBLAS on 4 threads. On 8 threads it reaches 115-169%, partly because OpenBLAS is slower on 8 threads than on 4 on this CPU at n = 512 and 1024. [Performance](../../README.md#performance) in the root README has the method, more sizes and the parallel figures. x86_64 has not been compared with OpenBLAS since the comparison was fixed in 0.2.3.

## Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `default` | Core BLAS operations (`std` in `oxiblas-core` and `oxiblas-matrix`) | ✓ |
| `parallel` | Rayon-based parallelization (`Par::Rayon`, `Par::RayonWith`, the 2D GEMM driver, the parallel TRSM panels of `trsm_with_par`, parallel batched routines) | |
| `runtime-tuning` | `level3::runtime_autotune`: benchmarks GEMM block sizes at run time and caches them | |
| `force-scalar` | Use the portable scalar GEMM kernel and report no SIMD level; on aarch64 also the portable DOT, NRM2, GEMV and TRSM/TRMM tile code (see below) | |
| `max-simd-128` | Cap the GEMM kernel at 128-bit SIMD (SSE4.2 on x86_64, NEON on AArch64) | |
| `max-simd-256` | Cap the GEMM kernel at 256-bit SIMD (AVX2) | |

The three SIMD flags are forwarded to `oxiblas-core` and take effect in its capability detection and in GEMM kernel selection. On aarch64, `force-scalar` also keeps the generic `dot`, `nrm2` and `cnrm2`, `gemv`/`gemv_with_par`, the unit-stride `cblas_?gemv`, the TRSM/TRMM tile kernels, the GEMM B-panel packers and `gemm_small` on portable loops. The typed `*_f64`/`*_f32` Level-1 kernels (`dot_f64`, `nrm2_f32`, `axpy_f64`, ...) and the x86_64 Level-1 and `gemv` kernels pick AVX2 or NEON on their own and ignore all three flags.

## Architecture

```text
oxiblas-blas/
├── level1/               # Vector-vector operations
│   ├── dot.rs
│   ├── axpy.rs
│   ├── scal.rs
│   ├── strided.rs        # StridedSlice views and *_strided routines
│   └── ...
├── level2/               # Matrix-vector operations
│   ├── gemv/
│   ├── ger.rs
│   ├── symv.rs
│   └── ...
├── level3/               # Matrix-matrix operations
│   ├── gemm.rs           # General matrix multiply
│   ├── gemm_kernel/      # SIMD micro-kernels
│   ├── gemm_parallel.rs  # 2D parallel driver (`parallel` feature)
│   ├── autotune.rs       # Cache-aware blocking
│   ├── tri_gemm.rs       # Triangle-only blocked kernel (SYRK, SYR2K, GEMMT)
│   ├── syrk.rs
│   ├── trmm.rs
│   ├── trsm/             # Blocked TRSM driver and tile kernels (shared with TRMM)
│   └── ...
├── cblas/                # CBLAS-compatible extern "C" entry points
├── tensor.rs             # Einstein summation, batched matmul, Tensor3
└── ndtensor.rs           # N-dimensional tensor type
```

## Examples

The examples live in the `oxiblas` meta-crate:

```bash
# Run BLAS examples
cargo run -p oxiblas --example basic_blas
cargo run -p oxiblas --example tensor_operations
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

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.
