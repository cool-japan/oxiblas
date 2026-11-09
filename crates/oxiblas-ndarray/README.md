# oxiblas-ndarray

**ndarray integration for OxiBLAS**

[![Crates.io](https://img.shields.io/crates/v/oxiblas-ndarray.svg)](https://crates.io/crates/oxiblas-ndarray)
[![Documentation](https://docs.rs/oxiblas-ndarray/badge.svg)](https://docs.rs/oxiblas-ndarray)

## Overview

`oxiblas-ndarray` provides integration between OxiBLAS and the ndarray crate (0.17), enabling you to use OxiBLAS BLAS/LAPACK operations on ndarray arrays. The API is a set of free functions, re-exported through `oxiblas_ndarray::prelude`, that take and return `Array1` / `Array2` values.

## Features

- **Conversions** between `ndarray::Array2` and `oxiblas_matrix::Mat`: the owned conversions always copy, while the view conversions (`ArrayView2` to `MatRef`) borrow the array's buffer without copying
- **BLAS operations** on ndarray arrays: Level 1 (dot, nrm2, asum, axpy, scal), Level 2 (gemv), Level 3 (gemm), matrix norms and helpers, with `Complex32` / `Complex64` variants
- **LAPACK decompositions** (LU, QR, SVD, Cholesky, symmetric, Hermitian and general eigenvalue decompositions, Schur), solvers, inverse, pseudo-inverse, determinant, rank, condition number and randomized SVD
- **Parallel GEMM** with Rayon (`parallel` feature) and **sparse** CSR/CSC conversions, SpMV and a CG solver (`sparse` feature)
- **Type-safe** API: result structs (`LuResult`, `QrResult`, `SvdResult`, ...) and a `LapackError` type for the LAPACK wrappers
- **Compatible** with existing ndarray code

## Installation

```toml
[dependencies]
oxiblas-ndarray = "0.2"

# With parallel GEMM (Rayon)
oxiblas-ndarray = { version = "0.2", features = ["parallel"] }

# With sparse (CSR/CSC) integration
oxiblas-ndarray = { version = "0.2", features = ["sparse"] }
```

## Usage

### BLAS Operations

```rust
use ndarray::{array, Array1, Array2};
use oxiblas_ndarray::prelude::*;

let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
let b = array![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];

// Matrix multiplication using OxiBLAS
let c = matmul(&a, &b);
assert_eq!(c, array![[58.0, 64.0], [139.0, 154.0]]);

// C = alpha * A * B + beta * C
let mut c2 = Array2::<f64>::zeros((2, 2));
gemm_ndarray(1.0, &a, &b, 0.0, &mut c2);
assert_eq!(c2, c);

// Matrix-vector product and dot product
let x = array![1.0, 0.0, 1.0];
let y = matvec(&a, &x);                 // [4.0, 10.0]
let d = dot_ndarray(&x, &Array1::from_vec(vec![1.0, 2.0, 3.0]));  // 4.0
```

### Decompositions

```rust
use ndarray::{array, Array1};
use oxiblas_ndarray::prelude::*;

fn main() -> LapackResult<()> {
    let a = array![
        [2.0, 1.0, 1.0],
        [4.0, 3.0, 3.0],
        [8.0, 7.0, 9.0],
    ];

    // LU decomposition (A = P * L * U)
    let lu = lu_ndarray(&a)?;
    let det = lu.det();                              // 4.0
    let b = Array1::from_vec(vec![4.0, 10.0, 24.0]);
    let x = lu.solve(&b);                            // solves A x = b: [1.0, 1.0, 1.0]

    // QR decomposition
    let qr = qr_ndarray(&a)?;
    let (q, r) = (&qr.q, &qr.r);

    // SVD (singular values in descending order)
    let svd = svd_ndarray(&a)?;
    let (u, s, vt) = (&svd.u, &svd.s, &svd.vt);

    // Cholesky (symmetric positive definite matrices)
    let spd = array![[4.0, 1.0], [1.0, 3.0]];
    let chol = cholesky_ndarray(&spd)?;
    let l = &chol.l;

    // Symmetric eigenvalue decomposition (eigenvalues in ascending order)
    let evd = eig_symmetric(&spd)?;
    let eigenvalues = &evd.eigenvalues;
    let eigenvectors = &evd.eigenvectors;

    Ok(())
}
```

### Linear Solvers

```rust
use ndarray::array;
use oxiblas_ndarray::prelude::*;

fn main() -> LapackResult<()> {
    let a = array![[2.0, 1.0], [1.0, 3.0]];
    let b = array![5.0, 8.0];

    // Solve linear system
    let x = solve_ndarray(&a, &b)?;
    // x = [1.4, 2.2]

    // Multiple right-hand sides
    let b_multi = array![[5.0, 1.0], [8.0, 2.0]];
    let x_multi = solve_multiple_ndarray(&a, &b_multi)?;

    // Least squares, inverse, determinant, rank
    let x_ls = lstsq_ndarray(&a, &b)?;
    let a_inv = inv_ndarray(&a)?;
    let det = det_ndarray(&a)?;                      // 5.0
    let rank = rank_ndarray(&a)?;                    // 2

    Ok(())
}
```

### Conversions

```rust
use ndarray::{array, Array2};
use oxiblas_ndarray::prelude::*;

// ndarray -> oxiblas (copies into a new Mat)
let nd = array![[1.0, 2.0], [3.0, 4.0]];
let ox = array2_to_mat(&nd);

// oxiblas -> ndarray (copies)
let back: Array2<f64> = mat_to_array2(&ox);
assert_eq!(back, nd);

// Zero-copy view of a column-major array (None for layouts that cannot be borrowed)
let col_major: Array2<f64> = zeros_f(3, 3);
let view = col_major.view();
let mat_ref = array_view_to_mat_ref(&view);
assert!(mat_ref.is_some());

// A row-major array is borrowed as a transposed MatRef; the flag says so
let row_major = Array2::<f64>::zeros((3, 3));
let view = row_major.view();
let (mat_ref, transposed) = array_view_to_mat_ref_or_transposed(&view).expect("contiguous");
assert!(transposed);
```

### Parallel and Sparse

```rust
// Requires the `parallel` and `sparse` features
use ndarray::array;
use oxiblas_ndarray::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = array![[1.0, 2.0], [3.0, 4.0]];
    let b = array![[5.0, 6.0], [7.0, 8.0]];
    let c = matmul_par(&a, &b);                      // Rayon-parallel GEMM

    // Dense -> CSR (keeps every entry that is not exactly zero)
    let dense = array![[4.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 4.0]];
    let csr = array2_to_csr(&dense);
    assert_eq!(csr.nnz(), 7);

    // y = A * x
    let x = array![1.0, 2.0, 3.0];
    let y = spmv_ndarray(&csr, &x);                  // [6.0, 12.0, 14.0]

    // Conjugate gradient solve (A must be symmetric positive definite)
    let solution = sparse_solve_ndarray(&csr, &y)?;  // ~ [1.0, 2.0, 3.0]
    Ok(())
}
```

## Performance

The BLAS and LAPACK wrappers delegate to OxiBLAS's optimized SIMD kernels:

- **GEMM**: runs `oxiblas-blas` GEMM; see [Performance](../../README.md#performance) in the root README for measured figures
- **Copying wrappers** - The BLAS/LAPACK convenience functions copy their inputs into `Mat`; only the view conversions are zero-copy
- **Column-major** - `matmul` and `matmul_par` return column-major arrays (`matmul_c` returns row-major); input layout is handled by the copy

## API Reference

All of the items below are available through `oxiblas_ndarray::prelude::*` unless noted.

### BLAS (`blas`)

| Level | Functions |
|-------|-----------|
| Level 1 | `dot_ndarray`, `dot_view`, `nrm2_ndarray`, `asum_ndarray`, `axpy_ndarray`, `scal_ndarray`; complex: `dotc_c64_ndarray`, `dotu_c64_ndarray`, `nrm2_c64_ndarray`, `asum_c64_ndarray`, `axpy_c64_ndarray`, `scal_c64_ndarray` and the `_c32_` counterparts |
| Level 2 | `gemv_ndarray` (takes a `Transpose`), `matvec`, `matvec_t` |
| Level 3 | `gemm_ndarray(alpha, a, b, beta, c)`, `matmul`, `matmul_c`, `matmul_into`; with `parallel`: `gemm_par_ndarray`, `matmul_par` |
| Norms | `frobenius_norm`, `norm_1`, `norm_inf`, `norm_max` (the last three for `f64`), complex `*_c32` / `*_c64` versions |
| Helpers | `eye`, `eye_f`, `trace`, `transpose`, `conj_transpose_c32`, `conj_transpose_c64`, `eye_c32`, `eye_c64`, `trace_c32`, `trace_c64` |

### LAPACK (`lapack`)

| Area | Functions and results |
|------|-----------------------|
| Decompositions | `lu_ndarray` (`LuResult { l, u, perm }` with `solve`, `det`), `qr_ndarray` (`QrResult { q, r }`), `svd_ndarray` / `svd_truncated` (`SvdResult { u, s, vt }`), `cholesky_ndarray` (`CholeskyResult { l }` with `solve`, `det`), `schur_ndarray` (`SchurResult`) |
| Eigenvalues | `eig_symmetric` / `eigvals_symmetric` (`SymEvdResult { eigenvalues, eigenvectors }`), `eig_ndarray` / `eigvals_ndarray` (general matrices, `Eigenvalue` values) |
| Randomized | `rsvd_ndarray`, `rsvd_power_ndarray` (`RandomizedSvdResult { u, s, v }`), `low_rank_approx_ndarray` |
| Solvers | `solve_ndarray`, `solve_multiple_ndarray`, `lstsq_ndarray`, `tridiag_solve_ndarray`, `tridiag_solve_spd_ndarray`, `tridiag_solve_multiple_ndarray` |
| Matrix functions | `inv_ndarray`, `pinv_ndarray`, `det_ndarray`, `cond_ndarray`, `rank_ndarray` |
| Complex (`lapack` module only) | `svd_complex_ndarray`, `qr_complex_ndarray`, `cholesky_hermitian_ndarray`, `eig_hermitian_ndarray` |
| Errors | `LapackError` (`Singular`, `NotPositiveDefinite`, `DimensionMismatch`, `NotConverged`, `Other`) and `LapackResult<T>` |

### Conversions (`conversions`)

| Kind | Functions |
|------|-----------|
| Copying | `array2_to_mat`, `array2_into_mat`, `mat_to_array2`, `mat_to_array2_c`, `mat_ref_to_array2`, and `ArrayD` versions (`arrayd_to_mat`, `mat_to_arrayd`, `arrayd_to_array2`, `array2_to_arrayd`, ...) |
| Zero-copy views | `array_view_to_mat_ref`, `array_view_to_mat_ref_or_transposed`, `array_view_mut_to_mat_mut`, and `ArrayViewD` versions; they return `None` for layouts that cannot be borrowed (not contiguous along one axis, or a negative stride) |
| Layout helpers | `zeros_f`, `filled_f`, `is_column_major`, `is_row_major`, `to_column_major`, `array1_to_vec`, `slice_to_array1` |

### Sparse (`sparse`, feature `sparse`)

`array2_to_csr`, `array2_to_csc` (and `*_with_tolerance`), `csr_to_array2`, `csc_to_array2`, `spmv_ndarray`, `spmv_full_ndarray`, `sparse_solve_ndarray`, `sparse_solve_ndarray_with_options` and `SparseNdarrayError`.

## Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `parallel` | Rayon-parallel GEMM (`gemm_par_ndarray`, `matmul_par`) via `oxiblas-blas/parallel` and `oxiblas-core/parallel` | |
| `sparse` | CSR/CSC conversions, SpMV and a CG solver via `oxiblas-sparse` | |

The default features enable `std` support in `oxiblas-core` and `oxiblas-matrix`.

## Benchmarks

```bash
cargo bench --package oxiblas-ndarray --bench ndarray_blas
cargo bench --package oxiblas-ndarray --bench faer_comparison
```

## Comparison with ndarray-linalg

| Feature | oxiblas-ndarray | ndarray-linalg |
|---------|----------------|----------------|
| Pure Rust | ✓ | |
| No C dependencies (library and its normal dependencies) | ✓ | |
| BLAS/LAPACK | ✓ | ✓ |
| OpenBLAS backend | | ✓ |
| Intel MKL backend | | ✓ |
| Custom backend | ✓ (OxiBLAS) | |
| Performance | See [Performance](../../README.md#performance) | Depends on the backend |
| Cross-compilation | Easy | Difficult |

## Related Crates

- [`ndarray`](https://crates.io/crates/ndarray) - N-dimensional arrays
- [`oxiblas`](../oxiblas/) - Core BLAS/LAPACK implementation
- [`oxiblas-matrix`](../oxiblas-matrix/) - Matrix types

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.
