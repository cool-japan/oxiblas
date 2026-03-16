# oxiblas-ndarray

**ndarray integration for OxiBLAS**

[![Crates.io](https://img.shields.io/crates/v/oxiblas-ndarray.svg)](https://crates.io/crates/oxiblas-ndarray)
[![Documentation](https://docs.rs/oxiblas-ndarray/badge.svg)](https://docs.rs/oxiblas-ndarray)

## Overview

`oxiblas-ndarray` provides seamless integration between OxiBLAS and the ndarray crate, enabling you to use high-performance BLAS/LAPACK operations on ndarray arrays.

## Features

- **Zero-copy conversions** between `ndarray::Array` and `oxiblas_matrix::Mat`
- **BLAS operations** on ndarray arrays
- **LAPACK decompositions** (LU, QR, SVD, Cholesky, etc.)
- **Type-safe** API leveraging Rust's type system
- **Compatible** with existing ndarray code

## Installation

```toml
[dependencies]
oxiblas-ndarray = "0.2"
```

## Usage

### BLAS Operations

```rust
use ndarray::Array2;
use oxiblas_ndarray::NdarrayExt;

// Create ndarray matrices
let a = Array2::<f64>::from_shape_vec((2, 3), vec![
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
]).unwrap();

let b = Array2::<f64>::from_shape_vec((3, 2), vec![
    7.0, 8.0,
    9.0, 10.0,
    11.0, 12.0,
]).unwrap();

// Matrix multiplication using OxiBLAS
let c = a.gemm(&b, 1.0, 0.0)?;
// c = [[58, 64], [139, 154]]

// Or use method syntax
let c = a.matmul(&b)?;
```

### Decompositions

```rust
use ndarray::Array2;
use oxiblas_ndarray::NdarrayExt;

let a = Array2::<f64>::from_shape_vec((3, 3), vec![
    2.0, 1.0, 1.0,
    4.0, 3.0, 3.0,
    8.0, 7.0, 9.0,
]).unwrap();

// LU decomposition
let lu = a.lu()?;
let det = lu.determinant();
let inv = lu.inverse()?;

// Solve Ax = b
let b = Array1::from_vec(vec![4.0, 10.0, 24.0]);
let x = lu.solve(&b)?;

// QR decomposition
let qr = a.qr()?;
let q = qr.q();
let r = qr.r();

// SVD
let svd = a.svd()?;
let singular_values = svd.singular_values();
let u = svd.u();
let vt = svd.vt();

// Cholesky (for SPD matrices)
let chol = a.cholesky()?;
let l = chol.l();
```

### Linear Solvers

```rust
use ndarray::{Array1, Array2};
use oxiblas_ndarray::NdarrayExt;

let a = Array2::from_shape_vec((2, 2), vec![
    2.0, 1.0,
    1.0, 3.0,
]).unwrap();

let b = Array1::from_vec(vec![5.0, 8.0]);

// Solve linear system
let x = a.solve(&b)?;
// x = [1.0, 3.0]

// Multiple right-hand sides
let b_multi = Array2::from_shape_vec((2, 2), vec![
    5.0, 1.0,
    8.0, 2.0,
]).unwrap();
let x_multi = a.solve_multiple(&b_multi)?;
```

### Conversions

```rust
use ndarray::Array2;
use oxiblas_matrix::Mat;
use oxiblas_ndarray::{FromNdarray, ToNdarray};

// ndarray -> oxiblas
let nd = Array2::<f64>::zeros((3, 3));
let ox = Mat::from_ndarray(&nd);

// oxiblas -> ndarray
let ox = Mat::<f64>::zeros(3, 3);
let nd = ox.to_ndarray();

// Zero-copy views (when possible)
let nd = Array2::<f64>::zeros((3, 3));
let ox_view = MatRef::from_ndarray_view(nd.view());
```

## Performance

All operations use OxiBLAS's optimized SIMD kernels:

- **GEMM**: 80-172% of OpenBLAS performance
- **Zero-copy** when data layout permits
- **Column-major** conversion handled automatically

## API Reference

### Extension Trait

The `NdarrayExt` trait extends `Array2<T>` with BLAS/LAPACK methods:

```rust
pub trait NdarrayExt<T: Scalar> {
    // BLAS Level 3
    fn gemm(&self, other: &Array2<T>, alpha: T, beta: T) -> Result<Array2<T>>;
    fn matmul(&self, other: &Array2<T>) -> Result<Array2<T>>;

    // LAPACK decompositions
    fn lu(&self) -> Result<Lu<T>>;
    fn qr(&self) -> Result<Qr<T>>;
    fn svd(&self) -> Result<Svd<T>>;
    fn cholesky(&self) -> Result<Cholesky<T>>;
    fn evd(&self) -> Result<SymmetricEvd<T>>;

    // Linear solvers
    fn solve(&self, b: &Array1<T>) -> Result<Array1<T>>;
    fn solve_multiple(&self, b: &Array2<T>) -> Result<Array2<T>>;

    // Utilities
    fn det(&self) -> Result<T>;
    fn inv(&self) -> Result<Array2<T>>;
    fn rank(&self, tol: f64) -> Result<usize>;
}
```

### Conversion Traits

```rust
pub trait FromNdarray<T> {
    fn from_ndarray(arr: &Array2<T>) -> Self;
    fn from_ndarray_view(view: ArrayView2<T>) -> Self;
}

pub trait ToNdarray<T> {
    fn to_ndarray(&self) -> Array2<T>;
    fn to_ndarray_view(&self) -> ArrayView2<T>;
}
```

## Examples

```bash
cargo run --example ndarray_integration
```

## Comparison with ndarray-linalg

| Feature | oxiblas-ndarray | ndarray-linalg |
|---------|----------------|----------------|
| Pure Rust | ✓ | |
| No C dependencies | ✓ | |
| BLAS/LAPACK | ✓ | ✓ |
| OpenBLAS backend | | ✓ |
| Intel MKL backend | | ✓ |
| Custom backend | ✓ (OxiBLAS) | |
| Performance | 80-172% OpenBLAS | 100% (uses OpenBLAS) |
| Cross-compilation | Easy | Difficult |

## Related Crates

- [`ndarray`](https://crates.io/crates/ndarray) - N-dimensional arrays
- [`oxiblas`](../oxiblas/) - Core BLAS/LAPACK implementation
- [`oxiblas-matrix`](../oxiblas-matrix/) - Matrix types

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.
