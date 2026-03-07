# oxiblas-lapack

**LAPACK (Linear Algebra PACKage) decompositions and solvers for OxiBLAS**

[![Crates.io](https://img.shields.io/crates/v/oxiblas-lapack.svg)](https://crates.io/crates/v/oxiblas-lapack)
[![Documentation](https://docs.rs/oxiblas-lapack/badge.svg)](https://docs.rs/oxiblas-lapack)

## Overview

`oxiblas-lapack` implements matrix decompositions and linear system solvers in pure Rust. Includes LU, QR, Cholesky, SVD, EVD, Schur decomposition and more, with optimizations that deliver **6-23× speedup** over naive implementations.

## Features

### Decompositions

#### LU Factorization
- **`Lu`** - LU with partial pivoting (PA = LU)
- **`LuFullPiv`** - LU with full pivoting (PAQ = LU)
- **`BandLu`** - Banded LU factorization for tridiagonal/pentadiagonal systems

#### Cholesky Factorization
- **`Cholesky`** - LL^T for symmetric positive definite matrices
- **`Ldlt`** - LDL^T decomposition (no square roots)
- **`BandCholesky`** - Banded Cholesky for sparse symmetric systems

#### QR Factorization
- **`Qr`** - Standard QR decomposition
- **`QrPivot`** - QR with column pivoting
- **`Lq`** - LQ decomposition (QR variant)
- **`Rq`** - RQ decomposition
- **`Cod`** - Complete orthogonal decomposition

#### SVD (Singular Value Decomposition)
- **`Svd`** - Standard SVD using Jacobi method
- **`SvdDc`** - SVD with divide-and-conquer (faster for large matrices)

#### Eigenvalue Decomposition
- **`SymmetricEvd`** - Symmetric eigenvalue decomposition (Jacobi)
- **`SymmetricEvdDc`** - Symmetric EVD with divide-and-conquer
- **`GeneralEvd`** - General eigenvalue decomposition (QR algorithm)

#### Other Decompositions
- **`Schur`** - Schur decomposition (A = QTQ^T)
- **`Hessenberg`** - Hessenberg reduction

### Solvers

- **`solve`** - General linear system Ax = b
- **`solve_multiple`** - Multiple right-hand sides
- **`solve_triangular`** - Triangular system solve
- **`lstsq`** - Least squares solution
- **`tridiag_solve`** - Tridiagonal system (Thomas algorithm)

### Utilities

- **`det`** - Determinant
- **`inv`** - Matrix inverse
- **`pinv`** - Pseudoinverse (via SVD)
- **`cond`** - Condition number
- **`rcond`** - Reciprocal condition number
- **`rank`** - Matrix rank
- **`nullity`** - Nullity (dimension of null space)
- **`null_space`** - Null space basis
- **`col_space`** - Column space basis
- **Norms**: `norm_1`, `norm_2`, `norm_inf`, `norm_frobenius`

## Installation

```toml
[dependencies]
oxiblas-lapack = "0.1"
```

## Usage Examples

### LU Decomposition

```rust
use oxiblas_lapack::lu::Lu;
use oxiblas_matrix::Mat;

let a = Mat::from_rows(&[
    &[2.0, 1.0, 1.0],
    &[4.0, 3.0, 3.0],
    &[8.0, 7.0, 9.0],
]);

// Compute LU factorization
let lu = Lu::compute(a.as_ref())?;

// Get L and U factors
let l = lu.l();
let u = lu.u();

// Solve Ax = b
let b = vec![4.0, 10.0, 24.0];
let x = lu.solve(&b)?;

// Determinant
let det = lu.determinant();

// Inverse
let inv = lu.inverse()?;
```

### QR Decomposition

```rust
use oxiblas_lapack::qr::Qr;
use oxiblas_matrix::Mat;

let a = Mat::from_rows(&[
    &[12.0, -51.0, 4.0],
    &[6.0, 167.0, -68.0],
    &[-4.0, 24.0, -41.0],
]);

// Compute QR
let qr = Qr::compute(a.as_ref())?;

// Get Q and R factors
let q = qr.q();
let r = qr.r();

// Solve least squares problem
let b = vec![1.0, 2.0, 3.0];
let x = qr.solve(&b)?;
```

### Cholesky Decomposition

```rust
use oxiblas_lapack::cholesky::Cholesky;
use oxiblas_matrix::Mat;

// Symmetric positive definite matrix
let a = Mat::from_rows(&[
    &[4.0, 12.0, -16.0],
    &[12.0, 37.0, -43.0],
    &[-16.0, -43.0, 98.0],
]);

// Compute Cholesky (LL^T)
let chol = Cholesky::compute(a.as_ref())?;

// Get L factor
let l = chol.l();

// Solve Ax = b (faster than LU for SPD matrices)
let b = vec![1.0, 2.0, 3.0];
let x = chol.solve(&b)?;

// Determinant
let det = chol.determinant();
```

### SVD (Singular Value Decomposition)

```rust
use oxiblas_lapack::svd::Svd;
use oxiblas_matrix::Mat;

let a = Mat::from_rows(&[
    &[3.0, 1.0],
    &[1.0, 3.0],
    &[1.0, 1.0],
]);

// Compute SVD: A = U Σ V^T
let svd = Svd::compute(a.as_ref())?;

// Get singular values
let singular_values = svd.singular_values();  // [4.24, 2.00]

// Get U and V^T matrices
let u = svd.u();
let vt = svd.vt();

// Pseudoinverse
let pinv = svd.pseudoinverse(1e-10)?;

// Rank and nullity
let rank = svd.rank(1e-10);
let nullity = svd.nullity(1e-10);

// Condition number
let cond = svd.condition_number();
```

### Eigenvalue Decomposition

```rust
use oxiblas_lapack::evd::SymmetricEvd;
use oxiblas_matrix::Mat;

// Symmetric matrix
let a = Mat::from_rows(&[
    &[4.0, 1.0, 1.0],
    &[1.0, 3.0, 2.0],
    &[1.0, 2.0, 3.0],
]);

// Compute eigenvalues and eigenvectors
let evd = SymmetricEvd::compute(a.as_ref())?;

// Get eigenvalues (sorted)
let eigenvalues = evd.eigenvalues();

// Get eigenvectors
let eigenvectors = evd.eigenvectors();

// Reconstruct: A = V Λ V^T
let reconstructed = evd.reconstruct();
```

### Linear System Solvers

```rust
use oxiblas_lapack::solve;
use oxiblas_matrix::Mat;

let a = Mat::from_rows(&[
    &[2.0, 1.0],
    &[1.0, 3.0],
]);
let b = vec![5.0, 8.0];

// Solve Ax = b
let x = solve(a.as_ref(), &b)?;
// x = [1.0, 2.0]

// Multiple right-hand sides
let b_multi = Mat::from_rows(&[
    &[5.0, 1.0],
    &[8.0, 2.0],
]);
let x_multi = solve_multiple(a.as_ref(), b_multi.as_ref())?;
```

### Matrix Utilities

```rust
use oxiblas_lapack::{det, inv, rank, norm_2};
use oxiblas_matrix::Mat;

let a = Mat::from_rows(&[
    &[1.0, 2.0],
    &[3.0, 4.0],
]);

// Determinant
let d = det(a.as_ref())?;  // -2.0

// Inverse
let a_inv = inv(a.as_ref())?;

// Matrix rank
let r = rank(a.as_ref(), 1e-10)?;  // 2

// 2-norm (spectral norm)
let n = norm_2(a.as_ref())?;  // ≈ 5.465
```

## Performance

### Optimization Achievements

| Operation | Size | Speedup vs Naive | Status |
|-----------|------|------------------|--------|
| **LU (blocked)** | 1024×1024 | **14-23×** | 🟢 |
| **Cholesky (blocked)** | 1024×1024 | **6-10×** | 🟢 |
| **QR** | 500×500 | ~**3-4×** | 🟢 |
| **SVD (D&C)** | 500×500 | ~**2-3×** | 🟢 |
| **SymmetricEVD (D&C)** | 500×500 | ~**2-3×** | 🟢 |

### Benchmarks vs OpenBLAS

| Algorithm | Target Performance | Status |
|-----------|-------------------|--------|
| QR Factorization | ~75% OpenBLAS | 🟢 Achieved |
| SVD | ~70% OpenBLAS | 🟢 Achieved |
| EVD | ~70% OpenBLAS | 🟢 Achieved |

## Algorithms

### LU Decomposition
- **Partial pivoting** for numerical stability
- **Blocked algorithm** for Level 3 BLAS performance
- **Cache-aware tiling** (14-23× speedup)

### Cholesky Factorization
- **Blocked algorithm** using SYRK/TRSM
- **Right-looking variant** for better cache locality
- **6-10× speedup** over naive implementation

### QR Decomposition
- **Householder reflections** for orthogonality
- **Blocked algorithm** for efficiency
- **Column pivoting** for rank-revealing

### SVD
- **Jacobi method** - stable for small to medium matrices
- **Divide-and-conquer** - faster for large matrices (SvdDc)
- **Two-stage approach** - Bidiagonalization + Golub-Kahan

### Eigenvalue Decomposition
- **Jacobi method** - symmetric EVD (robust)
- **Divide-and-conquer** - faster symmetric EVD
- **QR algorithm** - general eigenvalues (with shifts)
- **Schur decomposition** - intermediate step for general EVD

## Architecture

```
oxiblas-lapack/
├── lu/
│   ├── lu.rs          # LU with partial pivoting
│   ├── lu_full_piv.rs # LU with full pivoting
│   └── band_lu.rs     # Banded LU
├── cholesky/
│   ├── cholesky.rs    # Cholesky decomposition
│   ├── ldlt.rs        # LDL^T decomposition
│   └── band_chol.rs   # Banded Cholesky
├── qr/
│   ├── qr.rs          # Standard QR
│   ├── qr_pivot.rs    # QR with pivoting
│   ├── lq.rs          # LQ decomposition
│   ├── rq.rs          # RQ decomposition
│   └── cod.rs         # Complete orthogonal
├── svd/
│   ├── svd.rs         # Jacobi SVD
│   └── svd_dc.rs      # Divide-and-conquer SVD
├── evd/
│   ├── symmetric.rs   # Symmetric EVD (Jacobi)
│   ├── symmetric_dc.rs # Symmetric EVD (D&C)
│   └── general.rs     # General EVD (QR algorithm)
├── schur/
│   ├── schur.rs       # Schur decomposition
│   └── hessenberg.rs  # Hessenberg reduction
├── solve.rs           # Linear system solvers
└── utils.rs           # Determinant, inverse, norms, etc.
```

## Examples

```bash
cargo run --example lapack_decompositions
```

## Benchmarks

```bash
# QR benchmarks
cargo bench --package oxiblas-benchmarks --bench lapack_qr

# SVD benchmarks
cargo bench --package oxiblas-benchmarks --bench lapack_svd

# Factorization benchmarks
cargo bench --package oxiblas-benchmarks --bench lapack_factorization
```

## Numerical Stability

- **Pivoting** used where necessary (LU, QR)
- **Householder reflections** (QR) - numerically stable
- **Givens rotations** (Jacobi) - stable for symmetric problems
- **Divide-and-conquer** algorithms for large-scale problems

## Related Crates

- [`oxiblas-core`](../oxiblas-core/) - Core traits and SIMD
- [`oxiblas-matrix`](../oxiblas-matrix/) - Matrix types
- [`oxiblas-blas`](../oxiblas-blas/) - BLAS operations
- [`oxiblas-sparse`](../oxiblas-sparse/) - Sparse decompositions
- [`oxiblas`](../oxiblas/) - Meta-crate

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.
