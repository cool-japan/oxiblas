# oxiblas-sparse

**Sparse matrix operations, iterative solvers, and preconditioners for OxiBLAS**

[![Crates.io](https://img.shields.io/crates/v/oxiblas-sparse.svg)](https://crates.io/crates/oxiblas-sparse)
[![Documentation](https://docs.rs/oxiblas-sparse/badge.svg)](https://docs.rs/oxiblas-sparse)

## Overview

`oxiblas-sparse` provides comprehensive sparse linear algebra functionality including 9 storage formats, sparse BLAS-style operations, iterative solvers, direct factorizations, advanced preconditioners, sparse eigenvalue/SVD algorithms, fill-reducing orderings, graph utilities and Matrix Market I/O.

## Features

### Sparse Matrix Formats (9 types)

- **CSR** (Compressed Sparse Row) - `CsrMatrix`, general sparse matrices
- **CSC** (Compressed Sparse Column) - `CscMatrix`, efficient column operations
- **COO** (Coordinate) - `CooMatrix` / `CooMatrixBuilder`, easy construction and modification
- **ELL** (ELLPACK) - `EllMatrix`, GPU-friendly format
- **DIA** (Diagonal) - `DiaMatrix`, band-diagonal matrices
- **BSR** (Block Sparse Row) - `BsrMatrix`, block-structured matrices
- **BSC** (Block Sparse Column) - `BscMatrix`, block column format
- **HYB** (Hybrid ELL+COO) - `HybMatrix`, adaptive format
- **SELL** (Sliced ELLPACK) - `SellMatrix`, optimized for GPUs

The `convert` module converts between the formats (for example `coo_to_csr`, `csr_to_dia`, `csr_to_sell`), and `analyze_sparsity_pattern` recommends a format for a given matrix.

### Sparse Operations

- **SpMV / SpMM / SpGEMM** (`ops` module) - `spmv`, `spmv_transpose`, `spmv_csc`, `spmm` (sparse times dense), `spmm_sparse` (sparse times sparse), plus symmetric, Hermitian and triangular variants
- **Arithmetic** - `spadd`, `scale`, `scale_rows`, `scale_cols`, rank-1 updates, `asum`, `frobenius_norm`, `one_norm`, `infinity_norm`
- **Triangular solves** - `sptrsv_lower`, `sptrsv_upper`
- **Parallel variants** (with the `parallel` feature) - `spmv_f64_par`, `spmv_f32_par`, `spmm_sparse_par`; `spmv_auto` and `spmm_sparse_auto` pick the best variant for the matrix

### Iterative Solvers (10+ methods)

Solvers live in `oxiblas_sparse::linalg` (also re-exported by `linalg::prelude` and `oxiblas_sparse::prelude`; the `linalg::iterative` paths still work). Each takes the matrix, the right-hand side (a slice of vectors for the block solvers) and an initial guess `x0`. Preconditioned variants take the preconditioner as a closure that computes `z = M^-1 r` (`|r: &[T]| -> Vec<T>`).

#### Krylov Subspace Methods
- **GMRES** - Generalized Minimal Residual (with restart): `gmres`
- **FGMRES** - Flexible GMRES (variable preconditioning): `fgmres`, `fgmres_ir`
- **PGMRES** - Preconditioned GMRES: `pgmres`
- **CG** - Conjugate Gradient (symmetric positive definite): `cg`
- **PCG** - Preconditioned CG: `pcg`
- **Block-CG** - Multiple right-hand sides: `block_cg`, `block_pcg`
- **BiCGStab** - BiConjugate Gradient Stabilized: `bicgstab`
- **MINRES** - Minimal Residual (symmetric): `minres`
- **PMINRES** - Preconditioned MINRES: `pminres`
- **QMR** - Quasi-Minimal Residual: `qmr`, `pqmr`
- **TFQMR** - Transpose-Free QMR: `tfqmr`, `ptfqmr`
- **IDR(s)** - Induced Dimension Reduction: `idrs`, `pidrs`
- **Block-GMRES** - Multiple RHS: `block_gmres`

Convergence monitoring (stagnation and divergence detection, rate estimation) is in `linalg::convergence`.

#### Eigenvalue Methods
- **Lanczos** - Symmetric matrices: `Lanczos`
- **Block Lanczos** - Multiple eigenvalues: `BlockLanczos`
- **Arnoldi** - General matrices: `Arnoldi`
- **Block Arnoldi** - Multiple eigenvalues: `BlockArnoldi`
- **IRAM** - Implicitly Restarted Arnoldi Method: `IRAM`
- **Thick-restart Lanczos** and **LOBPCG** - `ThickRestartLanczos`, `Lobpcg` (in `linalg::eigenvalue`)
- **Generalized problems** - `GeneralizedEigen` (A x = lambda B x)

### Direct Solvers

- **Cholesky** - `SparseCholesky` (optional AMD ordering via `with_ordering`)
- **LU** - `SparseLU`; threshold and static pivoting in `SparseLuThreshold` / `SparseLuStaticPivot`; `SparseLdlt` for symmetric indefinite matrices
- **QR** - `SparseQR`, `SparseQRGivens` and `SparseQr` (COLAMD column ordering); least squares for overdetermined systems
- **Supernodal and multifrontal** - `SupernodalCholesky`, `SupernodalLU`, `MultifrontalCholesky`, `MultifrontalLU`
- **Out-of-core** - `OutOfCoreCholesky`, `OutOfCoreLu`
- **Triangular solvers** - `solve_lower`, `solve_upper`

### Advanced Preconditioners (12+ types)

#### Incomplete Factorizations
- **ILU(0)** - Zero-fill incomplete LU: `ILU0`
- **ILUT** - ILU with threshold: `ILUT`
- **ILUTP** - ILUT with pivoting: `ILUTP`
- **IC0** - Incomplete Cholesky (zero-fill): `IC0`
- **ICT** - IC with threshold: `ICT`

#### Iterative Preconditioners
- **Jacobi** - Diagonal scaling: `Jacobi`
- **Block Jacobi** - Block diagonal: `BlockJacobi`
- **Gauss-Seidel** - Forward/backward sweeps: `GaussSeidel`
- **SOR** - Successive Over-Relaxation: `SOR`
- **SSOR** - Symmetric SOR: `SSOR`

#### Multigrid Methods
- **AMG** - Algebraic Multigrid (classical Ruge-Stüben): `AMG`
- **SA-AMG** - Smoothed Aggregation AMG: `SAMG`

#### Approximate Inverse
- **SPAI** - Sparse Approximate Inverse: `SPAI`
- **AINV** - Approximate Inverse: `AINV`

#### Domain Decomposition
- **Additive Schwarz** - Domain decomposition: `AdditiveSchwarz`
- **Polynomial** - Neumann/Chebyshev polynomial preconditioners: `Polynomial`

### Sparse Eigenvalue & SVD

- **Lanczos method** (symmetric)
- **Arnoldi iteration** (general)
- **Shift-invert** spectral transformation: `ShiftInvertLanczos`
- **Polynomial filtering** (Chebyshev): `PolynomialFilteredLanczos`
- **Interval eigenvalue** computation (Sturm sequence): `IntervalEigen`
- **Truncated SVD**: `TruncatedSVD`
- **Randomized SVD**: `RandomizedSparseSvd`, `randomized_sparse_svd`
- **Incremental SVD** (Brand algorithm): `IncrementalSVD`

### Matrix Reordering

Orderings take a `CscMatrix` and return a permutation (`Vec<usize>`).

- **RCM** - Reverse Cuthill-McKee (bandwidth reduction): `reverse_cuthill_mckee`
- **AMD** - Approximate Minimum Degree: `approximate_minimum_degree`
- **MMD** - Multiple Minimum Degree: `ordering::multiple_minimum_degree`
- **COLAMD** - Column AMD (for unsymmetric matrices): `ordering::colamd`
- **Nested Dissection** - Level-set based ordering: `nested_dissection` (and `ordering::multilevel_nested_dissection`)

### Other Utilities

- **Graph algorithms** (`graph`) - connected components, bipartite matching, level sets, graph partitioning, bandwidth and profile, pseudo-peripheral vertices
- **Matrix Market I/O** (`mtx`) - `read_matrix_market`, `write_matrix_market` and string variants
- **Stochastic estimators** (`stochastic`) - Hutchinson, Hutch++ and XTrace trace estimation, diagonal and log-determinant estimation
- **Test matrices** (`test_matrices`) - `laplacian_2d`, `laplacian_3d`, `poisson_1d`, `tridiagonal`, `diagonal`, `arrow_matrix`, `random_spd`

## Installation

```toml
[dependencies]
oxiblas-sparse = "0.2"

# With Rayon-parallel SpMV / SpGEMM
oxiblas-sparse = { version = "0.2", features = ["parallel"] }
```

## Usage Examples

`use oxiblas_sparse::prelude::*;` brings the formats, `spmv` and the `linalg` solvers, preconditioners and orderings into scope.

### Creating Sparse Matrices

```rust
use oxiblas_sparse::ops::spmv;
use oxiblas_sparse::{CooMatrix, CsrMatrix};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // From COO (easy construction)
    let mut coo = CooMatrix::<f64>::new_empty(4, 4);
    coo.push(0, 0, 4.0);
    coo.push(0, 1, -1.0);
    coo.push(1, 0, -1.0);
    coo.push(1, 1, 4.0);
    // ... add more elements

    // Convert to CSR (efficient operations); duplicate entries are summed
    let csr: CsrMatrix<f64> = coo.to_csr();

    // Or build CSR directly from its arrays: [1 0 2; 0 3 0; 4 0 5]
    let a = CsrMatrix::new(
        3,
        3,
        vec![0, 2, 3, 5],             // row pointers
        vec![0, 2, 1, 0, 2],          // column indices
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
    )?;
    assert_eq!(a.nnz(), 5);

    // Sparse matrix-vector product: y = alpha * A * x + beta * y
    let x = vec![1.0; 4];
    let mut y = vec![0.0; 4];
    spmv(1.0, &csr, &x, 0.0, &mut y);
    Ok(())
}
```

### Iterative Solvers

```rust
use oxiblas_sparse::laplacian_2d;
use oxiblas_sparse::linalg::gmres;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 2-D Laplacian on a 10×10 grid (100 unknowns)
    let a = laplacian_2d(10, 10)?;
    let b = vec![1.0; 100];
    let x0 = vec![0.0; 100];

    // Solve Ax = b using restarted GMRES
    let result = gmres(
        &a, &b, &x0,
        30,          // restart
        1e-10,       // tolerance (relative to ||b||)
        500,         // max iterations
    )?;

    println!("Converged: {}", result.converged);
    println!("Residual: {}", result.residual_norm);
    println!("Iterations: {}", result.iterations);
    println!("Solution: {:?}", &result.x[..3]);
    Ok(())
}
```

### Preconditioned Solvers

```rust
use oxiblas_sparse::laplacian_2d;
use oxiblas_sparse::linalg::IC0;
use oxiblas_sparse::linalg::iterative::pcg;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = laplacian_2d(10, 10)?;
    let b = vec![1.0; 100];
    let x0 = vec![0.0; 100];

    // Incomplete Cholesky preconditioner; apply() solves (L L^T) z = r
    let ic = IC0::new(&a)?;

    // Solve with preconditioned CG: the preconditioner is a closure z = M^-1 r
    let result = pcg(
        &a, &b, &x0,
        |r| ic.apply(r), // preconditioner
        1e-8,            // tolerance (relative to ||b||)
        1000,            // max iterations
    )?;
    println!("Converged in {} iterations", result.iterations);
    Ok(())
}
```

### AMG Preconditioner

```rust
use oxiblas_sparse::laplacian_2d;
use oxiblas_sparse::linalg::iterative::pcg;
use oxiblas_sparse::linalg::{AMG, AMGConfig, AMGCycleType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = laplacian_2d(10, 10)?;
    let b = vec![1.0; 100];
    let x0 = vec![0.0; 100];

    // Create AMG preconditioner (multilevel)
    let amg = AMG::new(&a, AMGConfig {
        max_levels: 10,
        cycle_type: AMGCycleType::V,
        ..Default::default()
    })?;

    // Solve with AMG-preconditioned CG (apply() writes into an output slice)
    let result = pcg(&a, &b, &x0, |r| {
        let mut z = vec![0.0; r.len()];
        amg.apply(r, &mut z);
        z
    }, 1e-8, 1000)?;
    println!("Converged in {} iterations", result.iterations);
    Ok(())
}
```

### Direct Solvers

```rust
use oxiblas_sparse::laplacian_2d;
use oxiblas_sparse::linalg::SparseCholesky;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Direct solvers take CSC input
    let a = laplacian_2d(10, 10)?.to_csc();
    let b = vec![1.0; 100];

    // Sparse Cholesky with an approximate-minimum-degree ordering
    let chol = SparseCholesky::with_ordering(&a)?;
    let x = chol.solve(&b);
    println!("x[0] = {}", x[0]);
    Ok(())
}
```

### Sparse Eigenvalues

```rust
use oxiblas_sparse::laplacian_2d;
use oxiblas_sparse::linalg::{Lanczos, LanczosConfig, WhichEigenvalues};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = laplacian_2d(10, 10)?;

    // Compute the largest eigenvalues using Lanczos
    let config = LanczosConfig {
        num_eigenvalues: 4,                        // number of eigenvalues
        which: WhichEigenvalues::LargestMagnitude,
        max_iterations: 300,                       // max iterations
        tolerance: 1e-10,                          // tolerance
        ..Default::default()
    };
    let result = Lanczos::new(config).compute(&a, None)?;

    println!("Eigenvalues: {:?}", result.eigenvalues);
    println!("Converged: {}", result.converged);
    Ok(())
}
```

### Matrix Reordering

```rust
use oxiblas_sparse::laplacian_2d;
use oxiblas_sparse::linalg::ordering::{bandwidth_with_ordering, reverse_cuthill_mckee};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = laplacian_2d(10, 10)?.to_csc();

    // Reverse Cuthill-McKee ordering (bandwidth reduction); perm[new] = old
    let perm = reverse_cuthill_mckee(&a);

    // Bandwidth of P*A*P^T under that ordering
    let bandwidth = bandwidth_with_ordering(&a, &perm);
    println!("Bandwidth after RCM: {bandwidth}");
    Ok(())
}
```

### Matrix Market I/O

```rust
use oxiblas_sparse::{CsrMatrix, read_matrix_market_str, write_matrix_market_str};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = CsrMatrix::<f64>::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0])?;

    // Serialize to Matrix Market text and read it back
    let text = write_matrix_market_str(&a, Some("diagonal 2x2"))?;
    let back: CsrMatrix<f64> = read_matrix_market_str(&text)?;
    assert_eq!(back.nnz(), 2);
    Ok(())
}
```

## Performance

### Solver Convergence

Iterations to a residual of 1e-8 relative to `||b||` on `laplacian_2d(100, 100)` (10,000 unknowns, 49,600 non-zeros), with `b = 1` and a zero initial guess:

| Solver | Preconditioner | Iterations |
|--------|----------------|-----------|
| CG | None | 187 |
| PCG | IC(0) | 79 |
| PCG | AMG (V-cycle) | 9 |
| GMRES(30) | None | 1,398 |

Before 0.2.3, GMRES(30) restarted after every step (a lucky-breakdown test read a sub-diagonal entry that had already been zeroed), so it behaved like GMRES(1) and needed 37,571 iterations here. `gmres`, `pgmres`, `fgmres`, `block_gmres`, `idrs`, `pidrs`, `qmr` and `pminres` confirm convergence with the true residual `‖b − A·x‖` (and report it as `residual_norm`); the other solvers stop on their recursive residual and consult the true residual when they break down.

### Memory Efficiency

Indices are `usize`, so on a 64-bit target each f64 non-zero costs 16 bytes in CSR and 24 bytes in COO:

| Format | Bytes | 1M×1M with 5M non-zeros | vs Dense |
|--------|-------|-------------------------|----------|
| Dense | 8 × nrows × ncols | 8 TB | 100% |
| CSR | 16 × nnz + 8 × (nrows + 1) | 88 MB | 0.0011% |
| COO | 24 × nnz | 120 MB | 0.0015% |

## Algorithms

### GMRES
- **Arnoldi process** for Krylov subspace
- **Restart** mechanism for large problems
- **Flexible** variant for variable preconditioning

### CG (Conjugate Gradient)
- **Short recurrences** - O(n) memory, no basis storage
- **Preconditioned** variants
- **Block** version for multiple RHS

### AMG (Algebraic Multigrid)
- **Ruge-Stüben coarsening** - classical AMG
- **Smoothed aggregation** - modern variant
- **V-cycle/W-cycle** - multilevel hierarchy

### Lanczos
- **Symmetric** eigenvalue problems
- **Reorthogonalization** for numerical stability
- **Shift-invert** for interior eigenvalues

## Architecture

```text
oxiblas-sparse/
├── csr.rs, csc.rs, coo.rs     # CSR, CSC, COO formats
├── ell.rs, dia.rs, hyb.rs     # ELL, DIA, HYB formats
├── bsr.rs, bsc.rs, sell.rs    # BSR, BSC, SELL formats
├── convert.rs                 # Format conversions, sparsity analysis
├── ops/                       # SpMV, SpMM, SpGEMM, scaling, norms
├── linalg/
│   ├── iterative/             # CG, GMRES, FGMRES, BiCGStab, MINRES, QMR, TFQMR, IDR(s), block solvers
│   ├── precond/               # Jacobi, Gauss-Seidel/SOR/SSOR, AMG, SAMG, SPAI, AINV, Schwarz, polynomial
│   ├── cholesky.rs, lu.rs     # Direct solvers, IC0/ICT and ILU0/ILUT/ILUTP
│   ├── qr.rs, sparse_qr.rs    # Sparse QR
│   ├── supernodal.rs, multifrontal_*.rs, out_of_core.rs, pivoting.rs
│   ├── eigenvalue/            # Lanczos, Arnoldi, IRAM, block, shift-invert, TRL, LOBPCG, ...
│   ├── svd/                   # Truncated, randomized and incremental SVD
│   ├── ordering/              # RCM, AMD, MMD, COLAMD, nested dissection
│   ├── triangular.rs          # solve_lower, solve_upper
│   └── convergence.rs         # Convergence monitoring
├── graph/                     # Components, matching, partitioning, level sets
├── mtx.rs                     # Matrix Market I/O
├── stochastic.rs              # Stochastic trace and diagonal estimators
└── test_matrices.rs           # Laplacian, Poisson, random SPD generators
```

## Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `parallel` | Rayon-parallel SpMV and SpGEMM (`spmv_f64_par`, `spmv_f32_par`, `spmm_sparse_par`, ...) via `rayon` and `oxiblas-core/parallel` | |

The default features enable `std` support in `oxiblas-core` and `oxiblas-matrix`.

## Examples

```bash
cargo run -p oxiblas --example sparse_matrices
```

## Benchmarks

```bash
cargo bench --package oxiblas-benchmarks --bench sparse_ops
cargo bench --package oxiblas-benchmarks --bench sparse_eigen_svd
```

## Related Crates

- [`oxiblas-core`](../oxiblas-core/) - Core traits
- [`oxiblas-blas`](../oxiblas-blas/) - Dense BLAS
- [`oxiblas-lapack`](../oxiblas-lapack/) - Dense decompositions
- [`oxiblas`](../oxiblas/) - Meta-crate

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.
