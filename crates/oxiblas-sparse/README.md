# oxiblas-sparse

**Sparse matrix operations, iterative solvers, and preconditioners for OxiBLAS**

[![Crates.io](https://img.shields.io/crates/v/oxiblas-sparse.svg)](https://crates.io/crates/oxiblas-sparse)
[![Documentation](https://docs.rs/oxiblas-sparse/badge.svg)](https://docs.rs/oxiblas-sparse)

## Overview

`oxiblas-sparse` provides comprehensive sparse linear algebra functionality including 9 storage formats, iterative solvers, advanced preconditioners, and sparse eigenvalue/SVD algorithms.

## Features

### Sparse Matrix Formats (9 types)

- **CSR** (Compressed Sparse Row) - General sparse matrices
- **CSC** (Compressed Sparse Column) - Efficient column operations
- **COO** (Coordinate) - Easy construction and modification
- **ELL** (ELLPACK) - GPU-friendly format
- **DIA** (Diagonal) - Band-diagonal matrices
- **BSR** (Block Sparse Row) - Block-structured matrices
- **BSC** (Block Sparse Column) - Block column format
- **HYB** (Hybrid ELL+COO) - Adaptive format
- **SELL** (Sliced ELLPACK) - Optimized for GPUs

### Iterative Solvers (10+ methods)

#### Krylov Subspace Methods
- **GMRES** - Generalized Minimal Residual (with restart)
- **FGMRES** - Flexible GMRES (variable preconditioning)
- **PGMRES** - Preconditioned GMRES
- **CG** - Conjugate Gradient (symmetric positive definite)
- **PCG** - Preconditioned CG
- **Block-CG** - Multiple right-hand sides
- **BiCGStab** - BiConjugate Gradient Stabilized
- **MINRES** - Minimal Residual (symmetric)
- **PMINRES** - Preconditioned MINRES
- **QMR** - Quasi-Minimal Residual
- **TFQMR** - Transpose-Free QMR
- **IDR(s)** - Induced Dimension Reduction
- **Block-GMRES** - Multiple RHS

#### Eigenvalue Methods
- **Lanczos** - Symmetric matrices
- **Block Lanczos** - Multiple eigenvalues
- **Arnoldi** - General matrices
- **Block Arnoldi** - Multiple eigenvalues
- **IRAM** - Implicitly Restarted Arnoldi Method

### Advanced Preconditioners (12+ types)

#### Incomplete Factorizations
- **ILU(0)** - Zero-fill incomplete LU
- **ILUT** - ILU with threshold
- **ILUTP** - ILUT with pivoting
- **IC0** - Incomplete Cholesky (zero-fill)
- **ICT** - IC with threshold

#### Iterative Preconditioners
- **Jacobi** - Diagonal scaling
- **Block Jacobi** - Block diagonal
- **Gauss-Seidel** - Forward/backward sweeps
- **SOR** - Successive Over-Relaxation
- **SSOR** - Symmetric SOR

#### Multigrid Methods
- **AMG** - Algebraic Multigrid (classical Ruge-Stüben)
- **SA-AMG** - Smoothed Aggregation AMG

#### Approximate Inverse
- **SPAI** - Sparse Approximate Inverse
- **AINV** - Approximate Inverse

#### Domain Decomposition
- **Additive Schwarz** - Domain decomposition
- **Polynomial** - Neumann/Chebyshev polynomial preconditioners

### Sparse Eigenvalue & SVD

- **Lanczos method** (symmetric)
- **Arnoldi iteration** (general)
- **Shift-invert** spectral transformation
- **Polynomial filtering** (Chebyshev)
- **Interval eigenvalue** computation (Sturm sequence)
- **Truncated SVD**
- **Randomized SVD**
- **Incremental SVD** (Brand algorithm)

### Matrix Reordering

- **RCM** - Reverse Cuthill-McKee (bandwidth reduction)
- **AMD** - Approximate Minimum Degree
- **MMD** - Multiple Minimum Degree
- **COLAMD** - Column AMD (for unsymmetric matrices)
- **Nested Dissection** - Level-set based ordering

## Installation

```toml
[dependencies]
oxiblas-sparse = "0.2"
```

## Usage Examples

### Creating Sparse Matrices

```rust
use oxiblas_sparse::{CsrMatrix, CooMatrix};

// From COO (easy construction)
let mut coo = CooMatrix::<f64>::new(1000, 1000);
coo.push(0, 0, 4.0);
coo.push(0, 1, -1.0);
coo.push(1, 0, -1.0);
coo.push(1, 1, 4.0);
// ... add more elements

// Convert to CSR (efficient operations)
let csr = CsrMatrix::from_coo(&coo);

// Matrix-vector multiplication
let x = vec![1.0; 1000];
let y = csr.matvec(&x);
```

### Iterative Solvers

```rust
use oxiblas_sparse::{CsrMatrix, gmres};

// Solve Ax = b using GMRES
let a = CsrMatrix::from_...;
let b = vec![/*...*/];

let result = gmres(&a, &b,
    1e-10,       // tolerance
    100,         // max iterations
    30,          // restart
    None,        // no preconditioner
)?;

println!("Solution: {:?}", result.x);
println!("Residual: {}", result.residual);
println!("Iterations: {}", result.iterations);
```

### Preconditioned Solvers

```rust
use oxiblas_sparse::{CsrMatrix, IluPreconditioner, pcg};

let a = CsrMatrix::from_...;
let b = vec![/*...*/];

// Create ILU preconditioner
let ilu = IluPreconditioner::compute(&a, 0.01)?; // drop tolerance

// Solve with preconditioned CG
let result = pcg(&a, &b,
    &ilu,        // preconditioner
    1e-10,       // tolerance
    1000,        // max iterations
)?;
```

### AMG Preconditioner

```rust
use oxiblas_sparse::{CsrMatrix, AmgPreconditioner, pcg};

let a = CsrMatrix::from_...;
let b = vec![/*...*/];

// Create AMG preconditioner (multilevel)
let amg = AmgPreconditioner::build(&a, AmgConfig {
    max_levels: 10,
    coarsening: CoarseningType::RugeStuben,
    smoother: SmootherType::GaussSeidel,
    ..Default::default()
})?;

// Solve with AMG-preconditioned CG
let result = pcg(&a, &b, &amg, 1e-10, 1000)?;
```

### Sparse Eigenvalues

```rust
use oxiblas_sparse::{CsrMatrix, lanczos};

let a = CsrMatrix::from_...;

// Compute largest eigenvalues using Lanczos
let result = lanczos(&a,
    10,          // number of eigenvalues
    100,         // max iterations
    1e-10,       // tolerance
)?;

println!("Eigenvalues: {:?}", result.eigenvalues);
println!("Eigenvectors: {:?}", result.eigenvectors);
```

### Matrix Reordering

```rust
use oxiblas_sparse::{CsrMatrix, rcm_ordering};

let a = CsrMatrix::from_...;

// Reverse Cuthill-McKee ordering (bandwidth reduction)
let perm = rcm_ordering(&a)?;

// Reorder matrix: P*A*P^T
let a_reordered = a.permute(&perm, &perm)?;

// Significant bandwidth reduction for better cache performance!
```

## Performance

### Solver Convergence

| Matrix Type | Solver | Preconditioner | Iterations |
|-------------|--------|----------------|-----------|
| Poisson 2D (10K unknowns) | CG | None | ~7000 |
| Poisson 2D (10K unknowns) | CG | ILU(0) | ~150 |
| Poisson 2D (10K unknowns) | CG | AMG | ~15 |
| General (sparse) | GMRES(30) | None | ~500 |
| General (sparse) | FGMRES(30) | ILUT | ~80 |

### Memory Efficiency

| Format | Non-zeros | Memory (vs Dense) |
|--------|-----------|-------------------|
| CSR | 1M | 0.001% |
| COO | 1M | 0.0015% |
| Dense equivalent | 1M | 100% |

For a 1000×1000 matrix with 1M non-zeros, sparse format uses **~1000× less memory**!

## Algorithms

### GMRES
- **Arnoldi process** for Krylov subspace
- **Restart** mechanism for large problems
- **Flexible** variant for variable preconditioning

### CG (Conjugate Gradient)
- **Three-term recurrence** - memory efficient
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

```
oxiblas-sparse/
├── formats/
│   ├── csr.rs         # CSR format
│   ├── csc.rs         # CSC format
│   ├── coo.rs         # COO format
│   ├── ell.rs         # ELL format
│   ├── dia.rs         # DIA format
│   └── ...
├── solvers/
│   ├── gmres.rs       # GMRES solver
│   ├── cg.rs          # Conjugate Gradient
│   ├── bicgstab.rs    # BiCGStab
│   ├── minres.rs      # MINRES
│   └── ...
├── precond/
│   ├── ilu.rs         # Incomplete LU
│   ├── ic.rs          # Incomplete Cholesky
│   ├── amg.rs         # Algebraic Multigrid
│   ├── spai.rs        # Sparse Approximate Inverse
│   └── ...
├── eigen/
│   ├── lanczos.rs     # Lanczos method
│   ├── arnoldi.rs     # Arnoldi iteration
│   └── ...
└── reorder/
    ├── rcm.rs         # Reverse Cuthill-McKee
    ├── amd.rs         # Approximate Minimum Degree
    └── ...
```

## Examples

```bash
cargo run --example sparse_matrices
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
