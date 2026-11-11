# oxiblas-lapack

**LAPACK (Linear Algebra PACKage) decompositions and solvers for OxiBLAS**

[![Crates.io](https://img.shields.io/crates/v/oxiblas-lapack.svg)](https://crates.io/crates/oxiblas-lapack)
[![Documentation](https://docs.rs/oxiblas-lapack/badge.svg)](https://docs.rs/oxiblas-lapack)

## Overview

`oxiblas-lapack` implements matrix decompositions and linear system solvers in pure Rust. Includes LU, QR, Cholesky, SVD, EVD, Schur decomposition and more, with blocked algorithms built on the `oxiblas-blas` Level 3 kernels.

## Features

### Decompositions

#### LU Factorization
- **`Lu`** - LU with partial pivoting (PA = LU)
- **`LuRook`** - LU with rook pivoting (PAQ = LU)
- **`LuFullPiv`** - LU with full pivoting (PAQ = LU)
- **`BandLu`** - Banded LU factorization for tridiagonal/pentadiagonal systems

#### Cholesky Factorization
- **`Cholesky`** - LL^T for symmetric positive definite matrices
- **`HermitianCholesky`** - LL^H for Hermitian positive definite (complex) matrices
- **`Ldlt`** - LDL^T decomposition (no square roots)
- **`BunchKaufman`**, **`Aasen`** - Pivoted factorizations of symmetric indefinite matrices (block-diagonal D and tridiagonal T respectively)
- **`BandCholesky`** - Banded Cholesky for symmetric positive definite band matrices
- **`PackedCholesky`**, **`PackedLdlt`** - Cholesky and LDL^T for packed triangular storage

#### QR Factorization
- **`Qr`** - Standard QR decomposition
- **`QrPivot`** - QR with column pivoting
- **`Lq`** - LQ decomposition (QR variant)
- **`Rq`** - RQ decomposition
- **`Ql`** - QL decomposition
- **`CompleteOrthogonalDecomp`** - Complete orthogonal decomposition
- **`UnitaryQr`** - QR decomposition of complex matrices
- **`orgqr`**, **`ormqr`**, **`ungqr`**, **`unmqr`** - Form or apply Q from Householder reflectors

#### SVD (Singular Value Decomposition)
- **`Svd`** - Standard SVD using the one-sided Jacobi method
- **`SvdDc`** - SVD with divide-and-conquer (the LAPACK `dgesdd` scheme)
- **`QrSvd`** - Golub-Kahan-Reinsch SVD (bidiagonalization + implicit QR)
- **`ComplexSvd`**, **`ComplexSvdDc`** - SVD of complex matrices
- **`RandomizedSvd`**, **`TruncatedSvd`**, **`SelectiveSvd`** - Randomized, truncated and subset SVDs

#### Eigenvalue Decomposition
- **`SymmetricEvd`** - Symmetric eigenvalue decomposition (Householder tridiagonalization + implicit QR)
- **`SymmetricEvdDc`** - Symmetric EVD with divide-and-conquer (the LAPACK `DSYEVD` scheme)
- **`HermitianEvd`**, **`HermitianEvdDc`** - Hermitian (complex) EVD: implicit QR, and divide-and-conquer as LAPACK `ZHEEVD`
- **`TridiagEvd`**, **`MrrrEvd`** - Symmetric tridiagonal EVD (bisection and inverse iteration; MRRR)
- **`GeneralEvd`** - General eigenvalue decomposition (QR algorithm)
- **`ComplexGeneralEvd`** - Complex general EVD
- **`RandomizedEvd`** - Dominant eigenpairs of a symmetric matrix by random projection

#### Generalized Eigenvalue Problems
- **`Qz`** - QZ (generalized Schur) decomposition of a pencil (A, B), a Pure Rust equivalent of LAPACK `DGGES`: `Qz::compute`, `Qz::compute_sorted` (selected eigenvalues first), `Qz::reorder`
- **`GeneralizedEvd`** - A x = λ B x through an LU factorization of B, so B must be nonsingular; it is not a QZ decomposition
- **`SymmetricGeneralizedEvd`** - A x = λ B x for symmetric A and symmetric positive definite B (via Cholesky)

#### Other Decompositions
- **`Schur`** - Schur decomposition (A = QTQ^T)
- **`ComplexSchur`** - Complex Schur decomposition (A = QTQ^H)
- **`Hessenberg`**, **`ComplexHessenberg`** - Hessenberg reduction

### Solvers

- **`solve`** - General linear system Ax = b
- **`solve_multiple`** - Multiple right-hand sides
- **`solve_triangular`** - Triangular system solve
- **`lstsq`** - Least squares solution
- **`tridiag_solve`** - Tridiagonal system (Thomas algorithm)
- **`solve_expert`**, **`solve_cholesky_expert`**, **`solve_symmetric_expert`** - Expert drivers with equilibration, condition estimates and error bounds
- **`refine_solution`** - Iterative refinement (with `_cholesky` and `_symmetric` variants)
- **`mixed_precision_solve`** - f32 factorization with f64 refinement (also `_lu`, `_cholesky`, `_symmetric` and `_qr` variants)

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
- **Norms**: `norm_1`, `norm_2`, `norm_inf`, `norm_frobenius`, `norm_max`, `norm_nuclear`
- **Matrix functions**: `expm`, `logm`, `sqrtm`, `powm`, `sinm`, `cosm`, `signm`
- **Kronecker products**: `kron`, `kron_sum`, `khatri_rao`, `vec_mat`, and related helpers

## Installation

```toml
[dependencies]
oxiblas-lapack = "0.2"

# With parallel variants
oxiblas-lapack = { version = "0.2", features = ["parallel"] }
```

### Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `default` | Core LAPACK operations (`std` in `oxiblas-core` and `oxiblas-matrix`) | ✓ |
| `parallel` | Rayon-based variants: `Lu::compute_blocked_par`, `Cholesky::compute_blocked_par`, `SymmetricEvd::compute_with_par`, `SymmetricEvdDc::compute_with_par`, `SvdDc::compute_with_par`, `ParallelSvdDc`, `ParallelSymmetricEvd`; also enables `oxiblas-blas/parallel` | |
| `force-scalar` | Forwarded to `oxiblas-core` and `oxiblas-blas`: use the portable scalar GEMM kernel, and on aarch64 the portable DOT, NRM2, GEMV and TRSM/TRMM code | |
| `max-simd-128` | Forwarded: cap the GEMM kernel at 128-bit SIMD | |
| `max-simd-256` | Forwarded: cap the GEMM kernel at 256-bit SIMD | |

## Usage Examples

Decompositions live in their own modules (`lu`, `qr`, `cholesky`, `svd`, `evd`), solvers in `solve` and utility functions in `utils`. `oxiblas_lapack::prelude` re-exports the common ones. The dense solves take the right-hand side as a matrix (`MatRef`), so a vector is an n x 1 matrix.

### LU Decomposition

```rust
use oxiblas_lapack::lu::Lu;
use oxiblas_matrix::Mat;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Mat::from_rows(&[
        &[2.0, 1.0, 1.0],
        &[4.0, 3.0, 3.0],
        &[8.0, 7.0, 9.0],
    ]);

    // Compute LU factorization
    let lu = Lu::compute(a.as_ref())?;

    // Get L and U factors
    let l = lu.l_factor();
    let u = lu.u_factor();

    // Solve Ax = b (b is an n x 1 matrix)
    let b = Mat::from_rows(&[&[4.0], &[10.0], &[24.0]]);
    let x = lu.solve(b.as_ref())?;
    // x = [[1.0], [1.0], [1.0]]

    // Determinant
    let det = lu.determinant();  // 4.0

    // Inverse
    let inv = lu.inverse()?;
    Ok(())
}
```

`Lu::compute` is the unblocked algorithm. `Lu::compute_auto` switches to the blocked (TRSM/GEMM-based) algorithm from n = 128, and `compute_blocked`, `compute_recursive` and, with the `parallel` feature, `compute_blocked_par` select a variant explicitly. `Cholesky` and `Qr` also have `compute_blocked`, `compute_recursive` and `compute_auto` constructors.

`SymmetricEvd::compute`, `SymmetricEvdDc::compute` and `SvdDc::compute` run on the calling thread. With the `parallel` feature, their `compute_with_par(a, par)` runs the GEMMs inside them with the given `Par` and returns bitwise the same result.

### QR Decomposition

```rust
use oxiblas_lapack::qr::Qr;
use oxiblas_matrix::Mat;

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    let b = Mat::from_rows(&[&[1.0], &[2.0], &[3.0]]);
    let x = qr.solve_least_squares(b.as_ref())?;
    Ok(())
}
```

### Cholesky Decomposition

```rust
use oxiblas_lapack::cholesky::Cholesky;
use oxiblas_matrix::Mat;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Symmetric positive definite matrix
    let a = Mat::from_rows(&[
        &[4.0, 12.0, -16.0],
        &[12.0, 37.0, -43.0],
        &[-16.0, -43.0, 98.0],
    ]);

    // Compute Cholesky (LL^T)
    let chol = Cholesky::compute(a.as_ref())?;

    // Get L factor
    let l = chol.l_factor();

    // Solve Ax = b (faster than LU for SPD matrices)
    let b = Mat::from_rows(&[&[1.0], &[2.0], &[3.0]]);
    let x = chol.solve(b.as_ref())?;

    // Determinant
    let det = chol.determinant();  // 36.0
    Ok(())
}
```

### SVD (Singular Value Decomposition)

```rust
use oxiblas_lapack::svd::Svd;
use oxiblas_matrix::Mat;

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    // Pseudoinverse (singular values below the tolerance are treated as zero)
    let pinv = svd.pseudoinverse(1e-10);

    // Numerical rank
    let rank = svd.rank(1e-10);

    // Condition number
    let cond = svd.condition_number();
    Ok(())
}
```

### Eigenvalue Decomposition

```rust
use oxiblas_lapack::evd::SymmetricEvd;
use oxiblas_matrix::Mat;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Symmetric matrix
    let a = Mat::from_rows(&[
        &[4.0, 1.0, 1.0],
        &[1.0, 3.0, 2.0],
        &[1.0, 2.0, 3.0],
    ]);

    // Compute eigenvalues and eigenvectors
    let evd = SymmetricEvd::compute(a.as_ref())?;

    // Get eigenvalues (sorted ascending)
    let eigenvalues = evd.eigenvalues();  // [1.0, 3.0, 6.0]

    // Get eigenvectors
    let eigenvectors = evd.eigenvectors();

    // Reconstruct: A = V Λ V^T
    let reconstructed = evd.reconstruct();
    Ok(())
}
```

### Generalized Eigenvalue Problems

```rust
use oxiblas_lapack::evd::Qz;
use oxiblas_matrix::Mat;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 10.0]]);
    let b = Mat::from_rows(&[&[2.0f64, 1.0, 0.0], &[0.0, 3.0, 1.0], &[1.0, 0.0, 4.0]]);

    // Generalized Schur form: A = Q S Zᵀ and B = Q T Zᵀ
    let qz = Qz::compute(a.as_ref(), b.as_ref())?;
    let (s, t, q, z) = (qz.s(), qz.t(), qz.q(), qz.z());

    // Eigenvalues are (alpha_real + i*alpha_imag) / beta; beta == 0 is an infinite eigenvalue
    for e in qz.eigenvalues() {
        println!("({} + {}i) / {}", e.alpha_real, e.alpha_imag, e.beta);
    }

    // Move the selected eigenvalues (here: negative real part) to the leading positions,
    // as LAPACK DGGES does with SORT = 'S'
    let (sorted, ordering) = Qz::compute_sorted(a.as_ref(), b.as_ref(), |e| e.alpha_real < 0.0)?;
    // ordering.sdim is the number of selected eigenvalues
    Ok(())
}
```

`Qz::reorder` applies the same reordering to an existing decomposition. The reordering fails with `QzReorderError::SwapFailed` if a swap of adjacent blocks is rejected as unstable. For a singular `B`, or when you need Schur factors, use `Qz`: `GeneralizedEvd` reduces the problem to `B⁻¹A` through an LU factorization and needs a nonsingular `B`.

### Linear System Solvers

```rust
use oxiblas_lapack::solve::{solve, solve_multiple};
use oxiblas_matrix::Mat;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Mat::from_rows(&[
        &[2.0, 1.0],
        &[1.0, 3.0],
    ]);
    let b = Mat::from_rows(&[&[4.0], &[7.0]]);

    // Solve Ax = b
    let x = solve(a.as_ref(), b.as_ref())?;
    // x = [[1.0], [2.0]]

    // Multiple right-hand sides
    let b_multi = Mat::from_rows(&[
        &[4.0, 1.0],
        &[7.0, 2.0],
    ]);
    let x_multi = solve_multiple(a.as_ref(), b_multi.as_ref())?;
    Ok(())
}
```

### Matrix Utilities

```rust
use oxiblas_lapack::utils::{det, inv, rank, norm_2};
use oxiblas_matrix::Mat;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Mat::from_rows(&[
        &[1.0, 2.0],
        &[3.0, 4.0],
    ]);

    // Determinant
    let d = det(a.as_ref())?;  // -2.0

    // Inverse
    let a_inv = inv(a.as_ref())?;

    // Matrix rank (the tolerance is an Option: None uses a default)
    let r = rank(a.as_ref(), Some(1e-10))?;  // 2

    // 2-norm (spectral norm)
    let n = norm_2(a.as_ref())?;  // ≈ 5.465
    Ok(())
}
```

## Performance

Blocked algorithms against their unblocked forms on an Apple M3 on one thread, `f64`, measured on 2026-11-09. [Performance](../../README.md#performance) in the root README has the method and more sizes.

| Operation | Size | Unblocked | Blocked | Speedup |
|-----------|------|-----------|---------|---------|
| LU | 768×768 | 10.25 GFLOPS | 34.04 GFLOPS | 3.3× |
| LU | 1024×1024 | 10.70 GFLOPS | 37.91 GFLOPS | 3.5× |
| Cholesky | 1024×1024 | 11.49 GFLOPS | 40.88 GFLOPS | 3.6× |
| QR | 512×512 | 38.0 ms | 10.2 ms | 3.7× |
| QR | 1024×1024 | 306.4 ms | 58.3 ms | 5.3× |

The 0.2.3 LU and Cholesky rewrites (see [Algorithms](#algorithms)) made both forms faster, the unblocked ones most: before them, unblocked LU ran at 0.38-0.62 GFLOPS and blocked LU at 3.8-15.9, so the speedups were larger then.

On random matrices, `SvdDc` is 13-32× faster than the one-sided Jacobi `Svd`, and `SymmetricEvdDc` 1.6-2.2× faster than `SymmetricEvd`, at every size measured from n = 200 to 2000 on one thread; at n = 2000 they take 2.13 s and 0.96 s. `cargo run --release -p oxiblas --example evd_svd_bench` prints the times and accuracies of all four.

## Algorithms

### LU Decomposition
- **Partial pivoting** for numerical stability (rook and full pivoting in `LuRook` and `LuFullPiv`)
- **Unblocked** `Lu::compute`, eliminating column by column
- **Blocked right-looking algorithm** (`compute_blocked`, `compute_with_block_size`, and `Lu::compute_auto` from n = 128): each panel is factored recursively (LAPACK `?getrf2`), the block row of U is one TRSM and the trailing update one GEMM, in place; `compute_blocked_par` passes a `Par` to both
- **Recursive algorithm** (`compute_recursive`): `?getrf2` on the whole matrix, with TRSM and GEMM in place
- **Singularity test** relative to each column: a pivot is singular at `|u_jj| <= eps·n·s_j`, with `s_j` the largest finite entry of column `j` of the input, so the outcome does not change when the matrix is scaled by a power of two

### Cholesky Factorization
- **Unblocked** `Cholesky::compute`, column by column
- **Blocked algorithm** (`compute_blocked`, `compute_with_block_size`, `compute_recursive`, `compute_blocked_par`): TRSM for each block column, then GEMM on the off-diagonal blocks of the lower triangle and a triangle-only SYRK on its diagonal blocks, in place; `compute_recursive` splits the matrix recursively instead of by fixed panels
- **Positive-definiteness test** relative to the diagonal: the radicand must exceed `eps·n·a_jj` (`Ldlt`: `|d_j|` against the largest entry of the symmetric column)

### QR Decomposition
- **Householder reflections** for orthogonality
- **Blocked algorithm** for efficiency (`Qr::compute_auto`, `compute_blocked`, `compute_recursive`)
- **Column pivoting** for rank-revealing

### SVD
- **Jacobi method** - stable for small to medium matrices (`Svd`, one-sided Jacobi)
- **Divide-and-conquer** - `SvdDc`, the LAPACK `dgesdd` scheme: power-of-two scaling, blocked bidiagonalization (`dgebrd`/`dlabrd` with GEMM updates), divide and conquer on the bidiagonal ported from `dbdsdc` (`dbdsqr` QR for blocks of at most 25 rows), and `U`, `V` from compact-WY blocks of reflectors; backward stable at every scale. `ComplexSvdDc` shares the bidiagonal solver, and `ParallelSvdDc` is `SvdDc::compute_with_par(a, Par::Rayon)`
- **Two-stage approach** - Bidiagonalization + Golub-Kahan (`QrSvd`)

### Eigenvalue Decomposition
- **Tridiagonalization + implicit QR** - symmetric EVD (`SymmetricEvd`): blocked reduction, QR sweeps applied to row blocks as a wavefront, eigenvectors by GEMM
- **Divide-and-conquer** - `SymmetricEvdDc`, the LAPACK `DSYEVD` scheme (blocked `DSYTRD`, `DLAED`-style merges with GEMM, compact-WY `DORMTR`); `HermitianEvdDc` follows `ZHEEVD` on the same real divide and conquer
- **QR algorithm** - general eigenvalues (with shifts)
- **Schur decomposition** - intermediate step for general EVD
- **QZ algorithm** - generalized Schur form of a pencil (`Qz`)

## Architecture

```text
oxiblas-lapack/
├── lu/
│   ├── partial_piv.rs  # LU with partial pivoting
│   ├── rook_piv.rs     # LU with rook pivoting
│   ├── full_piv.rs     # LU with full pivoting
│   ├── band.rs         # Banded LU
│   └── tolerance.rs    # Per-column pivot tolerances (shared with Cholesky)
├── cholesky/
│   ├── llt.rs          # Cholesky decomposition
│   ├── ldlt.rs         # LDL^T decomposition
│   ├── hermitian.rs    # Hermitian Cholesky
│   ├── bunch_kaufman.rs, aasen.rs  # Symmetric indefinite
│   ├── packed.rs       # Packed storage
│   └── band.rs         # Banded Cholesky
├── qr/
│   ├── householder/    # Standard QR
│   ├── col_pivot.rs    # QR with pivoting
│   ├── lq.rs, rq.rs, ql.rs  # LQ, RQ, QL decompositions
│   ├── complete_orthogonal.rs
│   ├── unitary.rs      # Complex QR
│   └── ortho.rs        # orgqr, ormqr, ungqr, unmqr
├── svd/
│   ├── bidiagonal.rs   # Jacobi SVD (Svd)
│   ├── divide_conquer.rs  # Divide-and-conquer SVD
│   ├── qr_based.rs     # Golub-Kahan-Reinsch SVD
│   └── ...             # Complex, randomized, truncated, selective, parallel
├── evd/
│   ├── symmetric.rs    # Symmetric EVD (tridiagonalization + QR)
│   ├── symmetric_dc.rs # Symmetric EVD (D&C)
│   ├── general.rs      # General EVD (QR algorithm)
│   ├── schur.rs        # Schur decomposition
│   ├── hessenberg.rs   # Hessenberg reduction
│   ├── generalized.rs  # Generalized EVD (A x = λ B x)
│   ├── qz/             # QZ decomposition (DGGES equivalent)
│   └── ...             # Hermitian, complex, tridiagonal, MRRR, randomized, parallel
├── solve/              # Linear system solvers
├── utils/              # Determinant, inverse, norms, matrix functions, etc.
└── error.rs, info.rs, workspace.rs
```

## Examples

The examples live in the `oxiblas` meta-crate:

```bash
cargo run -p oxiblas --example lapack_decompositions
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
- **Scale-invariant pivot tests** in LU, Cholesky and LDLᵀ (relative to each column of the input), and power-of-two input scaling in the SVD and eigensolvers
- **Householder reflections** (QR) - numerically stable
- **Jacobi rotations** (one-sided Jacobi in `Svd`)
- **Divide-and-conquer** algorithms for large-scale problems

## Related Crates

- [`oxiblas-core`](../oxiblas-core/) - Core traits and SIMD
- [`oxiblas-matrix`](../oxiblas-matrix/) - Matrix types
- [`oxiblas-blas`](../oxiblas-blas/) - BLAS operations
- [`oxiblas-sparse`](../oxiblas-sparse/) - Sparse decompositions
- [`oxiblas`](../oxiblas/) - Meta-crate

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.
