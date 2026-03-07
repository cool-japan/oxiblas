# oxiblas-sparse TODO

Sparse matrix operations and solvers.

## Sparse Formats

### Current Status
- [x] CSR (Compressed Sparse Row)
- [x] CSC (Compressed Sparse Column)
- [x] COO (Coordinate)
- [x] ELL (ELLPACK) - implemented with SpMV
- [x] DIA (Diagonal) - implemented with SpMV
- [x] BSR (Block Sparse Row) - implemented with SpMV
- [x] BSC (Block Sparse Column) - column-oriented block format
- [x] HYB (Hybrid ELL+COO) - configurable width strategies
- [x] SELL (Sliced ELLPACK) - GPU-optimized with per-slice widths

### Format Operations
- [x] Efficient format conversion - all formats inter-convertible
- [x] Automatic format selection - `analyze_sparsity_pattern()`
- [x] Block detection for BSR - in sparsity analysis
- [x] Diagonal detection for DIA - in sparsity analysis

---

## Sparse BLAS

### Current Status
- [x] SpMV (sparse matrix-vector)
- [x] SpMV transpose
- [x] SpMM (sparse-dense matrix multiply)
- [x] SpGEMM - Sparse-sparse matrix multiply (A*B both sparse)
- [x] SpMV symmetric/Hermitian - `spmv_symmetric()`, `spmv_hermitian()`
- [x] SpMV triangular - `spmv_triangular()` with TriangularPart
- [x] Sparse triangular solve - `sptrsv_lower()`, `sptrsv_upper()`
- [x] Sparse rank-1 update - `rank1_update()`, `rank1_update_no_fill()`
- [x] Sparse-sparse addition (A + B) - `spadd()`
- [x] Sparse scaling - `scale()`, `scale_rows()`, `scale_cols()`, `scale_rows_cols()`
- [x] Matrix norms - `asum()`, `frobenius_norm()`, `infinity_norm()`, `one_norm()`

---

## Direct Solvers

### Current Status
- [x] Sparse Cholesky
- [x] Sparse LU (with ILU0)
- [x] Sparse QR (Givens-based)
- [x] Sparse QR (column ordering + Givens, SparseQr with least squares solve) (v0.2.0)

### Advanced Factorizations
- [x] Supernodal Cholesky (SupernodalCholesky) - BLAS-3 optimized with supernode detection
- [x] Supernodal LU (SupernodalLU) - Basic implementation with supernode structure

### Missing
- [x] Multifrontal methods - `MultifrontalCholesky`, `MultifrontalLU` (v0.2.0)
- [x] Out-of-core factorization - OutOfCoreLu, OutOfCoreCholesky with block I/O (v0.2.0)
- [x] Advanced pivoting strategies for sparse LU - threshold, static, Bunch-Kaufman LDL^T (v0.2.0)

---

## Iterative Solvers

### Current Status
- [x] CG (Conjugate Gradient)
- [x] BiCGStab
- [x] GMRES (with restart) - includes pgmres (preconditioned)
- [x] PCG (Preconditioned CG) - framework
- [x] Residual history tracking
- [x] Breakdown detection and recovery
- [x] MINRES (for symmetric indefinite) - includes pminres (preconditioned)
- [x] TFQMR (Transpose-Free Quasi-Minimal Residual)
- [x] FGMRES (Flexible GMRES) - with fgmres_ir (inner GMRES as preconditioner)
- [x] QMR (BiCG with quasi-minimization smoothing)
- [x] IDR(s) (Induced Dimension Reduction) - includes pidrs (preconditioned)
- [x] Block-CG (with Block-PCG preconditioned variant)
- [x] Block-GMRES (multiple right-hand sides)

### Features
- [x] Advanced convergence monitoring - `ConvergenceMonitor` with stagnation/divergence detection, rate estimation
- [x] Stopping criteria options - `StoppingCriteria` enum (Relative, Absolute, Mixed, RelativeResidualDecrease, EnergyNorm, And, Or)

---

## Preconditioners

### Current Status
- [x] ILU(0)
- [x] ILUT (ILU with threshold)
- [x] ILUTP (ILUT with pivoting)
- [x] IC(0) (Incomplete Cholesky)
- [x] ICT (IC with threshold)
- [x] Jacobi / Block Jacobi
- [x] Gauss-Seidel / SOR
- [x] SSOR
- [x] Polynomial preconditioners (Neumann series, Chebyshev)
- [x] AMG (Algebraic Multigrid) - classical Ruge-Stüben with V/W-cycle
- [x] SPAI (Sparse Approximate Inverse) - least-squares column computation
- [x] AINV (Approximate Inverse) - factored sparse approximate inverse
- [x] Additive Schwarz (Domain decomposition) - overlapping subdomains

### SA-AMG
- [x] Smoothed Aggregation AMG - `SAMG` with aggregation-based coarsening and smoothed prolongators

---

## Sparse Eigenvalue

### Current Status
- [x] Lanczos iteration (symmetric) - with full reorthogonalization
- [x] Arnoldi iteration (general) - with QR-based Hessenberg EVD
- [x] Shift-and-invert spectral transformation
- [x] Implicit restart (IRAM)
- [x] Block Lanczos
- [x] Block Arnoldi
- [x] Polynomial filtering - PolynomialFilteredLanczos with Chebyshev polynomial
- [x] Thick-restart Lanczos (TRL) - superior convergence vs basic Lanczos, Wu-Simon algorithm (v0.2.0)
- [x] LOBPCG (Locally Optimal Block Preconditioned CG) - Knyazev 2001, preconditioned block iteration (v0.2.0)

### Features
- [x] K largest/smallest eigenvalues
- [x] Largest/smallest algebraic eigenvalues
- [x] Eigenvalues in interval - IntervalEigen with Sturm sequence
- [x] Interior eigenvalues (shift-invert)
- [x] Generalized eigenvalue (A*x = λ*B*x)

---

## Sparse SVD

### Current Status
- [x] Truncated SVD (k largest singular values)
- [x] Lanczos-based SVD
- [x] Randomized sparse SVD - `RandomizedSparseSvd` with power iteration
- [x] Incremental SVD - `IncrementalSVD` with Brand algorithm

---

## Ordering and Symbolic

### Current Status
- [x] Elimination tree
- [x] Basic symbolic analysis
- [x] RCM (Reverse Cuthill-McKee)
- [x] AMD (Approximate Minimum Degree)
- [x] Nested dissection (level-set based)

### Missing
- [x] MMD (Multiple Minimum Degree) - `multiple_minimum_degree()` with mass elimination
- [x] COLAMD (column AMD) - `colamd()`, `colamd_aggressive()` for unsymmetric/rectangular
- [x] Inverse permutation - `inverse_permutation()` utility
- [x] Fill-reducing orderings comparison - `compare_orderings()`, `select_best_ordering()`
- [x] Fill-in estimation - `estimate_fill_in()`, `fill_in_ratio()`
- [x] Bandwidth/profile metrics - `bandwidth_with_ordering()`, `profile_with_ordering()`
- [x] OrderingAlgorithm enum - unified interface for AMD, MMD, RCM, NestedDissection, Natural
- [x] METIS-equivalent pure Rust multilevel nested dissection (multilevel.rs)
- [ ] Scotch interface

---

## Graph Operations

- [x] Connected components
- [x] Bipartite matching
- [x] Graph partitioning - Kernighan-Lin style bisection
- [x] Level set construction
- [x] Bandwidth/profile computation
- [x] Degree sequence
- [x] Structural symmetry check
- [x] Pseudo-peripheral vertex

---

## I/O and Testing

### Current Status
- [x] Matrix Market format support - read/write for real/pattern/symmetric

### Missing
- [x] Standard test matrices - `test_matrices` module with 7 generators (v0.2.0)
- [x] Unit tests (passing)
- [x] Convergence tests for iterative solvers - CG, PCG, BiCGStab, GMRES, PGMRES, MINRES, TFQMR, QMR, IDR(s), FGMRES
- [x] Preconditioner effectiveness tests - Jacobi, BlockJacobi, GaussSeidel, SOR, SSOR, IC0, ICT, ILU0, ILUT, AMG, SPAI, AINV, AdditiveSchwarz
- [x] Memory usage tests - 27 integration tests (v0.2.0)
- [x] Ordering quality tests - fill-in estimation, bandwidth, profile metrics
- [x] Iterative solvers refactored into 13 modules (iterative/)

---

## Stochastic Methods

- [x] Hutchinson trace estimator - unbiased, O(m * nnz) (v0.2.0)
- [x] Hutch++ trace estimator - improved variance reduction (v0.2.0)
- [x] XTrace estimator - minimum variance stochastic trace (v0.2.0)
- [x] Diagonal estimator - Bekas et al., per-element diagonal estimate (v0.2.0)
- [x] Stochastic Frobenius norm estimate (v0.2.0)
- [x] Stochastic log-determinant via Lanczos quadrature (v0.2.0)

---

## Performance

### Optimizations
- [x] SIMD SpMV kernels - `spmv_f64_simd()`, `spmv_f32_simd()` with multi-way accumulation
- [x] Parallel SpMV - `spmv_f64_par()`, `spmv_f32_par()` with Rayon (requires "parallel" feature)
- [x] Automatic SpMV dispatch - `spmv_auto()` selects best implementation by type
- [x] Parallel SpGEMM - `spmm_sparse_par()` with row-parallel computation, `spmm_sparse_auto()` dispatch
- [x] Cache-blocked SpMV - `spmv_f64_blocked()`, `spmv_f32_blocked()` with 256-row blocks
- [x] Hybrid SpMV - `spmv_f64_hybrid()` combines blocking with parallelism for very large matrices

### Targets

| Operation | Size | Current | Target |
|-----------|------|---------|--------|
| SpMV CSR | 100K x 100K, 1M nnz | - | 80% MKL Sparse |
| SpMM | 10K x 10K, 100K nnz | - | 75% MKL Sparse |
| CG (100 iter) | 10K | - | 70% PETSc |
| ILU(0) setup | 10K, 100K nnz | - | 70% ITSOL |
