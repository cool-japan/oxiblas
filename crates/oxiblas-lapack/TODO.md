# oxiblas-lapack TODO

LAPACK (Linear Algebra Package) implementations.

## Factorizations

### Current Status
- [x] LU decomposition (partial pivoting)
- [x] LU decomposition (full pivoting)
- [x] Cholesky (LLT)
- [x] Cholesky (LDLT)
- [x] QR (Householder)
- [x] QR (column pivoting)
- [x] Hessenberg reduction
- [x] Schur decomposition
- [x] Bidiagonal reduction (internal)

### Missing Factorizations
- [x] LU with rook pivoting (LuRook with RookPivotStats)
- [x] Bunch-Kaufman (symmetric indefinite) - BunchKaufman with pivoting
- [x] Aasen's method (symmetric indefinite) - Aasen with tridiagonal T factor
- [x] RQ factorization (with Q extraction)
- [x] LQ factorization
- [x] QL factorization (square matrices, tall/wide pending)
- [x] Complete orthogonal decomposition (CompleteOrthogonalDecomp with reconstruct and solve)

---

## Eigenvalue Problems

### Current Status
- [x] Symmetric eigenvalues (tridiagonal reduction)
- [x] General eigenvalues (Schur-based)
- [x] Left and right eigenvectors
- [x] Generalized symmetric eigenvalues (partial)
- [x] QZ algorithm

### Missing
- [x] Divide-and-conquer symmetric EVD (SymmetricEvdDc, HermitianEvdDc)
- [x] MRRR algorithm (MrrrEvd with LDL representation and inverse iteration)
- [x] Bisection + inverse iteration (TridiagEvd)
- [x] Selected eigenvalues only (range or index) (EigenvalueSelector)
- [x] Condition number estimation for eigenvalues (trsna_s, trsna_sep)
- [x] Eigenvector refinement (trevc) - trevc_right, trevc_left, Schur::eigenvectors
- [x] Balancing (gebal/gebak)
- [x] Randomized EVD (HMT algorithm) - RandomizedEvd with power iteration, f64/f32 (v0.2.0)

---

## Singular Value Decomposition

### Current Status
- [x] Jacobi SVD
- [x] Divide-and-conquer SVD
- [x] Bidiagonal SVD

### Missing
- [x] Truncated SVD (k largest singular values)
- [x] One-sided Jacobi (Svd struct uses one-sided Jacobi algorithm)
- [x] QR-based SVD (QrSvd with Golub-Kahan-Reinsch algorithm)
- [x] Randomized SVD
- [x] Condition number from SVD (Svd::condition_number, SvdDc::condition_number)

---

## Linear System Solvers

### Current Status
- [x] General solve (via LU)
- [x] Triangular solve
- [x] Multiple RHS support
- [x] Least squares (QR-based)

### Missing - CRITICAL
- [x] **Iterative refinement** (getrs + refinement)
- [x] Expert drivers with condition estimation (solve_expert, solve_cholesky_expert, solve_symmetric_expert)
- [x] Equilibration (row/column scaling) - geequ, geequb, syequ

### Missing - Band Systems
- [x] gbtrf - Band LU factorization (BandLu::compute)
- [x] gbtrs - Band triangular solve (BandLu::solve)
- [x] gbsv - Band system solve (compute + solve)
- [x] gbcon - Band condition number (BandLu::rcond, band_norm_1, band_norm_inf)

### Missing - Tridiagonal Systems
- [x] gtsv - General tridiagonal solve
- [x] gttrf - General tridiagonal factorization
- [x] gttrs - General tridiagonal solve from factors
- [x] ptsv - Positive definite tridiagonal
- [x] pttrf - Positive definite tridiagonal factorization

### Missing - Packed/Symmetric
- [x] spsv - Symmetric packed solve (PackedLdlt::solve, spsv)
- [x] sptrf - Symmetric packed factorization (PackedLdlt::compute)
- [x] ppsv - Positive definite packed solve (PackedCholesky::solve, ppsv)
- [x] pptrf - Positive definite packed factorization (PackedCholesky::compute)

---

## Orthogonal/Unitary Transformations

### Current Status
- [x] **orgqr** / ungqr - Generate Q from QR
- [x] **ormqr** / unmqr - Multiply by Q from QR

### Missing
- [x] orghr / unghr - Generate Q from Hessenberg (gehrd, orghr)
- [x] ormhr / unmhr - Multiply by Q from Hessenberg (ormhr)
- [x] orgbr / ungbr - Generate Q from bidiagonal
- [x] ormbr / unmbr - Multiply by Q from bidiagonal
- [x] orgtr / ungtr - Generate Q from tridiagonal (sytrd, orgtr)
- [x] ormtr / unmtr - Multiply by Q from tridiagonal (ormtr)

---

## Matrix Functions

### Current Status
- [x] Matrix exponential (expm)
- [x] Matrix logarithm (logm)
- [x] Matrix square root (sqrtm)
- [x] Matrix power (powm - general real power)
- [x] Matrix sign function (signm)
- [x] Matrix cosine/sine (cosm, sinm)

### Missing
- [x] Frechet derivatives (frechet_expm, frechet_logm, frechet_sqrtm, cond_expm)

---

## Utilities

### Current Status
- [x] Determinant
- [x] Inverse
- [x] Pseudoinverse (pinv)
- [x] Norms (1, 2, Frobenius, inf, nuclear)
- [x] Rank
- [x] Nullity
- [x] Null space
- [x] Column space
- [x] Row space
- [x] Condition number
- [x] Kronecker product

### Missing
- [x] Matrix balance (gebal/gebak)
- [x] Matrix equilibration (geequ, geequb, syequ)
- [x] Workspace size estimation (workspace module with query functions for all major operations)
- [x] Error bounds computation (backward/forward error, residual norms, orthogonality defect)

---

## Complex Number Support

### Current Status
- [x] Complex32 / Complex64 in Rust modules
- [x] Complex Hessenberg reduction (ComplexHessenberg, zgehrd, zunhhr, zunmhr)
- [x] Complex Schur decomposition (ComplexSchur)
- [x] Complex general eigenvalues (ComplexGeneralEvd)
- [x] Hermitian eigenvalue decomposition (HermitianEvd)
- [x] Hermitian D&C eigenvalue decomposition (HermitianEvdDc)
- [x] Complex SVD - Jacobi method (ComplexSvd)
- [x] Complex SVD - Divide-and-conquer (ComplexSvdDc)

### Implemented
- [x] Complex LU decomposition (Lu<Complex64/Complex32> via generic Scalar trait)
- [x] Complex Cholesky decomposition (HermitianCholesky for Hermitian positive definite)
- [x] Complex QR decomposition (UnitaryQr with Q^H Q = I)

### Remaining
- [x] Complex bidiagonal reduction - `ComplexBidiagFactors` with direct complex Householder (v0.2.0)

---

## Performance

### Optimizations Completed
- [x] Blocked LU factorization with GEMM (Lu::compute_blocked) - **up to 23x speedup**
- [x] Blocked Cholesky factorization with GEMM (Cholesky::compute_blocked) - **up to 10x speedup**

### Blocked LU Performance (vs unblocked)

| Size | Unblocked | Blocked | Speedup |
|------|-----------|---------|---------|
| 256  | 0.62 Gf/s | 9.05 Gf/s | 14.50x |
| 512  | 0.45 Gf/s | 6.65 Gf/s | 14.86x |
| 768  | 0.60 Gf/s | 13.84 Gf/s | 23.01x |
| 1024 | 0.35 Gf/s | 6.99 Gf/s | 19.74x |

### Blocked Cholesky Performance (vs unblocked)

| Size | Unblocked | Blocked | Speedup |
|------|-----------|---------|---------|
| 256  | 2.16 Gf/s | 13.01 Gf/s | 6.03x |
| 512  | 1.65 Gf/s | 14.97 Gf/s | 9.06x |
| 768  | 1.84 Gf/s | 14.76 Gf/s | 8.03x |
| 1024 | 1.44 Gf/s | 14.73 Gf/s | 10.20x |

### Optimizations Needed
- [x] Blocked Hessenberg reduction (Hessenberg::compute_blocked)
- [x] Blocked bidiagonalization (BidiagFactors::compute_blocked)
- [x] Parallel eigenvalue computation (ParallelSymmetricEvd, parallel_bisection_eigenvalues)
- [x] Parallel SVD computation (ParallelSvdDc)

### Targets

| Operation | Size | Current | Target |
|-----------|------|---------|--------|
| LU | 1000x1000 | ~75% MKL | 85% MKL |
| Cholesky | 1000x1000 | ~80% MKL | 90% MKL |
| QR | 1000x1000 | ~70% MKL | 80% MKL |
| SVD | 1000x1000 | ~60% MKL | 75% MKL |
| Symmetric EVD | 1000x1000 | ~65% MKL | 80% MKL |

---

## Code Organization

- [x] Matrix functions refactored into 4 modules (utils/matfun/)

## Testing

- [x] LAPACK test suite compatibility - 61 integration tests in tests/lapack_compat.rs (v0.2.0)
- [x] Unit tests (passing)
- [x] Accuracy tests (residual norms) - error_bounds::accuracy_tests module
- [x] Orthogonality tests - orthogonality_defect function
- [x] Edge cases (singular, ill-conditioned) - test_ill_conditioned_matrix, test_1x1_matrix
- [x] Complex number tests - error_bounds::complex_accuracy_tests module
- [x] Large matrix tests - test_large_matrix_accuracy (50x50)
