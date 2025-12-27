# oxiblas-ffi TODO

C-compatible Foreign Function Interface for OxiBLAS.

## BLAS Level 1 FFI

### Current Status
- [x] sdot, ddot
- [x] snrm2, dnrm2
- [x] sasum, dasum
- [x] isamax, idamax
- [x] sscal, dscal
- [x] saxpy, daxpy
- [x] scopy, dcopy
- [x] sswap, dswap
- [x] srotg, drotg
- [x] srot, drot
- [x] srotm, drotm
- [x] srotmg, drotmg

### Complex Operations (Completed)
- [x] cdotu, zdotu - Complex dot product (unconjugated)
- [x] cdotc, zdotc - Complex conjugate dot product
- [x] scnrm2, dznrm2 - Complex norm
- [x] scasum, dzasum - Complex absolute sum
- [x] icamax, izamax - Complex max index
- [x] cscal, zscal - Complex scale by complex scalar
- [x] csscal, zdscal - Complex scale by real scalar
- [x] caxpy, zaxpy - Complex axpy
- [x] ccopy, zcopy - Complex copy
- [x] cswap, zswap - Complex swap
- [x] crotg, zrotg - Complex Givens rotation generation
- [x] csrot, zdrot - Real Givens rotation on complex vectors
- [x] crot, zrot - Complex Givens rotation application

---

## BLAS Level 2 FFI

### Current Status
- [x] sgemv, dgemv
- [x] cgemv, zgemv - Complex GEMV
- [x] ssymv, dsymv - Symmetric matrix-vector
- [x] chemv, zhemv - Hermitian matrix-vector
- [x] ssyr, dsyr - Symmetric rank-1
- [x] cher, zher - Hermitian rank-1
- [x] ssyr2, dsyr2 - Symmetric rank-2
- [x] cher2, zher2 - Hermitian rank-2
- [x] strmv, dtrmv, ctrmv, ztrmv - Triangular matrix-vector
- [x] strsv, dtrsv, ctrsv, ztrsv - Triangular solve
- [x] sger, dger - General rank-1 update
- [x] cgeru, zgeru - Complex rank-1 update (unconjugated)
- [x] cgerc, zgerc - Complex rank-1 update (conjugated)

### Banded Operations (Completed)
- [x] sgbmv, dgbmv, cgbmv, zgbmv - General banded
- [x] ssbmv, dsbmv - Symmetric banded
- [x] chbmv, zhbmv - Hermitian banded
- [x] stbmv, dtbmv, ctbmv, ztbmv - Triangular banded
- [x] stbsv, dtbsv, ctbsv, ztbsv - Triangular banded solve

### Packed Operations (Completed)
- [x] sspmv, dspmv - Symmetric packed
- [x] chpmv, zhpmv - Hermitian packed
- [x] stpmv, dtpmv, ctpmv, ztpmv - Triangular packed
- [x] stpsv, dtpsv, ctpsv, ztpsv - Triangular packed solve
- [x] sspr, dspr - Symmetric packed rank-1
- [x] chpr, zhpr - Hermitian packed rank-1
- [x] sspr2, dspr2 - Symmetric packed rank-2
- [x] chpr2, zhpr2 - Hermitian packed rank-2

---

## BLAS Level 3 FFI (Complete)

### All Operations Implemented
- [x] sgemm, dgemm, cgemm, zgemm - General matrix multiply
- [x] strsm, dtrsm, ctrsm, ztrsm - Triangular solve
- [x] strmm, dtrmm, ctrmm, ztrmm - Triangular multiply
- [x] ssymm, dsymm, csymm, zsymm - Symmetric multiply
- [x] chemm, zhemm - Hermitian multiply
- [x] ssyrk, dsyrk, csyrk, zsyrk - Symmetric rank-k
- [x] cherk, zherk - Hermitian rank-k
- [x] ssyr2k, dsyr2k, csyr2k, zsyr2k - Symmetric rank-2k
- [x] cher2k, zher2k - Hermitian rank-2k

---

## LAPACK FFI

### Current Status
- [x] sgetrf, dgetrf - LU factorization
- [x] zgetrf - Complex LU factorization
- [x] spotrf, dpotrf - Cholesky factorization
- [x] sgeqrf, dgeqrf - QR factorization
- [x] sgesvd, dgesvd - SVD
- [x] dsyev - Symmetric eigenvalues
- [x] dgeev - General eigenvalues (with left eigenvectors)
- [x] sgesv, dgesv - Linear system solve

### Solve and Condition (Completed)
- [x] **sgetrs, dgetrs** - Solve from LU factors
- [x] **spotrs, dpotrs** - Solve from Cholesky factors
- [x] **sgecon, dgecon** - Condition number estimation
- [x] **sinv, dinv** - Direct matrix inverse
- [x] **sgetri, dgetri** - Matrix inverse from LU factors
- [x] **spotri, dpotri** - Matrix inverse from Cholesky factors
- [x] **sgels, dgels** - Least squares solve

### Matrix Norms (Complete)
- [x] **slange, dlange** - General matrix norm (1, inf, Frobenius, max)
- [x] **slansy, dlansy** - Symmetric matrix norm
- [x] **slantr, dlantr** - Triangular matrix norm
- [x] **snorm2, dnorm2** - Spectral norm (2-norm via SVD)

### Determinant (Complete)
- [x] **sdet, ddet** - Matrix determinant
- [x] **sabsdet, dabsdet** - Absolute value of determinant
- [x] **sdetlu, ddetlu** - Determinant from LU factors
- [x] **slogdet, dlogdet** - Log-determinant (sign and magnitude)
- [x] **sdet_chol, ddet_chol** - Determinant via Cholesky (for SPD matrices)
- [x] **slogdet_chol, dlogdet_chol** - Log-determinant via Cholesky (for SPD matrices)

### Pseudoinverse (Complete)
- [x] **spinv, dpinv** - Moore-Penrose pseudoinverse
- [x] **spinv_default, dpinv_default** - Pseudoinverse with automatic tolerance

### Matrix Functions (Complete)
- [x] **sexpm, dexpm** - Matrix exponential (Padé approximation)
- [x] **slogm, dlogm** - Matrix logarithm
- [x] **ssqrtm, dsqrtm** - Matrix square root (Denman-Beavers)
- [x] **spowm, dpowm** - Matrix power (binary exponentiation)

### Condition Numbers (Complete)
- [x] **scond, dcond** - 2-norm condition number (via SVD)
- [x] **scond1, dcond1** - 1-norm condition number
- [x] **scondinf, dcondinf** - Infinity-norm condition number
- [x] **srcond, drcond** - Reciprocal condition number
- [x] **srcond_est, drcond_est** - Fast reciprocal condition estimate (Hager-Higham)

### Matrix Rank (Complete)
- [x] **srank, drank** - Matrix rank (via SVD)
- [x] **snullity, dnullity** - Matrix nullity

### Trace and Nuclear Norm (Complete)
- [x] **strace, dtrace** - Matrix trace
- [x] **snormnuc, dnormnuc** - Nuclear norm (trace norm)

### Kronecker Product (Complete)
- [x] **skron, dkron** - Kronecker product
- [x] **skronsum, dkronsum** - Kronecker sum
- [x] **skhatri_rao, dkhatri_rao** - Khatri-Rao product (column-wise Kronecker)
- [x] **svec, dvec** - Vectorize matrix (stack columns)
- [x] **sunvec, dunvec** - Reshape vector to matrix (inverse of vec)
- [x] **scommutation, dcommutation** - Commutation matrix
- [x] **sduplication, dduplication** - Duplication matrix
- [x] **selimination, delimination** - Elimination matrix
- [x] **skron_vec, dkron_vec** - Efficient Kronecker-vector product

### Schur Decomposition (Complete)
- [x] **sgees, dgees** - Schur decomposition A = Q T Q^T

### LDL^T Factorization (Complete)
- [x] **ssytrf, dsytrf** - LDL^T factorization for symmetric matrices
- [x] **ssytrs, dsytrs** - Solve from LDL^T factors
- [x] **sinertia, dinertia** - Matrix inertia (eigenvalue sign counts)
- [x] **sisposdef, disposdef** - Positive definiteness check
- [x] **sisnegdef, disnegdef** - Negative definiteness check
- [x] **slogabsdet_ldlt, dlogabsdet_ldlt** - Log absolute determinant via LDLT

### Hessenberg Reduction (Complete)
- [x] **sgehrd, dgehrd** - Reduce to upper Hessenberg form A = Q H Q^T

### Generalized EVD (Complete)
- [x] **ssygv, dsygv** - Symmetric generalized EVD (B positive definite)
- [x] **sggev, dggev** - General generalized EVD

### QZ Decomposition (Complete)
- [x] **sgges, dgges** - Generalized Schur decomposition

### QR with Column Pivoting (Complete)
- [x] **sgeqp3, dgeqp3** - Rank-revealing QR factorization

### LU with Full Pivoting (Complete)
- [x] **sgetc2, dgetc2** - LU with complete pivoting (PAQ = LU)
- [x] **sgesc2, dgesc2** - Solve from LU full pivoting factors

### Subspace Computations (Complete)
- [x] **snull, dnull** - Null space basis
- [x] **scolspace, dcolspace** - Column space basis
- [x] **srowspace, drowspace** - Row space basis
- [x] **slnull, dlnull** - Left null space basis

### SVD Divide-and-Conquer (Complete)
- [x] **sgesdd, dgesdd** - SVD using divide-and-conquer algorithm

### Thin SVD (Complete)
- [x] **sgesvd_thin, dgesvd_thin** - Thin (economy) SVD
- [x] **sgesdd_thin, dgesdd_thin** - Thin SVD using divide-and-conquer

### Iterative Refinement (Complete)
- [x] **sgerfs, dgerfs** - General system iterative refinement
- [x] **sporfs, dporfs** - SPD system iterative refinement (Cholesky)
- [x] **ssyrfs, dsyrfs** - Symmetric system iterative refinement (LDL^T)

### Complex Operations (Partial)
- [x] cgetrf, zgetrf - Complex LU factorization
- [x] cgesv, zgesv - Complex linear solve
- [x] cpotrf, zpotrf - Complex Cholesky (Hermitian positive definite)
- [x] cgeqrf, zgeqrf - Complex QR (unitary Q)
- [x] cgesvd, zgesvd - Complex SVD

### Eigenvalue (Partial)
- [x] ssyev - Single precision symmetric EVD
- [x] sgeev - Single precision general EVD
- [x] ssyevd, dsyevd - Divide-and-conquer symmetric EVD
- [x] cheev, zheev - Hermitian EVD
- [x] cgeev, zgeev - Complex general EVD
- [x] cheevd, zheevd - Divide-and-conquer Hermitian EVD

### Orthogonal/Unitary (Complete)
- [x] sorgqr, dorgqr - Generate Q from QR
- [x] sormqr, dormqr - Multiply by Q
- [x] cungqr, zungqr - Generate unitary Q from QR (complex)
- [x] cunmqr, zunmqr - Multiply by unitary Q (complex)

### Band Matrix Operations (Complete)
- [x] **sgbtrf, dgbtrf** - Band LU factorization
- [x] **sgbtrs, dgbtrs** - Band solve from LU factors
- [x] **sgbsv, dgbsv** - Band system solve
- [x] **spbtrf, dpbtrf** - Band Cholesky factorization (SPD)
- [x] **spbtrs, dpbtrs** - Band Cholesky solve
- [x] **spbsv, dpbsv** - Direct band SPD system solve

### Tridiagonal (Complete)
- [x] **sgtsv, dgtsv** - Tridiagonal solve (Thomas algorithm)
- [x] **sgttrf, dgttrf** - Tridiagonal LU factorization
- [x] **sgttrs, dgttrs** - Tridiagonal solve from factors
- [x] **sptsv, dptsv** - SPD tridiagonal solve (LDL^T)

### Expert Drivers (Complete)
- [x] sgesvx, dgesvx - Expert general solve (with equilibration, condition, error bounds)
- [x] sposvx, dposvx - Expert Cholesky solve (with equilibration, condition, error bounds)
- [x] ssysvx, dsysvx - Expert symmetric solve (with equilibration, condition, error bounds, inertia)

---

## API Design

### Compatibility
- [ ] CBLAS-compatible function signatures
- [ ] LAPACKE-compatible function signatures
- [ ] Thread-safety guarantees
- [ ] Error handling standardization

### Extensions
- [ ] Batch operations (multiple small matrices)
- [ ] Strided batch operations
- [ ] Custom memory allocation hooks
- [ ] Progress callbacks for long operations

---

## Build System

- [x] C header generation (cbindgen) - include/oxiblas.h with BLAS 1/2/3 and LAPACK
- [x] Build script (build.rs) for header generation
- [ ] pkg-config support
- [ ] CMake find module
- [x] Static library option (crate-type = ["staticlib"])
- [x] Dynamic library option (crate-type = ["cdylib"])
- [ ] Symbol versioning

---

## Code Organization

- [x] BLAS Level 1 refactored into 2 modules (blas1/)
- [x] BLAS Level 2 refactored into 5 modules (blas2/)
- [x] BLAS Level 3 refactored into 3 modules (blas3/)
- [x] LAPACK solve refactored into 3 modules (lapack/solve/)

## Testing

- [x] Unit tests (passing)
- [ ] BLAS-TESTER compatibility
- [ ] LAPACK-TESTER compatibility
- [ ] ABI compatibility tests
- [ ] Thread-safety tests
- [ ] Valgrind/ASan memory tests
- [ ] Fortran interop tests

---

## Documentation

- [ ] C API documentation
- [ ] Usage examples in C
- [ ] Migration guide from OpenBLAS
- [ ] Migration guide from MKL
- [ ] Performance comparison charts
