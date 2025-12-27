# oxiblas-ffi

**C FFI bindings for OxiBLAS - Drop-in replacement for BLAS/LAPACK libraries**

[![Crates.io](https://img.shields.io/crates/v/oxiblas-ffi.svg)](https://crates.io/crates/oxiblas-ffi)
[![Documentation](https://docs.rs/oxiblas-ffi/badge.svg)](https://docs.rs/oxiblas-ffi)

## Overview

`oxiblas-ffi` provides C-compatible FFI bindings for OxiBLAS, allowing it to be used as a drop-in replacement for OpenBLAS, Intel MKL, or other BLAS/LAPACK libraries in C/C++/Fortran code.

## Features

- **Complete BLAS API** (Level 1, 2, 3) with C bindings
- **LAPACK subset** for common decompositions (LU, QR, SVD, Cholesky)
- **Drop-in replacement** for existing BLAS/LAPACK libraries
- **No runtime dependencies** - single static or dynamic library
- **Pure Rust** implementation with C-compatible ABI

## Installation

### Building the Library

```bash
# Build static library
cargo build --release -p oxiblas-ffi
# Produces: target/release/liboxiblas_ffi.a

# Build dynamic library
cargo build --release -p oxiblas-ffi
# Produces: target/release/liboxiblas_ffi.{so,dylib,dll}
```

### Using from C/C++

```bash
# Link with OxiBLAS (static)
gcc -o myprogram myprogram.c -L/path/to/target/release -loxiblas_ffi -lm

# Link with OxiBLAS (dynamic)
gcc -o myprogram myprogram.c -L/path/to/target/release -loxiblas_ffi -Wl,-rpath,/path/to/target/release
```

### Drop-in Replacement

Replace OpenBLAS or MKL:

```bash
# Before (with OpenBLAS)
gcc -o myprogram myprogram.c -lopenblas

# After (with OxiBLAS)
gcc -o myprogram myprogram.c -L/path/to/oxiblas -loxiblas_ffi
```

## C API

### BLAS Level 1

```c
// Dot product
double cblas_ddot(int n, const double *x, int incx, const double *y, int incy);
float cblas_sdot(int n, const float *x, int incx, const float *y, int incy);

// AXPY: y = alpha*x + y
void cblas_daxpy(int n, double alpha, const double *x, int incx, double *y, int incy);
void cblas_saxpy(int n, float alpha, const float *x, int incx, float *y, int incy);

// Scale: x = alpha*x
void cblas_dscal(int n, double alpha, double *x, int incx);
void cblas_sscal(int n, float alpha, float *x, int incx);

// Euclidean norm
double cblas_dnrm2(int n, const double *x, int incx);
float cblas_snrm2(int n, const float *x, int incx);
```

### BLAS Level 2

```c
// GEMV: y = alpha*A*x + beta*y
void cblas_dgemv(
    CBLAS_LAYOUT layout,
    CBLAS_TRANSPOSE trans,
    int m, int n,
    double alpha,
    const double *a, int lda,
    const double *x, int incx,
    double beta,
    double *y, int incy
);
```

### BLAS Level 3

```c
// GEMM: C = alpha*A*B + beta*C
void cblas_dgemm(
    CBLAS_LAYOUT layout,
    CBLAS_TRANSPOSE transa,
    CBLAS_TRANSPOSE transb,
    int m, int n, int k,
    double alpha,
    const double *a, int lda,
    const double *b, int ldb,
    double beta,
    double *c, int ldc
);

void cblas_sgemm(
    CBLAS_LAYOUT layout,
    CBLAS_TRANSPOSE transa,
    CBLAS_TRANSPOSE transb,
    int m, int n, int k,
    float alpha,
    const float *a, int lda,
    const float *b, int ldb,
    float beta,
    float *c, int ldc
);
```

### LAPACK

```c
// LU factorization
int LAPACKE_dgetrf(
    int matrix_layout,
    int m, int n,
    double *a, int lda,
    int *ipiv
);

// Solve using LU
int LAPACKE_dgetrs(
    int matrix_layout,
    char trans,
    int n, int nrhs,
    const double *a, int lda,
    const int *ipiv,
    double *b, int ldb
);

// QR factorization
int LAPACKE_dgeqrf(
    int matrix_layout,
    int m, int n,
    double *a, int lda,
    double *tau
);

// SVD
int LAPACKE_dgesvd(
    int matrix_layout,
    char jobu, char jobvt,
    int m, int n,
    double *a, int lda,
    double *s,
    double *u, int ldu,
    double *vt, int ldvt,
    double *superb
);

// Cholesky factorization
int LAPACKE_dpotrf(
    int matrix_layout,
    char uplo,
    int n,
    double *a, int lda
);
```

## Usage Examples

### C Example

```c
#include <stdio.h>
#include <cblas.h>

int main() {
    // Matrix dimensions
    int m = 2, n = 3, k = 2;

    // Matrices (column-major order)
    double a[] = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};  // 2x3
    double b[] = {7.0, 9.0, 11.0, 8.0, 10.0, 12.0};  // 3x2
    double c[] = {0.0, 0.0, 0.0, 0.0};  // 2x2 (result)

    // C = A * B
    cblas_dgemm(
        CblasColMajor,
        CblasNoTrans, CblasNoTrans,
        m, n, k,
        1.0,  // alpha
        a, m,
        b, k,
        0.0,  // beta
        c, m
    );

    printf("Result:\n");
    printf("%.0f %.0f\n", c[0], c[2]);
    printf("%.0f %.0f\n", c[1], c[3]);
    // Output: 58 64
    //         139 154

    return 0;
}
```

### Fortran Example

```fortran
program test_oxiblas
    implicit none
    integer :: m, n, k
    real(8), dimension(2,3) :: a
    real(8), dimension(3,2) :: b
    real(8), dimension(2,2) :: c

    m = 2; n = 2; k = 3

    ! Initialize matrices
    a = reshape([1.0d0, 4.0d0, 2.0d0, 5.0d0, 3.0d0, 6.0d0], [2, 3])
    b = reshape([7.0d0, 9.0d0, 11.0d0, 8.0d0, 10.0d0, 12.0d0], [3, 2])

    ! C = A * B
    call dgemm('N', 'N', m, n, k, 1.0d0, a, m, b, k, 0.0d0, c, m)

    print *, 'Result:'
    print *, c(1,1), c(1,2)
    print *, c(2,1), c(2,2)

end program test_oxiblas
```

## Supported Operations

### BLAS Level 1 (Complete)
✓ All standard vector operations (SDOT, DDOT, SAXPY, DAXPY, etc.)

### BLAS Level 2 (Complete)
✓ All matrix-vector operations (SGEMV, DGEMV, SSYMV, DSYMV, etc.)

### BLAS Level 3 (Complete)
✓ All matrix-matrix operations (SGEMM, DGEMM, STRSM, DTRSM, etc.)

### LAPACK (Subset)
✓ LU factorization (SGETRF, DGETRF)
✓ LU solve (SGETRS, DGETRS)
✓ QR factorization (SGEQRF, DGEQRF)
✓ SVD (SGESVD, DGESVD)
✓ Cholesky (SPOTRF, DPOTRF)
✓ Eigenvalues (SSYEV, DSYEV for symmetric matrices)

## Performance

OxiBLAS FFI provides **80-172% of OpenBLAS performance** depending on operation and platform (see main README for detailed benchmarks).

## Building for Different Platforms

### Linux

```bash
cargo build --release -p oxiblas-ffi --target x86_64-unknown-linux-gnu
```

### macOS

```bash
cargo build --release -p oxiblas-ffi --target x86_64-apple-darwin
cargo build --release -p oxiblas-ffi --target aarch64-apple-darwin
```

### Windows

```bash
cargo build --release -p oxiblas-ffi --target x86_64-pc-windows-msvc
```

## Cross-Compilation

One major advantage of OxiBLAS FFI is easy cross-compilation (no C dependencies):

```bash
# Cross-compile for ARM
rustup target add aarch64-unknown-linux-gnu
cargo build --release -p oxiblas-ffi --target aarch64-unknown-linux-gnu
```

## Integration with Build Systems

### CMake

```cmake
find_library(OXIBLAS_LIBRARY
    NAMES oxiblas_ffi
    PATHS /path/to/oxiblas/target/release
)

target_link_libraries(myapp ${OXIBLAS_LIBRARY})
```

### Makefile

```makefile
LDFLAGS += -L/path/to/oxiblas/target/release -loxiblas_ffi
```

## Related Crates

- [`oxiblas`](../oxiblas/) - Pure Rust API
- [`oxiblas-blas`](../oxiblas-blas/) - BLAS operations
- [`oxiblas-lapack`](../oxiblas-lapack/) - LAPACK decompositions

## License

Licensed under MIT or Apache-2.0 at your option.
