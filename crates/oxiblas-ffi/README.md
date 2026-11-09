# oxiblas-ffi

> ## ⚠️ RETIRED / UNMAINTAINED
>
> **This crate is retired as of the v0.2.0 "Pure Rust ecosystem" pivot and is unsupported.**
>
> - It is **excluded from the Cargo workspace** and is **not built or tested by CI**.
> - It is **not published** to crates.io.
> - It has **known correctness defects that will not be fixed**.
> - **Do not use this crate.** For the supported, actively maintained path, use the
>   Pure Rust crates instead: [`oxiblas`](../oxiblas/), [`oxiblas-blas`](../oxiblas-blas/),
>   and [`oxiblas-lapack`](../oxiblas-lapack/).
>
> Everything below this notice is retained for historical reference only and may be
> inaccurate or aspirational.

**C FFI bindings for OxiBLAS - `oblas_*` BLAS and LAPACK-style functions**

## Overview

`oxiblas-ffi` exports C-callable functions that wrap `oxiblas-blas` and `oxiblas-lapack`. The crate's own code defines 354 `extern "C"` functions, all named `oblas_<name>` after the BLAS/LAPACK scheme (for example `oblas_dgemm` and `oblas_dgetrf`). It defines no `cblas_*`, `LAPACKE_*` or Fortran-style (`dgemm_`) symbols itself. A `cdylib` or `staticlib` build does carry the `cblas_*` entry points of its `oxiblas-blas` dependency, but LAPACKE and Fortran-ABI calls do not resolve, so it is **not** a drop-in replacement for OpenBLAS, Intel MKL or reference BLAS/LAPACK (see [Not a Drop-in Replacement](#not-a-drop-in-replacement)).

## Features

- **BLAS Level 1, 2 and 3** (50, 66 and 30 functions): the classic routines for `s`, `d`, `c` and `z`, without `dsdot`, `sdsdot`, `scabs1`, `dcabs1` and `?gemmtr`, plus `crot` and `zrot`
- **LAPACK-style routines** (208 functions): factorizations, solvers, eigenvalue and singular value decompositions, and extras such as norms, determinants, matrix functions and Kronecker products
- **No runtime dependencies** - single static or dynamic library
- **Pure Rust** implementation with C-compatible ABI

## Installation

### Building the Library

As the tree stands the crate cannot be built. It is excluded from the workspace (`exclude` in the root `Cargo.toml`) and its manifest inherits `version`, `edition` and its dependencies from the workspace, so `cargo build -p oxiblas-ffi` at the root fails with `package ID specification 'oxiblas-ffi' did not match any packages`, and `cargo build` inside `crates/oxiblas-ffi` fails with `failed to find a workspace root`. It builds after `crates/oxiblas-ffi` is moved from `exclude` to `members` in the root `Cargo.toml`:

```bash
cargo build --release -p oxiblas-ffi
# Produces: target/release/liboxiblas_ffi.a (static)
#           target/release/liboxiblas_ffi.{so,dylib} or oxiblas_ffi.dll (dynamic)
```

### Using from C/C++

The checked-in `include/oxiblas.h` contains only five C++ standard-library `#include` lines and declares no function, and `build.rs` does not generate a header (the cbindgen call is commented out). Declare the prototypes you call yourself, as in the examples below.

```bash
# Link with OxiBLAS (static)
gcc -o myprogram myprogram.c -L/path/to/target/release -loxiblas_ffi -lm

# Link with OxiBLAS (dynamic)
gcc -o myprogram myprogram.c -L/path/to/target/release -loxiblas_ffi -Wl,-rpath,/path/to/target/release
```

### Not a Drop-in Replacement

The LAPACK side is only reachable under the `oblas_*` names. References to `LAPACKE_*` or Fortran (`dgetrf_`, `dgemm_`) symbols do not resolve, so replacing `-lopenblas` with `-loxiblas_ffi` fails for any program that uses them.

A `cdylib` or `staticlib` build of this crate also contains the 152 `cblas_*` symbols of `oxiblas-blas`, because that crate's CBLAS layer is made of `#[no_mangle]` functions and this crate depends on it. A C program that declares and calls `cblas_dgemm` links against the library and runs. These entry points come from `oxiblas-blas`, not from this crate: there is no header for them (`include/oxiblas.h` is a stub), and they carry that crate's deviations (no `cblas_xerbla`, `cblas_i?amax` returns `int`, the enum arguments are Rust enums); see the [`oxiblas-blas` README](../oxiblas-blas/README.md#cblas-interface).

## C API

### Conventions

- Names follow the BLAS/LAPACK naming scheme with an `oblas_` prefix (`oblas_dgemm`, `oblas_cgemm`, `oblas_isamax`, `oblas_dznrm2`). Complex values are the structs `OblasComplex32` and `OblasComplex64` (`{ re, im }`).
- Level 1 routines take no layout argument. All Level 2 and Level 3 routines, and 184 of the 208 LAPACK-style routines, take `OblasLayout layout` as their first argument. The band routines (`?gbtrf`, `?gbtrs`, `?gbsv`, `?pbtrf`, `?pbtrs`, `?pbsv`) take none and use column-major band storage, and the tridiagonal routines (`?gtsv`, `?gttrf`, `?gttrs`, `?ptsv`, `?pttrf`, `?pttrs`) take none and receive the diagonals as separate arrays.
- The enum arguments are `#[repr(C)]` Rust enums with the `cblas.h` values: `OblasLayout` (`RowMajor` = 101, `ColMajor` = 102), `OblasTranspose` (`NoTrans` = 111, `Trans` = 112, `ConjTrans` = 113), `OblasUplo` (`Upper` = 121, `Lower` = 122), `OblasDiag` (`NonUnit` = 131, `Unit` = 132) and `OblasSide` (`Left` = 141, `Right` = 142). Passing any other value is undefined behaviour.
- Most LAPACK-style routines return an `int`: `0` on success and, mostly, an `OblasReturn` value on failure (`-1` invalid argument, `-2` singular, `-3` not converged, `-5` not positive definite; `-4`, memory error, is defined but never returned). The codes are not uniform: `oblas_dpotrf` reports a matrix that is not positive definite as `-3`, and the band and tridiagonal routines return a positive 1-based index for a zero pivot, as LAPACK `info` does. `oblas_?lange`, `oblas_?lansy` and `oblas_?lantr` return the norm itself.

### BLAS Level 1

```c
// Dot product
double oblas_ddot(int n, const double *x, int incx, const double *y, int incy);
float oblas_sdot(int n, const float *x, int incx, const float *y, int incy);

// AXPY: y = alpha*x + y
void oblas_daxpy(int n, double alpha, const double *x, int incx, double *y, int incy);
void oblas_saxpy(int n, float alpha, const float *x, int incx, float *y, int incy);

// Scale: x = alpha*x
void oblas_dscal(int n, double alpha, double *x, int incx);
void oblas_sscal(int n, float alpha, float *x, int incx);

// Euclidean norm
double oblas_dnrm2(int n, const double *x, int incx);
float oblas_snrm2(int n, const float *x, int incx);
```

### BLAS Level 2

```c
// GEMV: y = alpha*A*x + beta*y
void oblas_dgemv(
    OblasLayout layout,
    OblasTranspose trans,
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
void oblas_dgemm(
    OblasLayout layout,
    OblasTranspose trans_a,
    OblasTranspose trans_b,
    int m, int n, int k,
    double alpha,
    const double *a, int lda,
    const double *b, int ldb,
    double beta,
    double *c, int ldc
);

void oblas_sgemm(
    OblasLayout layout,
    OblasTranspose trans_a,
    OblasTranspose trans_b,
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
int oblas_dgetrf(
    OblasLayout layout,
    int m, int n,
    double *a, int lda,
    int *ipiv
);

// Solve using LU
int oblas_dgetrs(
    OblasLayout layout,
    OblasTranspose trans,
    int n, int nrhs,
    const double *a, int lda,
    const int *ipiv,
    double *b, int ldb
);

// QR factorization
int oblas_dgeqrf(
    OblasLayout layout,
    int m, int n,
    double *a, int lda,
    double *tau
);

// SVD (full; there are no jobu/jobvt/work arguments)
int oblas_dgesvd(
    OblasLayout layout,
    int m, int n,
    double *a, int lda,
    double *s,
    double *u, int ldu,
    double *vt, int ldvt
);

// Cholesky factorization
int oblas_dpotrf(
    OblasLayout layout,
    OblasUplo uplo,
    int n,
    double *a, int lda
);

// Symmetric eigenvalues; jobz is 'N' (eigenvalues only) or 'V' (also eigenvectors)
int oblas_dsyev(
    OblasLayout layout,
    unsigned char jobz,
    OblasUplo uplo,
    int n,
    double *a, int lda,
    double *w
);
```

## Usage Examples

### C Example

```c
#include <stdio.h>

/* include/oxiblas.h declares nothing, so declare what you call. */
typedef enum { OBLAS_ROW_MAJOR = 101, OBLAS_COL_MAJOR = 102 } OblasLayout;
typedef enum { OBLAS_NO_TRANS = 111, OBLAS_TRANS = 112, OBLAS_CONJ_TRANS = 113 } OblasTranspose;

void oblas_dgemm(OblasLayout layout, OblasTranspose trans_a, OblasTranspose trans_b,
                 int m, int n, int k, double alpha,
                 const double *a, int lda, const double *b, int ldb,
                 double beta, double *c, int ldc);

int main(void) {
    // Matrix dimensions
    int m = 2, n = 2, k = 3;

    // Matrices (column-major order)
    double a[] = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};  // 2x3
    double b[] = {7.0, 9.0, 11.0, 8.0, 10.0, 12.0};  // 3x2
    double c[] = {0.0, 0.0, 0.0, 0.0};  // 2x2 (result)

    // C = A * B
    oblas_dgemm(
        OBLAS_COL_MAJOR,
        OBLAS_NO_TRANS, OBLAS_NO_TRANS,
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

There are no Fortran-style symbols (`dgemm_`), so a Fortran program has to bind to the `oblas_*` functions through `iso_c_binding`:

```fortran
program test_oxiblas
    use iso_c_binding
    implicit none

    interface
        subroutine oblas_dgemm(layout, trans_a, trans_b, m, n, k, alpha, a, lda, &
                               b, ldb, beta, c, ldc) bind(C, name="oblas_dgemm")
            use iso_c_binding
            integer(c_int), value :: layout, trans_a, trans_b, m, n, k, lda, ldb, ldc
            real(c_double), value :: alpha, beta
            real(c_double), intent(in) :: a(*), b(*)
            real(c_double), intent(inout) :: c(*)
        end subroutine oblas_dgemm
    end interface

    integer(c_int), parameter :: col_major = 102, no_trans = 111
    integer(c_int) :: m, n, k
    real(c_double), dimension(2,3) :: a
    real(c_double), dimension(3,2) :: b
    real(c_double), dimension(2,2) :: c

    m = 2; n = 2; k = 3

    ! Initialize matrices
    a = reshape([1.0d0, 4.0d0, 2.0d0, 5.0d0, 3.0d0, 6.0d0], [2, 3])
    b = reshape([7.0d0, 9.0d0, 11.0d0, 8.0d0, 10.0d0, 12.0d0], [3, 2])

    ! C = A * B
    call oblas_dgemm(col_major, no_trans, no_trans, m, n, k, 1.0d0, a, m, b, k, 0.0d0, c, m)

    print *, 'Result:'
    print *, c(1,1), c(1,2)
    print *, c(2,1), c(2,2)

end program test_oxiblas
```

## Supported Operations

Every function is exported as `oblas_<name>`; braces list the precision prefixes that exist.

### BLAS Level 1 (50 functions)
✓ All classic routines for `s`, `d`, `c` and `z` (SDOT, DDOT, SAXPY, DAXPY, etc.), plus CROT and ZROT; DSDOT, SDSDOT, SCABS1 and DCABS1 are missing

### BLAS Level 2 (66 functions)
✓ All classic routines, including banded and packed (SGEMV, DGEMV, SSYMV, DSYMV, etc.)

### BLAS Level 3 (30 functions)
✓ All classic routines (SGEMM, DGEMM, STRSM, DTRSM, etc.); `?gemmtr` is missing

### LAPACK-style routines (208 functions)

| Area | Functions |
|------|-----------|
| LU, Cholesky, QR, solve, inverse | `{s,d,c,z}getrf`, `{s,d,c,z}potrf`, `{s,d,c,z}geqrf`, `{s,d,c,z}gesv`, `{s,d}getrs`, `{s,d}potrs`, `{s,d}gels`, `{s,d}getri`, `{s,d}potri`, `{s,d}inv`, `{s,d}geqp3`, `{s,d}getc2`, `{s,d}gesc2` |
| Orthogonal factors | `{s,d}orgqr`, `{s,d}ormqr`, `{c,z}ungqr`, `{c,z}unmqr`, `{s,d}orgbr`, `{s,d}ormbr`, `{s,d}gebrd`, `{s,d}orgqr_pivot` |
| SVD | `{s,d,c,z}gesvd`, `{s,d}gesvd_thin`, `{s,d}gesdd`, `{s,d}gesdd_thin` |
| Eigenvalues | `{s,d}syev`, `{s,d}syevd`, `{c,z}heev`, `{c,z}heevd`, `{s,d,c,z}geev`, `{s,d}sygv`, `{s,d}ggev`, `{s,d}gees`, `{s,d}gges`, `{s,d}gehrd` |
| LDL^T | `{s,d}sytrf`, `{s,d}sytrs`, `{s,d}inertia`, `{s,d}isposdef`, `{s,d}isnegdef`, `{s,d}logabsdet_ldlt` |
| Band and tridiagonal | `{s,d}gbtrf`, `{s,d}gbtrs`, `{s,d}gbsv`, `{s,d}pbtrf`, `{s,d}pbtrs`, `{s,d}pbsv`, `{s,d}gtsv`, `{s,d}gttrf`, `{s,d}gttrs`, `{s,d}ptsv`, `{s,d}pttrf`, `{s,d}pttrs` |
| Expert solve, refinement | `{s,d}gesvx`, `{s,d}posvx`, `{s,d}sysvx`, `{s,d}gerfs`, `{s,d}porfs`, `{s,d}syrfs` |
| Norms, conditioning, determinants, rank | `{s,d}lange`, `{s,d}lansy`, `{s,d}lantr`, `{s,d}norm2`, `{s,d}normnuc`, `{s,d}trace`, `{s,d}gecon`, `{s,d}cond`, `{s,d}cond1`, `{s,d}condinf`, `{s,d}rcond`, `{s,d}rcond_est`, `{s,d}det`, `{s,d}absdet`, `{s,d}detlu`, `{s,d}logdet`, `{s,d}det_chol`, `{s,d}logdet_chol`, `{s,d}pinv`, `{s,d}pinv_default`, `{s,d}rank`, `{s,d}nullity`, `{s,d}null`, `{s,d}colspace`, `{s,d}rowspace`, `{s,d}lnull` |
| Matrix functions, Kronecker products | `{s,d}expm`, `{s,d}logm`, `{s,d}sqrtm`, `{s,d}powm`, `{s,d}kron`, `{s,d}kronsum`, `{s,d}kron_vec`, `{s,d}khatri_rao`, `{s,d}vec`, `{s,d}unvec`, `{s,d}commutation`, `{s,d}duplication`, `{s,d}elimination` |

## Performance

Not benchmarked; see the retirement notice at the top of this document. Performance
comparisons against OpenBLAS are not available for this crate.

## Building for Different Platforms

These commands apply once the crate has been added back to the workspace (see
[Building the Library](#building-the-library)).

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

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.
