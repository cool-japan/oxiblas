# oxiblas-matrix

**Matrix types and views for OxiBLAS**

[![Crates.io](https://img.shields.io/crates/v/oxiblas-matrix.svg)](https://crates.io/crates/oxiblas-matrix)
[![Documentation](https://docs.rs/oxiblas-matrix/badge.svg)](https://docs.rs/oxiblas-matrix)

## Overview

`oxiblas-matrix` provides efficient, type-safe matrix types and views for the OxiBLAS library. It implements zero-copy views, lazy evaluation, and various matrix storage formats (dense, banded, packed, symmetric, Hermitian, triangular). It builds as `no_std` (with `alloc`) when the default `std` feature is disabled.

## Features

### Core Matrix Types

- **`Mat<T>`** - Owned, column-major dense matrix with cache-line aligned storage (each column is padded, so `row_stride()` can exceed `nrows()`); `Mat<T, A>` takes a custom allocator
- **`MatRef<'a, T>`** - Immutable view into a matrix, with independent row and column strides
- **`MatMut<'a, T>`** - Mutable view into a matrix (`rb()` / `rb_mut()` reborrow)
- **`DiagRef<'a, T>`** - Diagonal view (`MatRef::diagonal()`)

### Specialized Storage

- **`BandedMat<T>`** - Banded matrix storage in the BLAS band layout (efficient for tridiagonal, pentadiagonal, etc.), with `SymmetricBandedMat<T>` for symmetric bands
- **`PackedMat<T>`** - Packed storage of one triangle (`TriangularKind::Upper` or `Lower`), `n(n+1)/2` elements instead of `n²`
- **`SymmetricMat<T>`** / **`HermitianMat<T>`** - Symmetric / Hermitian matrix over packed storage; `get` / `set` on the opposite triangle reach the stored element (`HermitianMat` does not conjugate it for you)
- **`TriangularMat<T>`** - Packed triangular matrix (upper/lower with `DiagonalKind::Unit` or `NonUnit`); `TriangularView` / `TriangularViewMut` give the same view over dense storage

Each packed or banded type also has borrowed `...Ref` / `...Mut` forms over a slice (for example `PackedRef`, `BandedMut`).

### Advanced Features

- **Lazy evaluation** - `Expr` expression trees built with `LazyExt::lazy()` (`ExprAdd`, `ExprMul`, `ExprFma`, `ExprGemm`, ...); element-wise chains are evaluated in one fused pass
- **Copy-on-write** - `CowMat<T>` for efficient memory management
- **Memory-mapped matrices** - `MmapMat<T>` (read-only) and `MmapMatMut<T>` (read-write) for large out-of-core matrices (with `mmap` feature)
- **Nalgebra compatibility** - Convert to/from nalgebra types, with zero-copy view conversions (with `nalgebra` feature)
- **Serde** - `Serialize` / `Deserialize` for `Mat<T>` (with `serde` feature)
- **Matrix operations** - The `ops` module: in-place transpose, diagonal get/set, row/column permutation, block extract/set, `hcat` / `vcat` / `hstack` / `vstack`, trace and norms

### Safety & Performance

- **Zero-copy views** - No allocation for submatrices and transposes
- **Checked sizes** - Allocation sizes are computed with checked arithmetic, so absurd dimensions panic instead of wrapping around
- **SIMD-friendly layout** - Proper alignment for vectorization
- **Cache-aware** - Column-major (Fortran) order for BLAS compatibility
- **Prefetching support** - `MatrixPrefetcher`, `prefetch_column` and `prefetch_block` for performance-critical code

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
oxiblas-matrix = "0.2"

# With serialization
oxiblas-matrix = { version = "0.2", features = ["serde"] }

# With memory-mapped matrices
oxiblas-matrix = { version = "0.2", features = ["mmap"] }

# With nalgebra interop
oxiblas-matrix = { version = "0.2", features = ["nalgebra"] }

# All optional features
oxiblas-matrix = { version = "0.2", features = ["serde", "mmap", "nalgebra"] }

# no_std (requires alloc)
oxiblas-matrix = { version = "0.2", default-features = false }
```

## Usage

`use oxiblas_matrix::prelude::*;` brings the common matrix types, the lazy-evaluation traits and the `ops` module into scope.

### Creating Matrices

```rust
use oxiblas_matrix::Mat;

// From dimensions (initialized to zero)
let a = Mat::<f64>::zeros(3, 3);

// Filled with one value
let ones = Mat::<f64>::filled(2, 2, 1.0);

// From a slice (column-major order)
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let b = Mat::from_slice(2, 3, &data);

// From rows (more intuitive for initialization)
let c = Mat::from_rows(&[
    &[1.0, 2.0, 3.0],
    &[4.0, 5.0, 6.0],
]);

// Identity matrix
let id = Mat::<f64>::eye(4);

// Diagonal matrix
let diag = Mat::diag(&[1.0, 2.0, 3.0, 4.0]);
```

### Matrix Access

```rust
use oxiblas_matrix::Mat;

let mut a = Mat::from_rows(&[
    &[1.0, 2.0, 3.0],
    &[4.0, 5.0, 6.0],
]);

// Element access (row, col); indexing checks bounds in every build
let val = a[(0, 1)];  // 2.0
a[(1, 2)] = 9.0;      // Modify element
assert_eq!(a.get(5, 5), None); // get() returns an Option instead of panicking

// Row/column views (no copy)
let row = a.row(0);   // 1×3 view: [1.0, 2.0, 3.0]
let col = a.col(1);   // 2×1 view: [2.0, 5.0]

// Dimensions
let (nrows, ncols) = (a.nrows(), a.ncols());
```

### Matrix Views

```rust
use oxiblas_matrix::Mat;

let a = Mat::from_rows(&[
    &[1.0, 2.0, 3.0, 4.0],
    &[5.0, 6.0, 7.0, 8.0],
    &[9.0, 10.0, 11.0, 12.0],
]);

// Create immutable view
let view = a.as_ref();

// Submatrix view (zero-copy!)
let sub = view.submatrix(0, 0, 2, 2);  // Top-left 2×2 block

// Transpose view (zero-copy!)
let at = view.transpose();

// Mutable view
let mut b = a.clone();
let mut mut_view = b.as_mut();
mut_view[(0, 0)] = 99.0;
```

Views over an existing slice are built with `MatRef::from_column_major` and `MatRef::from_strided`, which return `None` when the slice is too short. The raw-pointer constructors (`MatRef::new`, `MatRef::new_strided`, `MatMut::new`) are `unsafe`.

### Banded Matrices

```rust
use oxiblas_matrix::BandedMat;

// Tridiagonal matrix (1 subdiagonal, 1 superdiagonal)
let mut band = BandedMat::<f64>::zeros(5, 5, 1, 1);

// Set diagonal
for i in 0..5 {
    band.set(i, i, 2.0);
}

// Set super/subdiagonals (set() panics for elements outside the band)
for i in 0..4 {
    band.set(i, i + 1, -1.0);  // Superdiagonal
    band.set(i + 1, i, -1.0);  // Subdiagonal
}

// BLAS band layout: (kl + ku + 1) × n values instead of n × n
assert_eq!(band.ldab(), 3);
let dense = band.to_dense();
assert_eq!(dense[(1, 0)], -1.0);
```

### Packed Matrices

```rust
use oxiblas_matrix::{PackedMat, SymmetricMat, TriangularKind};

// Upper triangle in packed storage: n(n+1)/2 = 10 values for a 4×4 matrix
let mut packed = PackedMat::<f64>::zeros(4, TriangularKind::Upper);
assert_eq!(packed.len(), 10);

// set() reports indices outside the stored triangle as an error
packed.set(0, 1, 2.0).expect("(0, 1) is in the upper triangle");
assert!(packed.set(1, 0, 2.0).is_err());

// SymmetricMat mirrors access across the diagonal
let mut sym = SymmetricMat::<f64>::zeros(4, TriangularKind::Upper);
sym.set(0, 1, 2.0);  // Also reachable as (1, 0)
assert_eq!(sym.get(1, 0), Some(&2.0));
```

### Lazy Evaluation

```rust
use oxiblas_matrix::{Expr, LazyExt, Mat};

let a = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
let b = Mat::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);

// Build an expression tree without computation
let lazy = a.as_ref().lazy() + b.as_ref().lazy();

// Evaluate when needed
let result = lazy.eval();  // Now computes a + b
assert_eq!(result[(1, 1)], 12.0);
```

### Memory-Mapped Matrices

```rust
// Requires the `mmap` feature
use oxiblas_matrix::{MmapMat, MmapMatMut};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::temp_dir().join(format!("oxiblas-readme-{}.oxiblas", std::process::id()));

    // Create a memory-mapped matrix file (backed by disk)
    {
        let mut mmat = MmapMatMut::<f64>::create(&path, 100, 100)?;
        mmat[(10, 20)] = 1.5;
        mmat.flush()?;
    }

    // Open it read-only; efficient for matrices larger than RAM. While a mapping
    // is alive, nothing else may write to or truncate the file (see Safety)
    let mmat = MmapMat::<f64>::open(&path)?;
    assert_eq!(mmat[(10, 20)], 1.5);

    std::fs::remove_file(&path)?;
    Ok(())
}
```

### Nalgebra Interop

```rust
// Requires the `nalgebra` feature
use nalgebra::DMatrix;
use oxiblas_matrix::{DMatrixOxiblasExt, Mat, MatNalgebraExt, dmatrix_to_mat, mat_to_dmatrix};

// From nalgebra (copies: the two owned types use different buffers)
let na_mat = DMatrix::<f64>::zeros(3, 3);
let ox_mat: Mat<f64> = dmatrix_to_mat(&na_mat);

// To nalgebra (copies)
let back: DMatrix<f64> = mat_to_dmatrix(&ox_mat);
let also_back = ox_mat.to_dmatrix();  // MatNalgebraExt

// Borrowing views are zero-copy
let view = ox_mat.to_dmatrix_view();
let ox_ref = na_mat.to_mat_ref();     // DMatrixOxiblasExt
```

## Matrix Storage Formats

### Dense (Column-Major)

Standard BLAS-compatible storage (padding after each column not shown):
```text
Matrix:     Storage:
[a b c]     [a d b e c f]
[d e f]
```
- **Memory**: `row_stride × n` elements, where `row_stride` (the leading dimension) is `m` rounded up to a whole number of cache lines, so every column starts cache-line aligned
- **Cache**: Column-wise access is cache-friendly
- **Use for**: General matrices

### Banded

Stores only diagonals, in the BLAS band layout (`kl = ku = 1` here; `*` marks unused slots):
```text
Matrix:          Storage (banded, 3 × 4):
[a b 0 0]        [* b g j]   <- superdiagonal
[e f g 0]        [a f i l]   <- main diagonal
[0 h i j]        [e h k *]   <- subdiagonal
[0 0 k l]
```
- **Memory**: `(kl + ku + 1) × n` where kl=lower bandwidth, ku=upper bandwidth
- **Cache**: Excellent for narrow banded systems
- **Use for**: Tridiagonal, pentadiagonal, sparse banded systems

### Packed (Symmetric/Triangular)

Stores only the upper or lower triangle, column by column:
```text
Symmetric:       Packed storage (upper triangle):
[a b c]          [a b d c e f]
[b d e]
[c e f]
```
- **Memory**: `n × (n + 1) / 2` (nearly 50% savings)
- **Cache**: Compact storage
- **Use for**: Symmetric or triangular matrices

## Performance Characteristics

| Operation | Dense | Banded (k<<n) | Packed |
|-----------|-------|---------------|--------|
| Element access | O(1) | O(1) | O(1) |
| Row access | O(n) | O(k) | O(n) |
| Memory (n×n) | n² | n×k | n×(n+1)/2 |
| BLAS ops | Optimized | Optimized | Optimized |

## Zero-Copy Design

All view types are zero-copy:

```rust
use oxiblas_matrix::Mat;

let a = Mat::from_rows(&[
    &[1.0, 2.0, 3.0, 4.0],
    &[5.0, 6.0, 7.0, 8.0],
]);

// No allocation - just pointer arithmetic
let sub1 = a.as_ref().submatrix(0, 0, 2, 2);
let sub2 = a.as_ref().submatrix(0, 2, 2, 2);
let transposed = a.as_ref().transpose();

// All views reference the same underlying data!
```

## Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `std` | Standard library support (also enables `oxiblas-core/std`); disable it for `no_std` with `alloc` | ✓ |
| `serde` | Serialization/deserialization support for `Mat<T>` (implies `std`) | |
| `mmap` | Memory-mapped matrices via memmap2 (implies `std`) | |
| `nalgebra` | Interoperability with nalgebra | |

## Safety

- **Bounds checking** - Indexing (`m[(i, j)]`) and `set` check bounds in every build; `get` returns an `Option`
- **Unsafe is explicit** - Views over slices (`MatRef::from_column_major`, `from_strided`) are safe; the raw-pointer constructors (`MatRef::new`, `MatRef::new_strided`, `MatMut::new`) and `get_unchecked` are `unsafe fn`. `ptr_at` may be called with any indices (wrapping arithmetic), but only the pointer of an in-range element may be dereferenced
- **Memory maps are the exception** - the `mmap` constructors are safe functions, but they cannot check two conditions their soundness depends on: no other writer may modify or truncate the file while it is mapped (never open one path twice when either handle is an `MmapMatMut`), and the element type must be one of the supported types (`f32`, `f64`, `Complex<f32>`, `Complex<f64>`, `i32`, `i64`). See the `mmap` module docs; making this explicit in the API is planned for 0.3.0
- **Lifetime safety** - Views cannot outlive their underlying data
- **Thread safety** - `Send` and `Sync` where appropriate

## Architecture

```text
oxiblas-matrix/
├── mat.rs              # Owned Mat<T> type
├── mat_ref.rs          # Immutable MatRef<'a, T>, DiagRef
├── mat_mut.rs          # Mutable MatMut<'a, T>
├── banded.rs           # BandedMat<T>, SymmetricBandedMat<T>
├── packed.rs           # PackedMat<T>, TriangularKind
├── symmetric.rs        # SymmetricMat<T>, HermitianMat<T>
├── triangular.rs       # TriangularMat<T>, TriangularView, DiagonalKind
├── lazy.rs             # Expr trees (lazy evaluation)
├── cow.rs              # CowMat<T> copy-on-write
├── mmap.rs             # MmapMat<T>, MmapMatMut<T> (feature-gated)
├── nalgebra_compat.rs  # Nalgebra conversions (feature-gated)
├── ops.rs              # Matrix helper functions (transpose, permute, hstack, ...)
└── prefetch.rs         # Prefetch utilities
```

## Examples

This crate has no examples of its own. The [`oxiblas` crate](../oxiblas/examples/) has end-to-end programs that build `Mat` values, for example `basic_blas.rs` and `lapack_decompositions.rs`.

## Testing

```bash
# Run all tests
cargo test --package oxiblas-matrix

# With all features
cargo test --package oxiblas-matrix --all-features

# Property-based tests (QuickCheck)
cargo test --package oxiblas-matrix --test property_tests
```

## Related Crates

- [`oxiblas-core`](../oxiblas-core/) - Core traits and SIMD abstractions
- [`oxiblas-blas`](../oxiblas-blas/) - BLAS operations on matrices
- [`oxiblas-lapack`](../oxiblas-lapack/) - LAPACK decompositions
- [`oxiblas`](../oxiblas/) - Meta-crate with unified API

## Comparison with Other Libraries

| Feature | oxiblas-matrix | ndarray | nalgebra |
|---------|----------------|---------|----------|
| Zero-copy views | ✓ | ✓ | ✓ |
| Banded storage | ✓ | | |
| Packed storage | ✓ | | |
| Memory-mapped | ✓ | | |
| Lazy evaluation | ✓ | | |
| BLAS-compatible layout | ✓ | ✓ (optional) | ✓ |
| Pure Rust | ✓ | ✓ | ✓ |

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.

## Contributing

Contributions are welcome! Areas of interest:

1. **Sparse matrix views** - Integration with oxiblas-sparse
2. **Block matrix** - Hierarchical block structure
3. **Strided matrices** - Non-contiguous storage patterns
4. **GPU support** - CUDA/ROCm memory views

## References

- [BLAS Standard](http://www.netlib.org/blas/) - Matrix storage conventions
- [LAPACK Documentation](http://www.netlib.org/lapack/) - Packed and banded formats
- [ndarray](https://github.com/rust-ndarray/ndarray) - General-purpose array library
- [nalgebra](https://nalgebra.org/) - Linear algebra library
