# oxiblas-matrix

**Matrix types and views for OxiBLAS**

[![Crates.io](https://img.shields.io/crates/v/oxiblas-matrix.svg)](https://crates.io/crates/oxiblas-matrix)
[![Documentation](https://docs.rs/oxiblas-matrix/badge.svg)](https://docs.rs/oxiblas-matrix)

## Overview

`oxiblas-matrix` provides efficient, type-safe matrix types and views for the OxiBLAS library. It implements zero-copy views, lazy evaluation, and various matrix storage formats (dense, banded, packed, symmetric, triangular).

## Features

### Core Matrix Types

- **`Mat<T>`** - Owned, column-major dense matrix
- **`MatRef<'a, T>`** - Immutable view into a matrix
- **`MatMut<'a, T>`** - Mutable view into a matrix
- **`DiagRef<'a, T>`** - Diagonal matrix view

### Specialized Storage

- **`BandedMatrix<T>`** - Banded matrix storage (efficient for tridiagonal, pentadiagonal, etc.)
- **`PackedMatrix<T>`** - Packed storage for symmetric/triangular matrices (saves 50% memory)
- **`SymmetricMatrix<T>`** - Symmetric matrix with upper/lower storage
- **`TriangularMatrix<T>`** - Triangular matrix (upper/lower with unit/non-unit diagonal)

### Advanced Features

- **Lazy evaluation** - `LazyMat<T>` for deferred computation and expression templates
- **Copy-on-write** - `CowMat<T>` for efficient memory management
- **Memory-mapped matrices** - `MmapMat<T>` for large out-of-core matrices (with `mmap` feature)
- **Nalgebra compatibility** - Convert to/from nalgebra types (with `nalgebra` feature)

### Safety & Performance

- **Zero-copy views** - No allocation for submatrices
- **Compile-time size checks** - Where possible using const generics
- **SIMD-friendly layout** - Proper alignment for vectorization
- **Cache-aware** - Column-major (Fortran) order for BLAS compatibility
- **Prefetching support** - Manual prefetch hints for performance-critical code

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
oxiblas-matrix = "0.1"

# With serialization
oxiblas-matrix = { version = "0.1", features = ["serde"] }

# With memory-mapped matrices
oxiblas-matrix = { version = "0.1", features = ["mmap"] }

# With nalgebra interop
oxiblas-matrix = { version = "0.1", features = ["nalgebra"] }

# All features
oxiblas-matrix = { version = "0.1", features = ["serde", "mmap", "nalgebra"] }
```

## Usage

### Creating Matrices

```rust
use oxiblas_matrix::Mat;

// From dimensions (initialized to zero)
let a = Mat::<f64>::zeros(3, 3);

// From a slice (column-major order)
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let b = Mat::from_slice(2, 3, &data);

// From rows (more intuitive for initialization)
let c = Mat::from_rows(&[
    &[1.0, 2.0, 3.0],
    &[4.0, 5.0, 6.0],
]);

// Identity matrix
let id = Mat::<f64>::identity(4);

// Diagonal matrix
let diag = Mat::from_diag(&[1.0, 2.0, 3.0, 4.0]);
```

### Matrix Access

```rust
use oxiblas_matrix::Mat;

let mut a = Mat::from_rows(&[
    &[1.0, 2.0, 3.0],
    &[4.0, 5.0, 6.0],
]);

// Element access (row, col)
let val = a[(0, 1)];  // 2.0
a[(1, 2)] = 9.0;      // Modify element

// Row/column access
let row = a.row(0);   // [1.0, 2.0, 3.0]
let col = a.col(1);   // [2.0, 5.0]

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
let at = view.t();

// Mutable view
let mut b = a.clone();
let mut_view = b.as_mut();
mut_view[(0, 0)] = 99.0;
```

### Banded Matrices

```rust
use oxiblas_matrix::BandedMatrix;

// Tridiagonal matrix (1 subdiagonal, 1 superdiagonal)
let mut band = BandedMatrix::<f64>::new(5, 5, 1, 1);

// Set diagonal
for i in 0..5 {
    band.set(i, i, 2.0);
}

// Set super/subdiagonals
for i in 0..4 {
    band.set(i, i + 1, -1.0);  // Superdiagonal
    band.set(i + 1, i, -1.0);  // Subdiagonal
}

// Use in BLAS operations
// (Much more efficient than full dense storage)
```

### Packed Matrices

```rust
use oxiblas_matrix::{PackedMatrix, Uplo};

// Symmetric matrix in packed storage (only upper triangle stored)
let mut packed = PackedMatrix::<f64>::new(4, Uplo::Upper);

// Set elements (only upper triangle)
packed.set(0, 0, 1.0);
packed.set(0, 1, 2.0);
packed.set(0, 2, 3.0);
// ... symmetric elements accessed via symmetry

// Saves 50% memory for large symmetric matrices!
```

### Lazy Evaluation

```rust
use oxiblas_matrix::{Mat, LazyMat};

let a = Mat::from_rows(&[[1.0, 2.0], [3.0, 4.0]]);
let b = Mat::from_rows(&[[5.0, 6.0], [7.0, 8.0]]);

// Build expression tree without computation
let lazy = LazyMat::add(
    LazyMat::from_mat(a.as_ref()),
    LazyMat::from_mat(b.as_ref()),
);

// Evaluate when needed
let result = lazy.eval();  // Now computes a + b
```

### Memory-Mapped Matrices

```rust
#[cfg(feature = "mmap")]
{
    use oxiblas_matrix::MmapMat;
    use std::path::Path;

    // Create or open memory-mapped matrix
    let path = Path::new("large_matrix.bin");
    let mmap = MmapMat::<f64>::create(path, 10000, 10000)?;

    // Access like regular matrix (backed by disk)
    let val = mmap[(100, 200)];

    // Efficient for matrices larger than RAM
}
```

### Nalgebra Interop

```rust
#[cfg(feature = "nalgebra")]
{
    use oxiblas_matrix::Mat;
    use nalgebra as na;

    // From nalgebra
    let na_mat = na::DMatrix::<f64>::zeros(3, 3);
    let ox_mat = Mat::from_nalgebra(&na_mat);

    // To nalgebra
    let back = ox_mat.to_nalgebra();
}
```

## Matrix Storage Formats

### Dense (Column-Major)

Standard BLAS-compatible storage:
```
Matrix:     Storage:
[a b c]     [a d b e c f]
[d e f]
```
- **Memory**: `m × n` elements
- **Cache**: Column-wise access is cache-friendly
- **Use for**: General matrices

### Banded

Stores only diagonals:
```
Matrix:          Storage (banded):
[a b 0 0]        [0 a b c d]
[e f g 0]        [a b c d 0]
[0 h i j]        [b c d 0 0]
[0 0 k l]
```
- **Memory**: `(kl + ku + 1) × n` where kl=lower bandwidth, ku=upper bandwidth
- **Cache**: Excellent for narrow banded systems
- **Use for**: Tridiagonal, pentadiagonal, sparse banded systems

### Packed (Symmetric/Triangular)

Stores only upper or lower triangle:
```
Symmetric:       Packed storage:
[a b c]          [a b c d e f]
[b d e]          (only upper triangle)
[c e f]
```
- **Memory**: `n × (n + 1) / 2` (50% savings)
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
let transposed = a.as_ref().t();

// All views reference the same underlying data!
```

## Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `default` | Core matrix types | ✓ |
| `serde` | Serialization/deserialization support | |
| `mmap` | Memory-mapped matrices via memmap2 | |
| `nalgebra` | Interoperability with nalgebra | |

## Safety

- **Bounds checking** - All indexing operations check bounds in debug mode
- **No unsafe in public API** - Unsafe code is internal and carefully audited
- **Lifetime safety** - Views cannot outlive their underlying data
- **Thread safety** - `Send` and `Sync` where appropriate

## Architecture

```
oxiblas-matrix/
├── mat.rs              # Owned Mat<T> type
├── mat_ref.rs          # Immutable MatRef<'a, T>
├── mat_mut.rs          # Mutable MatMut<'a, T>
├── banded.rs           # BandedMatrix<T>
├── packed.rs           # PackedMatrix<T>
├── symmetric.rs        # SymmetricMatrix<T>
├── triangular.rs       # TriangularMatrix<T>
├── lazy.rs             # LazyMat<T> expression templates
├── cow.rs              # CowMat<T> copy-on-write
├── mmap.rs             # MmapMat<T> (feature-gated)
├── nalgebra_compat.rs  # Nalgebra conversions (feature-gated)
├── ops.rs              # Operator overloads
└── prefetch.rs         # Prefetch utilities
```

## Examples

See the [examples directory](../../examples/) in the main repository:

- `matrix_basics.rs` - Creating and manipulating matrices
- `matrix_views.rs` - Zero-copy submatrix views
- `banded_matrices.rs` - Working with banded storage
- `mmap_matrices.rs` - Large out-of-core matrices (requires `mmap` feature)

## Testing

```bash
# Run all tests
cargo test --package oxiblas-matrix

# With all features
cargo test --package oxiblas-matrix --all-features

# Property-based tests (QuickCheck)
cargo test --package oxiblas-matrix -- --ignored
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
| BLAS-compatible layout | ✓ | ✓ (optional) | |
| Pure Rust | ✓ | ✓ | ✓ |

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.

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
