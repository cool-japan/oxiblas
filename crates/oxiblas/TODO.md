# oxiblas TODO

Main OxiBLAS crate - unified interface.

## Re-exports

### Current Status
- [x] Core traits
- [x] Matrix types
- [x] BLAS operations
- [x] LAPACK operations

### Missing
- [x] Sparse operations (re-exported as oxiblas::sparse)
- [x] ndarray integration (via `ndarray` feature flag)
- [ ] Feature-gated imports

---

## High-Level API

### Current Status
- [x] Matrix builder patterns (builder.rs)
- [x] Automatic algorithm selection (auto.rs)
- [x] Fluent API for chained operations (fluent.rs)

### Missing
- [x] Lazy evaluation support - via oxiblas-matrix::lazy module

---

## Prelude Module

- [x] Common imports for typical use
- [x] Type aliases for convenience (C32, C64, etc.)
- [x] Extension traits (MatrixOps, MatrixOpsMut, VectorOps in fluent.rs)

---

## Feature Flags

### Current
- [x] Default features
- [x] `parallel` - Rayon parallelism
- [x] `ndarray` - ndarray integration
- [x] `mmap` - Memory-mapped matrices for large datasets
- [x] `f16` - Half precision support
- [x] `f128` - Quad precision support
- [x] `full` - All features enabled

### Needed
- [x] `simd` - SIMD optimizations control (`force-scalar`, `max-simd-128`, `max-simd-256`)
- [x] `sparse` - Sparse support feature flag (enabled by default)
- [x] `serde` - Serialization support (Mat type)
- [ ] `no_std` - Embedded support (partial)

---

## Testing

- [x] Unit tests (passing)
- [x] Doc tests (passing)
- [x] Integration tests for all re-exports

## Documentation

- [x] Comprehensive crate-level docs
- [x] 5 example files (basic_blas, lapack_decompositions, extended_precision, tensor_operations, sparse_matrices)
- [x] Performance guide (in lib.rs)
- [x] Algorithm selection guide (auto.rs)
- [ ] Comparison with other libraries
