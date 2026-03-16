# oxiblas-matrix TODO

Matrix types and storage formats for OxiBLAS.

## Storage Formats

### Current Status
- [x] Dense row-major
- [x] Dense column-major
- [x] MatRef / MatMut views
- [x] Packed symmetric storage (upper/lower triangular packed)
- [x] Banded storage format (general and symmetric)
- [x] Block storage for blocked algorithms (via submatrix views)
- [x] Strided views with arbitrary strides

### Future Enhancements
- [x] Custom allocator support (Mat<T, A> with Alloc trait)
- [x] Memory-mapped matrices - `MmapMat`, `MmapMatMut`, `MmapBuilder` with column-major storage
- [x] Lazy evaluation for element-wise ops - `lazy.rs` with `Expr` trait, expression types (Add, Sub, Mul, Scale, Transpose, Conj, Hermitian), FMA and GEMM fusion, simplification optimizations
- [x] Copy-on-write semantics - `CowMat<T>` with Arc-based sharing, O(1) clone, copy-on-write mutation

## Matrix Types

- [x] SymmetricMat - enforced symmetric storage (packed)
- [x] HermitianMat - enforced Hermitian storage (packed)
- [x] TriangularMat - triangular storage (packed)
- [x] BandedMat - banded matrix type
- [x] PackedMat - packed triangular/symmetric
- [x] SymmetricBandedMat - symmetric banded storage

## Operations

- [x] In-place transpose
- [x] Block extraction
- [x] Submatrix views with bounds checking
- [x] Diagonal extraction/setting
- [x] Row/column permutation
- [x] Matrix copy utilities (copy, axpy)
- [x] Symmetrization (upper/lower)
- [x] Zero out triangles
- [x] Horizontal/vertical concatenation (hcat, vcat, hstack, vstack)
- [x] Trace and Frobenius norm

## Interoperability

- [x] ndarray integration (in oxiblas-ndarray crate)
- [x] From/Into nalgebra (optional feature) - `nalgebra` feature with DMatrix/DVector conversions
- [x] Raw pointer access with safety guarantees
- [x] C-compatible layout guarantees (column-major)

## Performance

- [x] Cache-aligned allocation
- [x] Prefetch hints for large matrices (prefetch module)
- [x] SIMD-friendly memory layout

## Testing

- [x] Unit tests (~187 passing)
- [x] Doc tests
- [x] Property-based tests (quickcheck) - 29 tests
- [x] Integration tests (blas_compat_tests - 16 tests, property_tests - 29 tests)
- [x] Memory safety tests (miri) - 96 tests passing (prefetch tests skipped - inline asm)
- [x] Layout compatibility tests (BLAS)

## Future Enhancements

- [x] Copy-on-write semantics - Implemented in cow.rs
- [x] Arena-based allocation for temporary matrices - Arena<ALIGN> in oxiblas-core/memory.rs with Cell-based interior mutability, ArenaVec, save/restore, with_blas_arena thread-local access

## Status (v0.2.1, 2026-03-16)

- 187 lib tests passing
- 12 doctests passing
- 0 todo!() stubs remaining
