# oxiblas-core TODO

Core traits and SIMD abstractions for OxiBLAS.

## Scalar Traits

- [x] Extended precision (f128) support via `twofloat` (QuadFloat) — 0.2.3: division, `recip` and negative `powi` were only double precision (twofloat 0.8.4 forms the residual without FMA) and are now FMA-based double-double; the elementary functions are still twofloat's, 12 to 23 digits in spot checks (see the root TODO.md backlog)
- [x] Half precision (f16) support via `half` crate
- [x] Better complex number ergonomics (C32, C64, c32, c64, ComplexExt, ToComplex)
- [x] Scalar trait specialization for performance
  - [x] HasFastFma marker trait
  - [x] SimdCompatible trait with SIMD_WIDTH
  - [x] ScalarBatch trait for batch operations (dot_batch, axpy_batch, etc.)
  - [x] ScalarClassify trait for compile-time type dispatch
  - [x] UnrollHints for loop optimization
  - [x] ExtendedPrecision trait for accumulator types
  - [x] KahanSum, KBKSum, pairwise_sum for accurate summation

## SIMD

### Current Status
- [x] AVX2 (x86_64) - 256-bit registers (F64x4, F32x8)
- [x] AVX-512 (x86_64) - 512-bit registers (F64x8, F32x16)
- [x] SSE4.2 (x86_64) - 128-bit registers (F64x2Sse, F32x4Sse)
- [x] NEON (aarch64) - 128-bit native + 256-bit emulated
- [x] WASM SIMD128 - 128-bit native + 256-bit emulated (F64x4, F32x8)
- [x] Scalar fallback (ScalarF64, ScalarF32)

### Advanced Features
- [x] AVX-512BW (Byte/Word)
  - [x] I16x32 (32-lane i16 SIMD)
  - [x] I8x64 (64-lane i8 SIMD)
  - [x] U8x64 (64-lane u8 SIMD)
  - [x] Saturating arithmetic (adds, subs)
- [x] AVX-512VNNI (Vector Neural Network)
  - [x] Avx512Vnni struct with vpdpbusd, vpdpbusds, vpdpwssd, vpdpwssds
  - [x] I32x16 with dpbusd, dpwssd operations
  - [x] Fallback implementations for non-VNNI CPUs
  - [x] Avx512Features for feature detection
- [x] ARM SVE support (scalable vector extension)
  - [x] SveSupport with vector length detection
  - [ ] SveF64, SveF32 scalable vector types: not expressible in current Rust (sizeless types cannot be struct fields, see `simd/aarch64.rs`); SVE is exposed through the free-function kernels below
  - [x] sve_dot_f64, sve_dot_f32 optimized dot products
  - [x] sve_axpy_f64 optimized AXPY
  - [x] `sve-nightly` builds: `lib.rs` declares `feature(stdarch_aarch64_sve)` under `cfg(all(feature = "sve-nightly", target_arch = "aarch64", target_feature = "sve"))` (0.2.3; compile-checked on nightly with `-C target-feature=+sve`, not run on SVE hardware)

### Optimizations
- [x] FMA (fused multiply-add) utilization - all platforms
- [x] Prefetch hints (prefetch_read, prefetch_write with locality)
- [x] Cache line alignment utilities (`CACHE_LINE_SIZE`: 128 on aarch64, chosen for Apple cores, 64 elsewhere; `SimdCapabilities::cache_line_bytes` reports the same value since 0.2.3)
- [x] SIMD horizontal operations (reduce_sum, reduce_max, reduce_min)

## Parallelism

- [x] ParThreshold tuning utilities
- [x] Work partitioning strategies
- [x] Thread-local accumulation (ThreadLocalAccum)
- [x] NUMA-aware helpers (NumaTopology, NumaAllocHint, numa_distribute_work)
- [x] Custom thread pool support (ThreadPool trait, SequentialPool, RayonGlobalPool, CustomRayonPool, PoolScope)

## Memory

- [x] Aligned allocation utilities (AlignedVec)
- [x] Memory pool for temporary allocations (MemoryPool, AlignedPool)
- [x] Prefetch distance calculator (PrefetchDistance)
- [x] Stack-based temporaries (MemStack, StackReq)
- [x] Cache-oblivious algorithms support (blocking.rs)

## Testing

- [x] SIMD correctness tests for all operations
- [x] Cross-platform SIMD validation (scalar, NEON, SSE, AVX2, AVX-512)
- [x] Performance microbenchmarks (benches/simd.rs, benches/blocking.rs)

## Code Organization

- [x] Memory module refactored into 6 modules (memory/)

## Future Enhancements

- [ ] RISC-V Vector (RVV) extension support
- [ ] PowerPC VSX support
- [x] Runtime SIMD dispatch (function multi-versioning) - `SimdCapabilities`, `SimdDispatcher`, `KernelSelector`, `simd_dispatch!` macro (v0.2.0)

## Status (v0.2.1, 2026-03-16)

- 146 lib tests passing
- 17 doctests passing
- 3 todo!() stubs remaining
