# oxiblas-core

**Core traits, SIMD abstractions, and scalar types for the OxiBLAS library**

[![Crates.io](https://img.shields.io/crates/v/oxiblas-core.svg)](https://crates.io/crates/oxiblas-core)
[![Documentation](https://docs.rs/oxiblas-core/badge.svg)](https://docs.rs/oxiblas-core)

## Overview

`oxiblas-core` is the foundational crate for OxiBLAS, providing the core abstractions and building blocks used throughout the library. It is designed to be platform-agnostic with architecture-specific SIMD backends for x86_64 (SSE4.2/AVX2/AVX-512), AArch64 (NEON) and WebAssembly (SIMD128), and it builds as `no_std` (with `alloc`) when the default `std` feature is turned off.

## Features

### Core Traits

- **`Scalar`** - Fundamental trait for numeric types supported by BLAS/LAPACK
  - Implemented for: `f32`, `f64`, `Complex<f32>`, `Complex<f64>`
  - Optional support for `f16` (half precision, `f16` feature) and `QuadFloat` (quad precision, `f128` feature)
  - Provides type-safe operations and conversions
- **`Real`**, **`ComplexScalar`**, **`Field`** - Refinements of `Scalar` for real types, complex types, and generic field arithmetic
- **`ScalarBatch`**, **`ScalarClassify`**, **`SimdCompatible`**, **`HasFastFma`**, **`UnrollHints`**, **`ExtendedPrecision`** - Slice-level operations (dot, axpy, sum, ...), compile-time type classification and tuning hints
- **Complex helpers** - `C32`/`C64` aliases, `c32()`/`c64()` constructors, `ComplexExt`, `ToComplex`

### SIMD Abstractions

A register-level SIMD layer (`SimdRegister`, `SimdScalar`) with runtime detection (`detect_simd_level`) and automatic fallback:

**x86_64:**
- **AVX-512F** (512-bit): `F64x8`, `F32x16` - 8×f64 or 16×f32 per register
- **AVX2/FMA** (256-bit): `F64x4`, `F32x8` - 4×f64 or 8×f32 per register
- **SSE4.2** (128-bit): `F64x2Sse`, `F32x4Sse` - 2×f64 or 4×f32 per register
- The 256-bit and 512-bit registers test for AVX2+FMA or AVX-512F on every operation (compile-time target features under `no_std`) and otherwise fall back to scalar lane loops, so constructing them is safe on any x86_64 CPU

**AArch64:**
- **NEON** (128-bit): `F64x2`, `F32x4` - 2×f64 or 4×f32 per register
- The 256-bit `F64x4` and `F32x8` are emulated with two NEON registers
- **SVE** is limited to the `SveSupport` query helper and three free-function kernels behind the nightly-only `sve-nightly` feature (see Feature Flags). The crate declares the `stdarch_aarch64_sve` feature gate for that configuration; it is compile-checked on nightly with `-C target-feature=+sve`, but has not run on SVE hardware

**WebAssembly:**
- **SIMD128** (128-bit): `F64x2`, `F32x4` (with the `simd128` target feature); the 256-bit `F64x4` and `F32x8` are emulated with two registers

**Fallback:**
- **Portable** array-backed registers (`PortableF64x4`, `PortableF64x8`, `PortableF32x8`, `PortableF32x16`) back `SimdScalar` on every other architecture
- **Scalar** operations (`ScalarF64`, `ScalarF32`) for platforms without SIMD support

The register types are not limited by `force-scalar` or `max-simd-*`; those flags cap runtime detection (see Feature Flags). Complex numbers have interleaved-layout registers in `simd::complex`, and `simd::dispatch` exposes the detected CPU features (`simd_caps()`, `SimdCapabilities`) and a `simd_dispatch!` macro for choosing an implementation at run time. The register layer is a general-purpose toolkit: the tuned kernels in `oxiblas-blas` carry their own intrinsics and do not call `SimdRegister`; only the capability layer (`KernelSelector`, built on `SimdCapabilities`) is used there to pick a GEMM micro-kernel.

### Extended Precision Types

- **`f16` (half precision)** - 16-bit floating point (with `f16` feature)
  - Useful for memory-constrained applications
  - `Real` functions such as `sqrt` and `exp` are evaluated in `f32` and rounded back to `f16`

- **`QuadFloat` (quad precision, `f128` feature)** - about 31 decimal digits for `+`, `-`, `*`, `/`, `recip` and `sqrt` (division is FMA-based since 0.2.3; twofloat's own division is only double precision). The elementary functions (`exp`, `ln`, `sin`, ...) are twofloat's and less accurate: 12 to 23 digits in spot checks, for example `ln(10)` about 12, `acosh` about 13 and `exp` about 18
  - Based on double-double arithmetic (the `twofloat` crate)
  - Essential for high-accuracy numerical computations
  - Implements `Scalar`, `Real` and `Field`

- **Accurate summation** (any scalar type, no feature needed) - `KahanSum`, `KBKSum` (Kahan-Babuska-Klein) and `pairwise_sum`
- **`ExtendedPrecision`** - Accumulate in a wider type: `f32` in `f64`, `Complex<f32>` in `Complex<f64>`, and `f64` in `QuadFloat` when `f128` is enabled

### Memory Management

- **Cache-aware allocation** - Platform-specific cache line alignment (`CACHE_LINE_SIZE`: 128 bytes on AArch64, 64 bytes elsewhere)
- **Memory alignment** - `AlignedVec<T, ALIGN>` with a const-generic power-of-two alignment (default `DEFAULT_ALIGN`, the cache line size) and a pluggable `Alloc` allocator trait
- **Workspace management** - Scratch space for temporary buffers: `StackReq`/`MemStack`, `MemoryPool`/`AlignedPool` and `Arena` allocators
- **NUMA helpers** (`std` only) - `NumaTopology`, `NumaVec`, `numa_distribute_work`
- **Prefetch hints** - `prefetch_read`/`prefetch_write` with `PrefetchLocality` and `PrefetchDistance`

### Blocking & Tuning

- **Block-size heuristics** (`blocking`) - `gemm_block_sizes`, `trsm_block_size` and `factorization_panel_width`, derived from fixed typical cache sizes (32 KiB L1, 256 KiB L2, 8 MiB L3)
- **Cache-oblivious helpers** - `BlockRange`, `RecursiveTask`, `BlockVisitor`, `cache_oblivious_traverse`, Morton ordering
- **Tuning** (`tuning`) - `TuningConfig::for_dimensions` picks block sizes from the detected SIMD level, `TuningCache` holds the shared choice, and `AutoTuner::tune_gemm` times a few candidates on a small probe matrix (`std` only)
- **Runtime cache-size detection** (Linux sysfs, macOS sysctl, x86_64 CPUID) is not part of this crate; it lives in `oxiblas-blas` (`oxiblas_blas::level3::get_cache_info`) together with the GEMM blocking that uses it

### Parallel Operations

- **Rayon integration** (with `parallel` feature, which implies `std`)
- **`Par`** - `Par::Seq`, plus `Par::Rayon` and `Par::RayonWith(n)` with `parallel`; used by the parallel drivers in `oxiblas-blas` and `oxiblas-lapack`
- **Work partitioning** - `ParThreshold`, `partition_work`/`WorkRange`, and the helpers `for_each_range`, `for_each_indexed` and `map_reduce`
- **Thread pools** - `ThreadPool` trait, `SequentialPool`, `RayonGlobalPool`, `CustomRayonPool`, `OxiblasThreadConfig`, `set_global_thread_pool` and `with_thread_count`

`Par::RayonWith(n)` uses `n` in the parallel gate and, in GEMV and GEMM, to size the work partition, but those drivers run on the ambient rayon pool: bound their thread count with `rayon::ThreadPool::install` or `RAYON_NUM_THREADS`. The helpers in this crate (`for_each_range`, `for_each_indexed`, `map_reduce`) do install an `n`-thread pool for `Par::RayonWith(n)`, and use a pool registered with `set_global_thread_pool` for `Par::Rayon`; the `oxiblas-blas` and `oxiblas-lapack` drivers do not consult that registry.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
oxiblas-core = "0.2"

# With extended precision
oxiblas-core = { version = "0.2", features = ["f16", "f128"] }

# With parallelization
oxiblas-core = { version = "0.2", features = ["parallel"] }

# Extended precision and parallelization together
oxiblas-core = { version = "0.2", features = ["f16", "f128", "parallel"] }

# no_std (requires alloc)
oxiblas-core = { version = "0.2", default-features = false }
```

## Usage

`use oxiblas_core::prelude::*;` imports the common scalar traits, memory, blocking, parallel, SIMD and tuning types.

### Basic Scalar Operations

```rust
use oxiblas_core::scalar::Scalar;

fn dot_product<T: Scalar>(x: &[T], y: &[T]) -> T {
    x.iter()
        .zip(y.iter())
        .map(|(a, b)| *a * *b)
        .fold(T::zero(), |acc, v| acc + v)
}

// Works with f32, f64, Complex<f32>, Complex<f64>
let x = vec![1.0f64, 2.0, 3.0];
let y = vec![4.0f64, 5.0, 6.0];
let result = dot_product(&x, &y); // 32.0
```

### SIMD Operations

```rust
use oxiblas_core::simd::portable::PortableF64x4;
use oxiblas_core::simd::{SimdLevel, SimdRegister, SimdScalar, detect_simd_level};

// Best instruction set found at runtime (honours force-scalar and max-simd-*)
let level: SimdLevel = detect_simd_level();
println!("{level:?}: {} f64 lanes per register", level.lanes::<f64>());

// Native 256-bit f64 register of this target (AVX2 on x86_64, two NEON registers on AArch64, ...)
type V = <f64 as SimdScalar>::Simd256;
let acc = V::splat(1.5).mul_add(V::splat(2.0), V::zero()); // 1.5 * 2.0 + 0.0 in every lane
assert_eq!(acc.reduce_sum(), 3.0 * V::LANES as f64);

// The portable array-backed register has the same API on every target
let x = PortableF64x4([1.0, 2.0, 3.0, 4.0]);
let r = x.mul_add(PortableF64x4::splat(2.0), PortableF64x4::zero()); // x * 2 + 0
assert_eq!(r.to_array(), [2.0, 4.0, 6.0, 8.0]);
assert_eq!(r.reduce_sum(), 20.0);
```

### Extended Precision

```rust
// Requires the `f128` feature
use oxiblas_core::scalar::{QuadFloat, Real, Scalar};

// Quad precision (double-double): + - * / and sqrt to about 31 decimal digits
let x = QuadFloat::from(2.0);
let sqrt_x = x.sqrt();
let err = Scalar::abs(sqrt_x * sqrt_x - x);
assert!(err < QuadFloat::from(1e-30)); // far below the f64 rounding error of ~1e-16
```

### Accurate Summation

```rust
use oxiblas_core::scalar::{KBKSum, KahanSum, pairwise_sum};

let values: Vec<f64> = vec![1.0, 1e-16, -1.0]; // Difficult for naive sum

let naive: f64 = values.iter().sum(); // 0.0: the 1e-16 is absorbed by the 1.0
assert_eq!(naive, 0.0);

// Compensated (Kahan) summation keeps the small term
let mut kahan = KahanSum::new();
for &v in &values {
    kahan.add(v);
}
assert!(kahan.sum() > 0.0);

// Kahan-Babuska-Klein summation with a second compensation term
let mut kbk = KBKSum::new();
for &v in &values {
    kbk.add(v);
}
assert!(kbk.sum() > 0.0);

// Pairwise (tree) summation: error grows with log(n) instead of n
let total = pairwise_sum(&values);
```

### Block Sizes and Tuning

```rust
use oxiblas_core::blocking::gemm_block_sizes;
use oxiblas_core::tuning::TuningConfig;

// Heuristic GEMM block sizes for an (m, n, k) problem, from fixed typical cache sizes
let (block_m, block_n, block_k) = gemm_block_sizes::<f64>(2048, 2048, 2048);
println!("block_m={block_m}, block_n={block_n}, block_k={block_k}");

// Block sizes picked from the detected SIMD level
let config = TuningConfig::for_dimensions(2048, 2048, 2048);
println!(
    "{:?}: {}x{}x{}",
    config.simd_level, config.block_m, config.block_n, config.block_k
);
```

### Parallel Reductions

```rust
// Requires the `parallel` feature for Par::Rayon and Par::RayonWith(n)
use oxiblas_core::parallel::{Par, ParThreshold, map_reduce};

let data: Vec<f64> = (0..100_000).map(f64::from).collect();

// Work is split into ranges, mapped in parallel and reduced; small inputs run sequentially
let total = map_reduce(
    data.len(),
    Par::Rayon,
    &ParThreshold::default(),
    0.0,
    |range| data[range.start..range.end].iter().sum::<f64>(),
    |a, b| a + b,
);
assert_eq!(total, 4_999_950_000.0);
```

## Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `std` | Standard library: runtime CPU feature detection, NUMA helpers, `OxiblasThreadConfig`, the timed `AutoTuner`. Without it the crate is `no_std` (needs `alloc`) and SIMD levels come from compile-time target features | ✓ |
| `parallel` | Rayon-based parallelization (implies `std`) | |
| `f16` | Half-precision (16-bit) floating point via `half` | |
| `f128` | Quad-precision `QuadFloat` via double-double (`twofloat`): about 31 digits for the arithmetic and `sqrt`, fewer for the elementary functions | |
| `sve-nightly` | AArch64 SVE kernels (`simd::aarch64::{sve_dot_f64, sve_dot_f32, sve_axpy_f64}`). Takes effect only on a target that enables SVE (for example `-C target-feature=+sve`), where `lib.rs` then declares `feature(stdarch_aarch64_sve)` and a nightly toolchain is required; elsewhere it compiles to nothing. Compile-checked on nightly, not run on SVE hardware | |
| `force-scalar` | Debug aid: `detect_simd_level()`, `SimdCapabilities` and `KernelSelector` report scalar only, so the complex batch helpers and the GEMM micro-kernel choice in `oxiblas-blas` take their scalar paths. The register types (`F64x4`, ...) are not affected. The `oxiblas-blas` feature of the same name (which enables this one) also keeps its aarch64 DOT, NRM2, GEMV and TRSM/TRMM kernels on portable code | |
| `max-simd-128` | Cap the same detection at 128-bit SIMD (SSE/NEON) | |
| `max-simd-256` | Cap the same detection at 256-bit SIMD (AVX2) | |

When several of the three are enabled, `force-scalar` wins over `max-simd-128`, which wins over `max-simd-256`.

## SIMD Support Matrix

| Platform | 128-bit | 256-bit | 512-bit |
|----------|---------|---------|---------|
| x86_64 (SSE4.2) | ✓ | | |
| x86_64 (AVX2) | ✓ | ✓ | |
| x86_64 (AVX-512) | ✓ | ✓ | ✓ |
| AArch64 (NEON) | ✓ | Emulated | |
| WebAssembly (SIMD128) | ✓ | Emulated | |
| Other architectures (portable registers) | | Array-backed | Array-backed |

On AArch64 and WebAssembly, `SimdScalar::Simd512` reuses the 256-bit emulation. SVE has no register type (scalable vectors are sizeless in Rust); only `SveSupport` and the `sve-nightly` kernels listed above exist, compile-checked but not run on SVE hardware. `ScalarF64` and `ScalarF32` are the one-lane scalar fallback.

## Architecture

```text
oxiblas-core/
├── scalar/
│   ├── traits.rs       # Scalar, Real, ComplexScalar, Field
│   ├── real_impl.rs    # f32 / f64 implementations
│   ├── complex_impl.rs # Complex<f32> / Complex<f64> implementations
│   ├── complex.rs      # C32/C64, c32()/c64(), ComplexExt, ToComplex
│   ├── extended.rs     # f16 and QuadFloat (f128) implementations
│   └── batch.rs        # ScalarBatch, KahanSum, KBKSum, pairwise_sum, ...
├── simd.rs             # SimdLevel, detection, SimdRegister / SimdScalar
├── simd/
│   ├── x86_64/         # SSE4.2 / AVX2 / AVX-512 registers (x86_64)
│   ├── aarch64.rs      # NEON registers, SVE helpers (AArch64)
│   ├── wasm32.rs       # SIMD128 registers (wasm32)
│   ├── portable.rs     # Array-backed registers for all other targets
│   ├── scalar.rs       # Scalar fallback registers
│   ├── complex.rs      # Complex (interleaved) SIMD registers
│   ├── dispatch.rs     # SimdCapabilities, KernelSelector, simd_dispatch!
│   └── multiver.rs     # SimdCapabilityInfo, simd_dispatch_caps!
├── memory/
│   ├── aligned_vec.rs  # AlignedVec
│   ├── alloc.rs        # Alloc trait, prefetch hints, alignment constants
│   ├── stack.rs        # StackReq, MemStack
│   ├── pool.rs         # MemoryPool, AlignedPool
│   ├── arena.rs        # Arena allocators
│   └── numa.rs         # NUMA-aware allocation (std only)
├── blocking.rs         # Block-size heuristics, cache-oblivious traversal
├── tuning.rs           # TuningConfig, TuningCache, AutoTuner
└── parallel.rs         # Par, work partitioning, thread pools
```

## Supported Platforms

### Tier 1 (Fully Tested)
- **x86_64**: Linux, macOS, Windows
- **AArch64**: macOS (Apple Silicon), Linux

### Tier 2 (Best Effort)
- **x86**: Linux, Windows
- **AArch64**: Android, iOS
- **WebAssembly**: wasm32 (SIMD128 backend with the `simd128` target feature)
- **RISC-V**: Linux (no SIMD intrinsics; portable registers)
- **Bare-metal `no_std`**: build-checked on `thumbv7em-none-eabihf` with `--no-default-features` (by `tests/no_std_build.rs`, when that target is installed)

## Requirements

- **Rust**: 1.89+ (Edition 2024)
- **No C/C++/Fortran in the normal dependencies** (with `std` on Unix the `libc` crate, Rust bindings only, is used for NUMA and page-size queries). The `criterion` dev-dependency of the benches compiles C through `alloca`
- **Optional**: Rayon for parallelization (the `parallel` feature)

## Examples

The examples live in the [`oxiblas` crate](../oxiblas/examples/):

- `extended_precision.rs` - `QuadFloat`, Kahan and pairwise summation (run with `cargo run -p oxiblas --example extended_precision --features f128`)

## Benchmarks

Run benchmarks:

```bash
# SIMD benchmarks
cargo bench --package oxiblas-core --bench simd

# Blocking parameter benchmarks
cargo bench --package oxiblas-core --bench blocking
```

## Safety

- Loads and stores through raw pointers (`SimdRegister::load_*` / `store_*`) are `unsafe fn`; register arithmetic is safe
- The AVX2 and AVX-512 registers test for CPU support at run time and fall back to scalar lane loops (under `no_std` they use the compile-time target features); they do not consult `force-scalar` or `max-simd-*`
- The `sve-nightly` kernels are `unsafe fn` and do not check for SVE support themselves; they have been compile-checked but not run on SVE hardware
- The alignment parameter of `AlignedVec` must be a power of two; this is checked at compile time

## Contributing

Contributions are welcome! Areas of interest:

1. **ARM SVE support** - Stable-Rust SVE support (today only the nightly-only `sve-nightly` kernels exist, and they have not run on SVE hardware yet; test runs on SVE machines are welcome)
2. **RISC-V vector** - Vector extension support
3. **Additional extended precision** - Alternative quad-float implementations
4. **Auto-tuning improvements** - Better platform detection

## Related Crates

- [`oxiblas-matrix`](../oxiblas-matrix/) - Matrix types built on oxiblas-core
- [`oxiblas-blas`](../oxiblas-blas/) - BLAS operations using oxiblas-core
- [`oxiblas-lapack`](../oxiblas-lapack/) - LAPACK decompositions
- [`oxiblas`](../oxiblas/) - Meta-crate with unified API

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](../../LICENSE) for details.

## References

- [BLIS Design](https://github.com/flame/blis) - Blocking and micro-kernel design inspiration
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/) - x86_64 SIMD reference
- [ARM NEON Intrinsics](https://developer.arm.com/architectures/instruction-sets/intrinsics/) - AArch64 SIMD reference
- [Kahan Summation](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) - Compensated summation algorithm
