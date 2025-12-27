# oxiblas-core

**Core traits, SIMD abstractions, and scalar types for the OxiBLAS library**

[![Crates.io](https://img.shields.io/crates/v/oxiblas-core.svg)](https://crates.io/crates/oxiblas-core)
[![Documentation](https://docs.rs/oxiblas-core/badge.svg)](https://docs.rs/oxiblas-core)

## Overview

`oxiblas-core` is the foundational crate for OxiBLAS, providing the core abstractions and building blocks used throughout the library. It is designed to be platform-agnostic with architecture-specific optimizations for x86_64 (AVX2/AVX-512) and AArch64 (NEON).

## Features

### Core Traits

- **`Scalar`** - Fundamental trait for numeric types supported by BLAS/LAPACK
  - Implemented for: `f32`, `f64`, `Complex<f32>`, `Complex<f64>`
  - Optional support for `f16` (half precision) and `f128` (quad precision)
  - Provides type-safe operations and conversions

### SIMD Abstractions

Architecture-specific vectorization with automatic fallback:

**x86_64:**
- **AVX-512** (512-bit): 8×f64 or 16×f32 per instruction
- **AVX2/FMA** (256-bit): 4×f64 or 8×f32 per instruction
- **SSE4.1/SSE4.2** (128-bit): 2×f64 or 4×f32 per instruction

**AArch64:**
- **NEON** (128-bit): 2×f64 or 4×f32 per instruction
- Advanced 4×6 micro-kernels optimized for Apple Silicon

**Fallback:**
- **Scalar** operations for platforms without SIMD support

### Extended Precision Types

- **`f16` (half precision)** - 16-bit floating point (with `f16` feature)
  - Useful for memory-constrained applications
  - Hardware acceleration on ARM and modern x86_64

- **`f128` (quad precision)** - ~31 decimal digits precision (with `f128` feature)
  - Based on double-double arithmetic
  - Essential for high-accuracy numerical computations
  - Kahan and pairwise summation algorithms

### Memory Management

- **Cache-aware allocation** - Platform-specific cache line alignment
- **Memory alignment** - SIMD-friendly memory layout (16/32/64-byte alignment)
- **Workspace management** - Efficient temporary buffer reuse for LAPACK algorithms

### Blocking & Tuning

- **Automatic blocking parameters** - Cache-aware tile sizes for GEMM and other operations
- **Platform detection** - Runtime detection of cache sizes (L1/L2/L3)
  - Linux: sysfs (`/sys/devices/system/cpu/`)
  - macOS: sysctl
  - x86_64: CPUID instruction
- **Optimized for**:
  - Intel Xeon (256KB-512KB L2): KC=192, MC=128
  - Apple Silicon (16MB L2): KC=448, MC=256
  - AMD Zen (512KB L2): KC=192, MC=Variable

### Parallel Operations

- **Rayon integration** (with `parallel` feature)
- **Multi-threaded BLAS Level 3** - Automatic parallelization for large matrices
- **Load balancing** - Efficient work distribution across cores
- **Cache-aware parallel blocking** - Minimizes false sharing

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
oxiblas-core = "0.1"

# With extended precision
oxiblas-core = { version = "0.1", features = ["f16", "f128"] }

# With parallelization
oxiblas-core = { version = "0.1", features = ["parallel"] }

# All features
oxiblas-core = { version = "0.1", features = ["f16", "f128", "parallel"] }
```

## Usage

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
use oxiblas_core::simd::{SimdType, SimdOps};

// Automatic SIMD selection based on platform
let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
let y: Vec<f64> = vec![5.0, 6.0, 7.0, 8.0];
let mut result = vec![0.0; 4];

// Uses AVX2/NEON automatically if available
unsafe {
    let simd = <f64 as SimdType>::simd();
    simd.fma(&x, &y, &mut result);
    // result = x * y + result
}
```

### Extended Precision

```rust
use oxiblas_core::scalar::QuadFloat;

#[cfg(feature = "f128")]
{
    // Quad precision (f128) - ~31 decimal digits
    let x = QuadFloat::from(2.0);
    let sqrt_x = x.sqrt();
    println!("√2 = {}", sqrt_x); // Very high precision
}
```

### Kahan Summation

```rust
use oxiblas_core::scalar::kahan_sum;

let values: Vec<f64> = vec![1.0, 1e-16, -1.0]; // Difficult for naive sum
let result = kahan_sum(&values); // Accurate result using compensated summation
```

### Cache Detection

```rust
use oxiblas_core::tuning::detect_cache_sizes;

let cache = detect_cache_sizes();
println!("L1D: {} KB", cache.l1d / 1024);
println!("L2:  {} KB", cache.l2 / 1024);
println!("L3:  {} KB", cache.l3 / 1024);
```

### Blocking Parameters

```rust
use oxiblas_core::blocking::BlockParams;

// Get optimal blocking parameters for GEMM
let params = BlockParams::for_gemm::<f64>();
println!("MC={}, KC={}, NC={}", params.mc, params.kc, params.nc);
// Automatically tuned for your system's cache hierarchy
```

## Feature Flags

| Feature | Description | Default |
|---------|-------------|---------|
| `default` | Core functionality (f32, f64, complex) | ✓ |
| `parallel` | Rayon-based parallelization | |
| `f16` | Half-precision (16-bit) floating point | |
| `f128` | Quad-precision (~31 digits) via double-double | |
| `nightly` | Nightly-only optimizations | |
| `force-scalar` | Disable SIMD, use scalar only (debug) | |
| `max-simd-128` | Limit to 128-bit SIMD (SSE/NEON) | |
| `max-simd-256` | Limit to 256-bit SIMD (AVX2) | |

## SIMD Support Matrix

| Platform | 128-bit | 256-bit | 512-bit |
|----------|---------|---------|---------|
| x86_64 (SSE4.1) | ✓ | | |
| x86_64 (AVX2) | ✓ | ✓ | |
| x86_64 (AVX-512) | ✓ | ✓ | ✓ |
| AArch64 (NEON) | ✓ | | |
| AArch64 (SVE) | ✓ | Planned | |
| Fallback (scalar) | ✓ | | |

## Performance

### SIMD Performance (Apple M3, NEON)

| Operation | Size | Scalar | NEON (128-bit) | Speedup |
|-----------|------|--------|----------------|---------|
| f64 Add | 4,096 | 15.2 µs | 7.98 µs | 1.9× |
| f64 FMA | 4,096 | 22.1 µs | 11.29 µs | 2.0× |
| f32 Add | 4,096 | 8.1 µs | 3.2 µs | 2.5× |
| f32 FMA | 4,096 | 11.5 µs | 4.8 µs | 2.4× |

### SIMD Performance (Linux x86_64, AVX2)

| Operation | Size | Scalar | AVX2 (256-bit) | Speedup |
|-----------|------|--------|----------------|---------|
| f64 Add | 4,096 | 18.4 µs | 7.98 µs | 2.3× |
| f64 FMA | 4,096 | 26.7 µs | 11.29 µs | 2.4× |
| f32 Add | 4,096 | 9.8 µs | 2.1 µs | 4.7× |
| f32 FMA | 4,096 | 14.2 µs | 3.2 µs | 4.4× |

## Architecture

```
oxiblas-core/
├── scalar.rs          # Scalar trait, f16, f128, extended precision
├── simd.rs            # SIMD abstraction layer
├── simd/
│   ├── avx2.rs        # AVX2/FMA kernels (x86_64)
│   ├── avx512.rs      # AVX-512 kernels (x86_64)
│   ├── neon.rs        # NEON kernels (AArch64)
│   └── scalar.rs      # Fallback scalar implementation
├── memory/
│   ├── align.rs       # Aligned allocation
│   ├── workspace.rs   # Temporary buffer management
│   └── cache.rs       # Cache-aware utilities
├── blocking.rs        # Blocking parameter calculation
├── tuning.rs          # Platform detection and auto-tuning
└── parallel.rs        # Parallel operations with Rayon
```

## Supported Platforms

### Tier 1 (Fully Tested)
- **x86_64**: Linux, macOS, Windows
- **AArch64**: macOS (Apple Silicon), Linux

### Tier 2 (Best Effort)
- **x86**: Linux, Windows
- **AArch64**: Android, iOS
- **RISC-V**: Linux (scalar only)

## Requirements

- **Rust**: 1.85+ (Edition 2024)
- **No external C dependencies**
- **Optional**: OpenMP or Rayon for parallelization

## Examples

See the [examples directory](../../examples/) in the main repository:

- `basic_simd.rs` - SIMD operations
- `extended_precision.rs` - f16 and f128 usage
- `cache_tuning.rs` - Platform-specific optimization

## Benchmarks

Run benchmarks:

```bash
# SIMD benchmarks
cargo bench --package oxiblas-core --bench simd

# Blocking parameter benchmarks
cargo bench --package oxiblas-core --bench blocking
```

## Safety

- All SIMD operations are properly marked `unsafe` where required
- Memory alignment is enforced at compile-time where possible
- Extensive testing across platforms ensures correctness
- No undefined behavior in safe APIs

## Contributing

Contributions are welcome! Areas of interest:

1. **ARM SVE support** - Scalable Vector Extension for future ARM
2. **RISC-V vector** - Vector extension support
3. **Additional extended precision** - Alternative quad-float implementations
4. **Auto-tuning improvements** - Better platform detection

## Related Crates

- [`oxiblas-matrix`](../oxiblas-matrix/) - Matrix types built on oxiblas-core
- [`oxiblas-blas`](../oxiblas-blas/) - BLAS operations using oxiblas-core
- [`oxiblas-lapack`](../oxiblas-lapack/) - LAPACK decompositions
- [`oxiblas`](../oxiblas/) - Meta-crate with unified API

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.

## References

- [BLIS Design](https://github.com/flame/blis) - Blocking and micro-kernel design inspiration
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/) - x86_64 SIMD reference
- [ARM NEON Intrinsics](https://developer.arm.com/architectures/instruction-sets/intrinsics/) - AArch64 SIMD reference
- [Kahan Summation](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) - Compensated summation algorithm
