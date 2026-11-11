# OxiBLAS Benchmarks

Comprehensive performance benchmarks for the OxiBLAS library using Criterion.rs, including direct comparisons with OpenBLAS.

## Quick Start

```bash
# Run all OxiBLAS benchmarks (no external dependencies required)
cargo bench --package oxiblas-benchmarks

# Compare OxiBLAS with OpenBLAS (requires OpenBLAS installed)
brew install openblas  # macOS (or apt-get install libopenblas-dev on Linux)
cargo bench --package oxiblas-benchmarks --bench comparison --features compare-openblas

# View results
open target/criterion/report/index.html
```

## Organization

Benchmarks are organized into separate files by category:

- **`blas_level1.rs`** - Vector-vector operations (dot, axpy, scal, nrm2)
- **`blas_level2.rs`** - Matrix-vector operations (gemv, ger)
- **`blas_level3.rs`** - Matrix-matrix operations (gemm, trmm, gemm3m)
- **`lapack_qr.rs`** - QR factorizations (standard, pivot, LQ, RQ, COD)
- **`lapack_svd.rs`** - Singular Value Decomposition
- **`lapack_factorization.rs`** - LU and Cholesky factorizations
- **`lapack_evd.rs`** - Eigenvalue decompositions
- **`lapack_solve.rs`** - Linear system solvers
- **`sparse_ops.rs`** - Sparse matrix operations and iterative solvers
- **`sparse_eigen_svd.rs`** - Sparse eigenvalue and SVD solvers
- **`size_variations.rs`** - GEMM at tiny, non-power-of-2, rectangular and large sizes
- **`precision_bench.rs`** - f32 vs f64 GEMM, f16 storage, mixed-precision LU
- **`new_features.rs`** - Recently added features (einsum, extended precision, tensor ops)
- **`advanced_features.rs`** - Parallel Level 1, Strassen, asymmetric GEMM, TRSM
- **`comparison.rs`** - Direct speed comparisons with OpenBLAS (requires `compare-openblas` feature)

## Running Benchmarks

### Run all benchmarks

```bash
cargo bench --package oxiblas-benchmarks
```

### Run specific benchmark suite

```bash
# BLAS Level 1 operations
cargo bench --package oxiblas-benchmarks --bench blas_level1

# BLAS Level 3 operations
cargo bench --package oxiblas-benchmarks --bench blas_level3

# QR factorizations
cargo bench --package oxiblas-benchmarks --bench lapack_qr

# New features
cargo bench --package oxiblas-benchmarks --bench new_features
```

### Run specific benchmark within a suite

```bash
# Benchmark only GEMM
cargo bench --package oxiblas-benchmarks --bench blas_level3 -- gemm_f64

# Benchmark only dot product
cargo bench --package oxiblas-benchmarks --bench blas_level1 -- dot_product
```

### Filter by size

```bash
# Benchmark GEMM only for size 256
cargo bench --package oxiblas-benchmarks --bench blas_level3 -- gemm_f64/256
```

## Comparison Benchmarks with OpenBLAS

### Prerequisites

The comparison loads an OpenBLAS shared library at run time with `dlopen`, so nothing is linked at build time. It runs on Unix only. Install OpenBLAS on your system:

**macOS:**
```bash
brew install openblas
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libopenblas-dev
```

**Fedora/RHEL:**
```bash
sudo dnf install openblas-devel
```

**Arch Linux:**
```bash
sudo pacman -S openblas
```

The library is taken from the `OPENBLAS_LIB` environment variable if that is set. Otherwise the default loader search for `libopenblas` is tried, then `brew --prefix openblas`.

OpenBLAS must not be linked into this benchmark. `oxiblas-blas` exports `#[no_mangle]` `cblas_*` symbols, so any `cblas_*` reference in the binary binds to OxiBLAS at link time and OpenBLAS never runs. That is what happened before 0.2.3. The benchmark now panics at startup if a routine resolves into its own binary, and it prints the library path, build configuration and kernel in use:

```text
OpenBLAS under test: /path/to/libopenblasp-r0.3.30.dylib
  config: OpenBLAS 0.3.30 DYNAMIC_ARCH NO_AFFINITY USE_OPENMP neoversen1 MAX_THREADS=56
  kernel: neoversen1, sequential groups measured at 1.00 cores
```

### Running Comparisons

```bash
# Run all comparison benchmarks
cargo bench --package oxiblas-benchmarks --bench comparison --features compare-openblas

# Run specific operation comparison
cargo bench --package oxiblas-benchmarks --bench comparison --features compare-openblas -- gemm_comparison

# Run comparison for specific size
cargo bench --package oxiblas-benchmarks --bench comparison --features compare-openblas -- 'gemm_comparison/.*/512'

# Use a specific OpenBLAS build
OPENBLAS_LIB=/path/to/libopenblas.so cargo bench --package oxiblas-benchmarks --bench comparison --features compare-openblas
```

Both libraries work on the same `Mat` buffers, so layout, alignment and leading dimensions match. The sequential groups run OpenBLAS on one thread, whatever `OPENBLAS_NUM_THREADS` says, because the OxiBLAS calls there are sequential and allocate only packing buffers: startup sets the count through `openblas_set_num_threads` and, in OpenMP builds, `omp_set_num_threads`, and panics if a threaded GEMM still uses more than one core of CPU time. The `*_parallel_comparison` groups give both libraries the same thread count, as described below.

This crate enables `oxiblas-blas/parallel`. Before the 0.2.3 TRSM rewrite, `trsm_in_place` then ran its GEMM updates on the Rayon pool from order 256 without being asked, so the published one-thread DTRSM figures at n = 256 and 512 were not single-threaded. TRSM is sequential now; `trsm_with_par` takes an explicit `Par`.

The GEMM groups count `n³` elements per call, so GFLOPS = 2 × Gelem/s.

The parallel groups use every available core. Set `RAYON_NUM_THREADS` to change that: OpenBLAS follows the same variable, so both libraries get the same thread count. On CPUs that mix performance and efficiency cores, compare more than one count, because a thread count that suits one library may not suit the other.

### Comparison Benchmark Coverage

The comparison suite benchmarks the following operations against OpenBLAS:

| Operation | Sizes | Description |
|-----------|-------|-------------|
| **DGEMM** | 64, 128, 256, 512, 1024 | Square matrix multiplication |
| **DGEMM (rect)** | 128×256×128, 256×512×256, 512×256×512 | Rectangular matrices |
| **SGEMM** | 64, 128, 256, 512, 1024 | Single precision matrix multiplication |
| **DGEMM, SGEMM (parallel)** | 512, 1024, 2048 | Matrix multiplication on every available core, or `RAYON_NUM_THREADS` threads |
| **DTRSM** | 64, 128, 256, 512 | Triangular solve with multiple right-hand sides |
| **DSYRK** | 64, 128, 256, 512 | Symmetric rank-k update |
| **DGEMV** | 100, 500, 1K, 5K, 10K | Matrix-vector multiplication |
| **DDOT** | 1K, 10K, 100K, 1M | Vector dot product |
| **DAXPY** | 1K, 10K, 100K, 1M | Vector addition |
| **DNRM2** | 1K, 10K, 100K, 1M | Vector Euclidean norm |

### Interpreting Comparison Results

Criterion will display results for both implementations:

```text
gemm_comparison/oxiblas/256
                        time:   [667.65 µs 668.82 µs 670.34 µs]
gemm_comparison/openblas/256
                        time:   [618.16 µs 619.23 µs 620.44 µs]
```

**Performance Analysis:**
- **OxiBLAS: 668.82 µs, OpenBLAS: 619.23 µs** → OpenBLAS is ~8% faster
- **Ratio = 619.23 / 668.82 = 93%** → OxiBLAS runs at 93% of OpenBLAS's speed

**Performance Ratio Interpretation:**

The published tables use this ratio, OxiBLAS's speed as a percentage of OpenBLAS's:

- **Above 100%**: OxiBLAS is faster than OpenBLAS
- **About 100%**: Similar performance (~5% difference)
- **Below 100%**: OpenBLAS is faster
- **Above 90%**: Competitive (within about 10%)
- **Above 67%**: Acceptable for pure Rust (OxiBLAS takes up to 1.5 times as long)
- **Below 50%**: Significant gap, optimization needed

### Current Results

Measured figures are in [Performance](../../README.md#performance) in the root README. Run the suite on your own hardware for numbers that apply to it.

## Viewing Results

After running benchmarks, Criterion generates HTML reports in:
```text
target/criterion/
```

Open `target/criterion/report/index.html` in your browser to view detailed results with:
- Performance graphs
- Statistical analysis
- Comparison with previous runs
- Throughput measurements

## Benchmark Coverage

### BLAS Level 1
- Vector sizes: 100, 1,000, 10,000, 100,000
- Operations: dot, axpy, scal, nrm2

### BLAS Level 2
- Matrix sizes: 50×50, 100×100, 200×200, 500×500
- Operations: gemv, ger

### BLAS Level 3
- Square matrices: 32×32, 64×64, 128×128, 256×256, 512×512
- Rectangular: 64×128×64, 128×64×128, 256×128×64
- Operations: gemm (f64), gemm3m (complex), trmm

### LAPACK
- QR factorizations: 50×50, 100×100, 200×200, 500×500
- Variants: Standard QR, QR with pivot, LQ, RQ, COD
- SVD (`lapack_svd`, full-rank fixed-seed random matrices): `svd_jacobi` (one-sided Jacobi `Svd`) at 50, 100 and 200, `svd_divide_conquer` (`SvdDc`) at 50, 100, 200 and 500, and the `svd_algorithm_comparison` group
- Symmetric EVD (`lapack_evd`): `symmetric_evd` and `symmetric_evd_dc` at 50-200, `symmetric_evd_large` and `symmetric_evd_dc_large` at 300, 500 and 1000, `symmetric_evd_random` and `symmetric_evd_dc_random` at 100, 200, 500 and 1000 (10 samples each for the large and random groups); general EVD, Hessenberg and Schur groups as well

The `evd_svd_bench` example of the `oxiblas` crate times the same four solvers at 200-2000 and also prints residual and orthogonality errors: `cargo run --release -p oxiblas --example evd_svd_bench -- [--reps N] [SIZE ...]`.

### New Features
- Extended precision dot products (Kahan, pairwise, mixed precision)
- Einsum patterns (matmul, outer product, Hadamard)
- Tensor operations (batched matmul)
- Outer product scaling

## Customization

To add new benchmarks:

1. Create a new file in `benches/` directory
2. Add benchmark entry in `Cargo.toml`:
   ```toml
   [[bench]]
   name = "your_benchmark"
   harness = false
   ```
3. Import criterion and implement benchmarks:
   ```rust,ignore
   use criterion::{criterion_group, criterion_main, Criterion};

   fn your_bench(c: &mut Criterion) {
       c.bench_function("operation", |b| {
           b.iter(|| {
               // Your operation here
           });
       });
   }

   criterion_group!(benches, your_bench);
   criterion_main!(benches);
   ```

## Performance Tips

For accurate benchmarks:

1. **Close background applications** that might affect CPU performance
2. **Disable CPU frequency scaling** if possible
3. **Stay on mains power** (laptops)
4. **Use release mode** (always, criterion handles this automatically)
5. **Multiple runs**: Criterion runs each benchmark multiple times for statistical accuracy

## Continuous Integration

For CI environments, use `--bench` mode without HTML reports:

```bash
cargo bench --package oxiblas-benchmarks --no-fail-fast -- --noplot
```

## Comparison with Previous Runs

Criterion automatically compares with previous benchmark runs. To save a baseline:

```bash
cargo bench --package oxiblas-benchmarks -- --save-baseline my-baseline
```

To compare against a baseline:

```bash
cargo bench --package oxiblas-benchmarks -- --baseline my-baseline
```

## Performance Analysis Workflow

### 1. Establish Baseline

Before making changes:

```bash
# Run all benchmarks and save as baseline
cargo bench --package oxiblas-benchmarks -- --save-baseline before-optimization

# Or just save comparison benchmarks
cargo bench --package oxiblas-benchmarks --bench comparison --features compare-openblas -- --save-baseline before-opti
```

### 2. Make Code Changes

Edit optimization code, implement SIMD kernels, etc.

### 3. Run Comparison

```bash
# Compare with baseline
cargo bench --package oxiblas-benchmarks -- --baseline before-optimization

# Look for lines showing improvement/regression
# Example: "change: -15.2%" means 15.2% faster (good!)
#          "change: +12.4%" means 12.4% slower (regression)
```

### 4. Analyze HTML Reports

Open `target/criterion/<benchmark_name>/report/index.html` to see:
- **Violin plots** - Distribution of measurements
- **Performance graphs** - Time vs input size
- **Statistical analysis** - Confidence intervals
- **Historical comparison** - Trend over time

### 5. Identify Bottlenecks

```bash
# Profile with perf (Linux)
cargo bench --package oxiblas-benchmarks --bench blas_level3 --profile-time 10

# Or use flamegraph
cargo install flamegraph
cargo flamegraph --bench blas_level3 --package oxiblas-benchmarks
```

## Troubleshooting

### OpenBLAS Not Found

**Error:** `could not load OpenBLAS (tried [...]); set OPENBLAS_LIB to its path`

**Solution:**
```bash
# macOS: Check installation
brew list openblas

# Linux: Install the library
sudo apt-get install libopenblas-dev

# Point the benchmark at the library
export OPENBLAS_LIB=/path/to/libopenblas.so
```

### Inconsistent Results

**Problem:** Benchmark times vary significantly between runs

**Solutions:**
1. **Close background apps** - Browsers, IDEs, etc.
2. **Disable CPU frequency scaling**:
   ```bash
   # Linux
   sudo cpupower frequency-set --governor performance

   # macOS (limited control)
   sudo pmset -a hibernatemode 0
   ```
3. **Run longer**: Let Criterion collect more samples
   ```bash
   cargo bench --package oxiblas-benchmarks -- --warm-up-time 5 --measurement-time 10
   ```
4. **Check thermal throttling**: Monitor CPU temperature

### Benchmarks Too Slow

**Problem:** Benchmarks take too long to run

**Solutions:**
```bash
# Run specific benchmarks only
cargo bench --package oxiblas-benchmarks --bench blas_level1

# Reduce sample size (less accurate)
cargo bench --package oxiblas-benchmarks -- --sample-size 10

# Quick mode for CI
cargo bench --package oxiblas-benchmarks -- --quick
```

## Advanced Usage

### Custom Benchmark Configuration

Create `Criterion.toml` in the benchmarks directory:

```toml
[default]
warm_up_time = { secs = 3, nanos = 0 }
measurement_time = { secs = 5, nanos = 0 }
sample_size = 100
confidence_level = 0.95
noise_threshold = 0.01
```

### Export Results to CSV

```bash
# Run with CSV export
cargo bench --package oxiblas-benchmarks -- --output-format csv > results.csv

# Process with external tools
python analyze_results.py results.csv
```

### Comparing Multiple Baselines

```bash
# Save baseline for different configurations
cargo bench --package oxiblas-benchmarks -- --save-baseline no-simd
cargo bench --package oxiblas-benchmarks --features simd -- --save-baseline with-simd

# Compare baselines
critcmp no-simd with-simd
```

## Contributing Benchmarks

When adding new benchmarks, follow these guidelines:

### 1. Structure

```rust,ignore
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

fn bench_operation(c: &mut Criterion) {
    let mut group = c.benchmark_group("operation_name");

    for size in [small, medium, large].iter() {
        // Set throughput for meaningful metrics
        group.throughput(Throughput::Elements(*size as u64));

        // Setup data
        let data = setup_data(*size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                // Use black_box to prevent optimization
                operation(black_box(&data));
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_operation);
criterion_main!(benches);
```

### 2. Best Practices

- **Use `black_box`** to prevent compiler optimization
- **Set throughput** for operations/sec metrics
- **Multiple sizes** to show scaling behavior
- **Realistic data** - avoid all-zeros or sequential patterns
- **Warm-up** included - let Criterion handle it
- **Document** what you're measuring in comments

### 3. Naming Conventions

- **Group name**: `operation_type` (e.g., `gemm_f64`, `qr_decomposition`)
- **Parameter**: Size or configuration (e.g., `256`, `1000x1000`)
- **Benchmark ID**: Descriptive (e.g., `gemm_f64/512`)

### 4. Performance Metrics

Always include:
- **Throughput**: `Throughput::Elements(n)` or `Throughput::Bytes(n)`
- **Multiple sizes**: Show scaling characteristics
- **Realistic workloads**: Match actual use cases

## Continuous Integration

### GitHub Actions Example

```yaml
name: Benchmarks

on:
  push:
    branches: [main]
  pull_request:

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Install OpenBLAS
        run: sudo apt-get install libopenblas-dev

      - name: Run benchmarks
        run: |
          cargo bench --package oxiblas-benchmarks --no-fail-fast -- --noplot
          cargo bench --package oxiblas-benchmarks --features compare-openblas --no-fail-fast -- --noplot

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: target/criterion/
```

## Performance Goals

The results come from `comparison.rs` and are published in [Performance](../../README.md#performance) in the root README. The gaps listed here before 0.2.3 (a full-GEMM SYRK, a slow TRSM and GEMV, DOT at about half of OpenBLAS and NRM2 at about two thirds, small GEMM at 73-85%) were the targets of the 0.2.3 kernel work: SYRK is triangle-only, TRSM is blocked with packed diagonal blocks, small GEMM lost most of its per-call overhead, and aarch64 has NEON DOT, NRM2 and GEMV kernels. What remains, measured on an Apple M3 on 2026-11-09:

- GEMV at 5000×5000 runs at 97% of OpenBLAS (3.64 ms against 3.53 ms), the one GEMV size where OpenBLAS is ahead.
- f64 GEMM on one thread is at 92-99% of OpenBLAS's faster per-process level, and rectangular GEMM at 96-98%.
- Complex GEMM and the complex blocked SYRK path run on a scalar 2×2 complex micro-kernel.
- x86_64 has not been compared with OpenBLAS since the comparison was fixed in 0.2.3, and the new Level-1/2 kernels are aarch64 only.

## See Also

- [TODO.md](TODO.md) - Planned benchmark improvements
- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [OpenBLAS](https://www.openblas.net/)
- [BLIS](https://github.com/flame/blis)
- [Performance tuning guide](../../docs/performance.md) _(planned)_
