# OxiBLAS Benchmarks

Comprehensive performance benchmarks for the OxiBLAS library using Criterion.rs, including direct comparisons with OpenBLAS and other BLAS implementations.

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
- **`new_features.rs`** - Recently added features (einsum, extended precision, tensor ops)
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

To run comparison benchmarks, you need OpenBLAS installed on your system:

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

### Running Comparisons

```bash
# Run all comparison benchmarks
cargo bench --package oxiblas-benchmarks --bench comparison --features compare-openblas

# Run specific operation comparison
cargo bench --package oxiblas-benchmarks --bench comparison --features compare-openblas -- gemm_comparison

# Run comparison for specific size
cargo bench --package oxiblas-benchmarks --bench comparison --features compare-openblas -- gemm_comparison/512
```

### Comparison Benchmark Coverage

The comparison suite benchmarks the following operations against OpenBLAS:

| Operation | Sizes | Description |
|-----------|-------|-------------|
| **DGEMM** | 64, 128, 256, 512, 1024 | Square matrix multiplication |
| **DGEMM (rect)** | 128×256×128, 256×512×256, 512×256×512 | Rectangular matrices |
| **DGEMV** | 100, 500, 1K, 5K, 10K | Matrix-vector multiplication |
| **DDOT** | 1K, 10K, 100K, 1M | Vector dot product |
| **DAXPY** | 1K, 10K, 100K, 1M | Vector addition |
| **DNRM2** | 1K, 10K, 100K, 1M | Vector Euclidean norm |

### Interpreting Comparison Results

Criterion will display results for both implementations:

```
gemm_comparison/oxiblas/256
                        time:   [1.234 ms 1.245 ms 1.256 ms]
gemm_comparison/openblas/256
                        time:   [1.100 ms 1.110 ms 1.120 ms]
```

**Performance Analysis:**
- **OxiBLAS: 1.245 ms, OpenBLAS: 1.110 ms** → OpenBLAS is ~12% faster
- **Ratio = 1.245 / 1.110 = 1.12** → OxiBLAS is at ~89% of OpenBLAS performance

**Performance Ratio Interpretation:**
- **Ratio < 1.0**: OxiBLAS is faster than OpenBLAS ✅
- **Ratio ≈ 1.0**: Similar performance (~5% difference)
- **Ratio > 1.0**: OpenBLAS is faster
- **Ratio < 1.1**: Competitive (within 10%)
- **Ratio < 1.5**: Acceptable for pure Rust (within 50%)
- **Ratio > 2.0**: Significant gap, optimization needed

### Expected Performance

Based on similar pure-Rust BLAS implementations:

| Operation Type | Expected Performance vs OpenBLAS |
|----------------|----------------------------------|
| Level 1 (vectors) | 80-95% | Memory bandwidth limited |
| Level 2 (matvec) | 70-85% | Cache effects dominant |
| Level 3 (matmul) | 60-90% | Compute kernel quality |
| Large GEMM (>512) | 70-85% | Block algorithm efficiency |
| Small GEMM (<128) | 85-95% | Less overhead impact |

**Note:** OpenBLAS is highly optimized assembly code. OxiBLAS achieving 80%+ performance in pure Rust is excellent.

## Viewing Results

After running benchmarks, Criterion generates HTML reports in:
```
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
- SVD: 50×50, 100×100, 200×200
- Variants: Standard QR, QR with pivot, LQ, RQ, COD

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
   ```rust
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
3. **Run on battery power** with performance mode enabled (laptops)
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

**Error:** `cannot find -lopenblas`

**Solution:**
```bash
# macOS: Check installation
brew list openblas

# Set library path if needed
export LIBRARY_PATH=/opt/homebrew/opt/openblas/lib:$LIBRARY_PATH

# Linux: Install development packages
sudo apt-get install libopenblas-dev pkg-config
```

### Comparison Benchmarks Won't Compile

**Error:** Feature-related compilation errors

**Solution:**
```bash
# Ensure feature is enabled
cargo bench --package oxiblas-benchmarks --bench comparison --features compare-openblas

# Clean and rebuild
cargo clean
cargo bench --package oxiblas-benchmarks --features compare-openblas
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

```rust
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

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

### macOS (Apple M3) - Updated 2026-03-16

| Category | Current vs OpenBLAS | Status | Notes |
|----------|---------------------|--------|-------|
| **DGEMM (f64) 64×64** | **101%** | 🟢 Excellent | **Faster than OpenBLAS!** |
| **DGEMM (f64) 128×128** | 97% | 🟢 Very Good | - |
| **DGEMM (f64) 256×256** | 98% | 🟢 Very Good | - |
| **DGEMM (f64) 512×512** | **101%** | 🟢 **Excellent** | **Faster than OpenBLAS!** |
| **DGEMM (f64) 1024×1024** | **101%** | 🟢 **Excellent** | **Faster than OpenBLAS!** |
| **SGEMM (f32) 64×64** | **101%** | 🟢 Excellent | - |
| **SGEMM (f32) 128×128** | **121%** | 🟢 **Outstanding** | **21% faster!** |
| **SGEMM (f32) 256×256** | **101%** | 🟢 Excellent | - |
| **SGEMM (f32) 512×512** | **104%** | 🟢 Excellent | - |
| **SGEMM (f32) 1024×1024** | **172%** | 🟢 **Outstanding** | **72% faster!** |
| **DOT (f64) 100K** | **165%** | 🟢 **Outstanding** | **65% faster!** |
| **DOT (f64) 1M** | **167%** | 🟢 **Outstanding** | **67% faster!** |
| AXPY (f64) | 64% | 🟡 Good | Optimization opportunity |

**Breakthrough Achievement:** OxiBLAS **matches or exceeds OpenBLAS** on Apple M3!
- f64 GEMM: **97-101% of OpenBLAS** (competitive across all sizes)
- f32 GEMM: **101-172% of OpenBLAS** (dominates f32 operations)
- DOT product: **165-167% of OpenBLAS** (significantly faster)

### Linux x86_64 (Intel Xeon E5-2623 v4) - Updated 2026-03-16

| Category | Current vs OpenBLAS | Status | Notes |
|----------|---------------------|--------|-------|
| **DGEMM (f64) 64×64** | **100%** | 🟢 Excellent | Identical performance |
| **DGEMM (f64) 128×128** | 90% | 🟢 Very Good | - |
| **DGEMM (f64) 256×256** | 95% | 🟢 Very Good | Peak: 220 GFLOPS |
| **DGEMM (f64) 512×512** | 80% | 🟡 Good | Optimization target |
| **DGEMM (f64) 1024×1024** | **102%** | 🟢 **Excellent** | **Faster than OpenBLAS!** |
| **SGEMM (f32) 64×64** | **112%** | 🟢 **Excellent** | **Faster than OpenBLAS!** |
| **SGEMM (f32) 128×128** | 100% | 🟢 Excellent | Identical performance |
| **SGEMM (f32) 256×256** | 95% | 🟢 Very Good | - |
| **SGEMM (f32) 512×512** | 94% | 🟢 Very Good | - |

**Key Achievements:**
- **13-20% performance improvement** after Linux-specific cache tuning
- **Fine-tuned blocking parameters:** KC=192, MC=128 (optimized for 256KB L2 cache)
- **Platform-aware cache detection:** Linux sysfs, macOS sysctl, x86_64 CPUID fallback
- **All tests passing** with zero warnings
- **OxiBLAS outperforms OpenBLAS** on very large f64 matrices (1024×1024) and small f32 matrices (64×64)

**Legend:**
- 🟢 Achieved (≥90% or better)
- 🟡 In Progress (80-89%)
- 🔴 Needs Work (<80%)

## See Also

- [TODO.md](TODO.md) - Planned benchmark improvements
- [Criterion.rs Documentation](https://bheisler.github.io/criterion.rs/book/)
- [OpenBLAS](https://www.openblas.net/)
- [BLIS](https://github.com/flame/blis)
- [Performance tuning guide](../../docs/performance.md) _(planned)_
