# OxiBLAS Benchmarks TODO

Tracking performance benchmarking improvements and comparative analysis.

## Current Status (Updated 2025-12-27)

- **12 benchmark suites**: Level 1, Level 2, Level 3, QR, SVD, Factorization, EVD, Solve, Sparse Ops, Sparse Eigen/SVD, New Features, Comparison
- **121 benchmark functions** across all suites
- **1 comparison suite**: OpenBLAS comparison (optional feature)
- **Coverage**:
  - BLAS L1/L2/L3 (complete, including Hermitian operations)
  - LAPACK QR/SVD/LU/Cholesky/EVD/Solve (complete)
  - Sparse operations: SpMV, SpMM, sparse add, triangular solve, Cholesky, format conversions
  - Sparse iterative: CG, PCG, BiCGStab, GMRES, MINRES, IDR(s), TFQMR, QMR, Block-CG, Block-GMRES
  - Sparse eigen/SVD: Lanczos, Arnoldi, IRAM, Truncated SVD, Randomized SVD
  - Sparse factorizations: LU, QR, ILU0, ILUT
  - Extended precision (f16, f32, f64), tensor operations
- **Framework**: Criterion.rs with HTML reports and statistical analysis
- **Performance**: DGEMM 79% of OpenBLAS (large matrices), 6-15× speedup in GEMM-based ops

---

## TIER 1 - HIGH PRIORITY

### Comparison Benchmarks
- [x] OpenBLAS comparison for BLAS operations (GEMM, GEMV, DOT, AXPY, NRM2)
- [ ] MKL comparison benchmarks (Intel Math Kernel Library)
- [ ] BLIS comparison benchmarks (BLAS-like Library Instantiation Software)
- [ ] Accelerate framework comparison (macOS)
- [ ] Comprehensive comparison report generation

### Performance Regression Tracking
- [ ] Automated baseline storage and comparison
- [ ] CI/CD integration for benchmark regression detection
- [ ] Historical performance tracking database
- [ ] Performance regression alerts (e.g., >5% slowdown)
- [ ] Automatic GitHub issue creation on regression
- [ ] Performance dashboard (web-based visualization)

### Missing Operation Coverage
- [x] BLAS Level 1: swap ✅ Added 2025-12-27
- [x] BLAS Level 1: copy ✅ Added 2025-12-27
- [x] BLAS Level 1: rot ✅ Added 2025-12-27
- [x] BLAS Level 1: rotg ✅ Added 2025-12-27
- [x] BLAS Level 1: rotm ✅ Added 2025-12-27
- [x] BLAS Level 1: rotmg ✅ Added 2025-12-27
- [x] BLAS Level 1: asum ✅ Added 2025-12-27
- [x] BLAS Level 2: syr ✅ Added 2025-12-27
- [x] BLAS Level 2: spr ✅ Added 2025-12-27
- [x] BLAS Level 2: symv ✅ Added 2025-12-27
- [x] BLAS Level 2: sbmv ✅ Added 2025-12-27
- [x] BLAS Level 2: tbmv ✅ Added 2025-12-27
- [x] BLAS Level 2: spmv ✅ Added 2025-12-27
- [x] BLAS Level 2: tpmv ✅ Added 2025-12-27
- [x] BLAS Level 2: tbsv ✅ Added 2025-12-27
- [x] BLAS Level 2: tpsv ✅ Added 2025-12-27
- [x] BLAS Level 2: her, hpr, hbmv, hpmv ✅ Added 2025-12-27
- [x] BLAS Level 3: syrk ✅ Added 2025-12-27
- [x] BLAS Level 3: trsm ✅ Added 2025-12-27
- [x] BLAS Level 3: syr2k ✅ Added 2025-12-27
- [x] BLAS Level 3: symm ✅ Added 2025-12-27
- [x] BLAS Level 3: hemm ✅ Added 2025-12-27
- [x] BLAS Level 3: herk ✅ Added 2025-12-27
- [x] BLAS Level 3: her2k ✅ Added 2025-12-27
- [x] LAPACK: LU factorization ✅ Added 2025-12-27
- [x] LAPACK: Cholesky ✅ Added 2025-12-27
- [x] LAPACK: eigenvalue (symmetric, general, Schur, Hessenberg) ✅ Added 2025-12-27
- [x] LAPACK: solve (triangular, general, tridiagonal, least squares) ✅ Added 2025-12-27
- [x] Sparse operations benchmarks (SpMV, SpMM, sparse add, triangular solve, CG, Cholesky) ✅ Added 2025-12-27
- [x] Iterative solvers: CG, PCG (preconditioned), BiCGStab, GMRES, MINRES, IDR(s), TFQMR, QMR ✅ Added 2025-12-27
- [x] Block iterative solvers: Block-CG, Block-GMRES (multiple RHS) ✅ Added 2025-12-27
- [x] Sparse eigenvalue: Lanczos, Arnoldi, IRAM ✅ Added 2025-12-27
- [x] Sparse SVD: Truncated SVD, Randomized SVD ✅ Added 2025-12-27
- [x] Sparse factorizations: LU, QR, ILU0, ILUT ✅ Added 2025-12-27

---

## TIER 2 - MEDIUM PRIORITY

### Advanced Analysis
- [ ] FLOPS (floating-point operations per second) reporting
- [ ] Cache miss analysis (requires perf integration)
- [ ] Memory bandwidth utilization
- [ ] SIMD instruction utilization analysis
- [ ] Parallel scaling benchmarks (thread count vs performance)
- [ ] Energy consumption measurements (where available)

### Platform-Specific Benchmarks
- [ ] ARM NEON optimization benchmarks
- [ ] AVX-512 optimization benchmarks
- [ ] Apple Silicon (M1/M2/M3) specific tuning
- [ ] x86-64 vs ARM comparison
- [ ] Cross-platform performance report

### Size Variation Testing
- [ ] Very small matrices (2×2, 4×4, 8×8) - overhead analysis
- [ ] Very large matrices (4K×4K, 8K×8K) - scaling analysis
- [ ] Non-power-of-2 sizes (100×100, 300×300, 777×777)
- [ ] Extremely rectangular matrices (10000×10, 10×10000)
- [ ] Memory-bound vs compute-bound analysis

### Accuracy vs Performance Trade-offs
- [ ] f32 vs f64 performance comparison
- [ ] f16 performance (half precision)
- [ ] f128 performance (quad precision)
- [ ] Mixed precision benchmarks
- [ ] Accuracy degradation analysis for fast algorithms

---

## TIER 3 - LOW PRIORITY

### Specialized Workloads
- [ ] Machine learning workload patterns
  - [ ] Convolution operations
  - [ ] Batch normalization
  - [ ] Attention mechanism (transformer)
  - [ ] RNN/LSTM patterns
- [ ] Scientific computing patterns
  - [ ] Iterative solvers convergence speed
  - [ ] Sparse-dense hybrid operations
  - [ ] Graph algorithm performance
- [ ] Financial computing patterns
  - [ ] Monte Carlo simulations
  - [ ] Portfolio optimization
  - [ ] Risk calculations

### Microbenchmarks
- [ ] Memory allocation overhead
- [ ] Cache blocking effectiveness
- [ ] Loop unrolling impact
- [ ] SIMD kernel performance in isolation
- [ ] Prefetching effectiveness

### Compiler Comparison
- [ ] rustc vs gcc/clang (via C FFI)
- [ ] LTO (link-time optimization) impact
- [ ] PGO (profile-guided optimization) impact
- [ ] Different optimization levels (-O2 vs -O3)
- [ ] Different CPU targets (native vs generic)

### Documentation & Reporting
- [ ] Performance tuning guide
- [ ] Architecture-specific optimization recommendations
- [ ] Benchmark result visualization tools
- [ ] Automated performance report generation (Markdown/PDF)
- [ ] Performance comparison tables for documentation

---

## TIER 4 - FUTURE ENHANCEMENTS

### Distributed Computing
- [ ] MPI distributed benchmarks
- [ ] Multi-node scaling analysis
- [ ] Network bandwidth impact

### Real-World Application Benchmarks
- [ ] Deep learning training workload
- [ ] Physics simulation
- [ ] Climate modeling kernel
- [ ] Computational chemistry workload
- [ ] Computer vision pipeline

---

## Infrastructure Requirements

### CI/CD Integration
- [ ] GitHub Actions workflow for benchmarks
- [ ] Dedicated benchmark runner (consistent hardware)
- [ ] Benchmark result storage (artifact uploads)
- [ ] Automated comparison with main branch
- [ ] PR comments with benchmark results

### Benchmark Data Management
- [ ] Database for historical results
- [ ] Result aggregation and analysis scripts
- [ ] Export to CSV/JSON for external analysis
- [ ] Integration with performance tracking services

### Hardware Testing Matrix
- [ ] x86-64 Intel (multiple generations)
- [ ] x86-64 AMD (multiple generations)
- [ ] ARM Cortex-A (various cores)
- [ ] Apple Silicon (M1/M2/M3)
- [ ] Server-grade hardware (Xeon, EPYC)

---

## Performance Targets

| Operation | Current | Target | OpenBLAS | MKL | Notes |
|-----------|---------|--------|----------|-----|-------|
| DGEMM (1024×1024) | TBD | 90% MKL | ~70% MKL | 100% | Square matrix mult |
| DGEMM (tall, 4096×256×256) | TBD | 85% MKL | ~65% MKL | 100% | Tall matrix |
| DGEMM (wide, 256×256×4096) | TBD | 85% MKL | ~65% MKL | 100% | Wide matrix |
| DGEMV (10000×10000) | TBD | 85% MKL | ~70% MKL | 100% | Matrix-vector |
| DDOT (1M elements) | TBD | 95% MKL | ~90% MKL | 100% | Dot product |
| DAXPY (1M elements) | TBD | 95% MKL | ~90% MKL | 100% | Vector add |
| QR (1000×1000) | TBD | 80% MKL | ~65% MKL | 100% | QR factorization |
| SVD (500×500) | TBD | 75% MKL | ~60% MKL | 100% | SVD decomposition |

**Legend:**
- Current: To be measured from comparison benchmarks
- Target: Performance goal relative to MKL (industry standard)
- OpenBLAS: Typical OpenBLAS performance for reference
- MKL: Intel Math Kernel Library (baseline = 100%)

---

## Benchmark Best Practices

### Running Benchmarks
1. **Dedicated Hardware**: Use consistent, idle hardware
2. **Multiple Runs**: Let Criterion handle statistical significance
3. **Thermal Throttling**: Monitor CPU temperature
4. **Background Processes**: Minimize system load
5. **Power Management**: Disable CPU frequency scaling if possible

### Interpreting Results
- **Mean Time**: Primary metric for comparison
- **Standard Deviation**: Measure of consistency
- **Throughput**: Operations per second
- **Comparison Ratio**: Direct performance comparison
- **Regression Detection**: >5% slowdown is significant

### Reporting Issues
When reporting performance issues, include:
- Hardware specifications (CPU model, cores, RAM)
- OS and kernel version
- Compiler version and flags
- Benchmark command used
- Full benchmark output
- Comparison with baseline/competitors

---

## Future Comparison Targets

### Open Source Libraries
- [ ] OpenBLAS (reference implementation)
- [ ] BLIS (BLAS-like Library Instantiation Software)
- [ ] Eigen (C++ template library)
- [ ] Armadillo (C++ linear algebra)
- [ ] GSL (GNU Scientific Library)

### Commercial Libraries
- [ ] Intel MKL (Math Kernel Library)
- [ ] AMD AOCL (Optimizing CPU Libraries)
- [ ] Apple Accelerate Framework
- [ ] ARM Performance Libraries

### Language-Specific Libraries
- [ ] NumPy (Python)
- [ ] SciPy (Python)
- [ ] Julia LinearAlgebra
- [ ] MATLAB (if accessible)

---

## Notes

- Benchmarks should run on CI but **not block** builds (informational only)
- Comparison features are **optional** to avoid mandatory dependencies
- Focus on **reproducibility** and **statistical validity**
- Document **hardware specifications** for all benchmark results
- Maintain **backward compatibility** in benchmark interfaces
