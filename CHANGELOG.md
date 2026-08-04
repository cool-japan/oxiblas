# Changelog

All notable changes to OxiBLAS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

A follow-on hardening pass over the 0.2.2 audit closure: soundness fixes for the
sibling call sites the original audit's fixes missed, plus workspace hygiene
(lint/dependency/CI policy, doctest and file-size compliance, fuzz coverage).
No public API breaking changes; a handful of previously-`unsafe fn`-eligible
constructors are now correctly marked `unsafe fn` (see Changed).

### Fixed
- `PackedRef::new`/`PackedMut::new` (packed.rs) and `BandedRef::new`/`BandedMut::new`
  (banded.rs) were still safe `pub fn` over raw pointers with no validation, even
  though the sibling `MatRef::new`/`MatMut::new` had already been hardened to
  `unsafe fn` — safe code could construct an out-of-bounds view and read/write past
  the backing allocation. All four are now `unsafe fn` with `# Safety` docs.
- `wasm32.rs` SIMD `extract()`/`insert()` still reached `unreachable_unchecked()` on
  an out-of-range lane index from safe caller code in release builds, after the
  identical bug had been fixed for x86_64/aarch64 — ported the
  `lane_index_out_of_range` panic helper to all six wasm32 call sites.
- `MmapMatMut::create`'s size computation (`row_stride * ncols * size_of::<T>()`)
  used unchecked arithmetic, so an oversized `(nrows, ncols)` could wrap to a small
  file while the returned struct still recorded the original huge dimensions,
  letting ordinary safe accessors write past the mapping. Now uses checked
  arithmetic and re-validates the layout against the mapped file size.
- The four CBLAS modules added to close the "CBLAS surface is incomplete" audit
  finding (`hermitian.rs`, `triangular_symmetric.rs`, `complex_level2.rs`,
  `complex_level1.rs`) shipped with zero `lda`/`ldb`/`ldc`/null-pointer validation,
  reproducing the exact bug class `cblas/basic.rs` had already been hardened
  against. Added a shared `cblas/validate.rs` module and wired
  `gemv_params_valid`/`gemm_params_valid`/-family checks plus null checks into
  every `extern "C"` entry point in the four files, no-oping (matching the
  `xerbla`-then-return convention) on invalid input instead of indexing OOB.
- `CscMatrix`'s `Index` impl still panicked via `expect()` on a structurally-zero
  element after the identical `CsrMatrix` impl had been removed for the same
  reason; removed to match.
- `packed_len()` (`n*(n+1)/2`) and the banded `ldab*ncols` allocation-size
  expressions were still plain, wrapping-in-release arithmetic after `mat.rs`'s
  allocations had been routed through a checked `checked_dim_mul` helper; both now
  use checked multiplication and panic on overflow instead of silently
  under-allocating.
- Matrix Market reader (`mtx.rs`) pre-allocated `Vec::with_capacity(header.nnz)`
  directly from the untrusted size line, so a ~30-byte crafted file
  (`3 3 18446744073709551615`) could trigger a capacity-overflow abort before any
  data line was read. `read_header` now rejects `nnz > nrows*ncols` (coordinate
  format, checked with `saturating_mul`) and uses checked multiplication for the
  array-format `nrows*ncols` cell count, so malformed headers return a
  `MtxError` instead of aborting the process.
- 57 of 64 public-API doctests marked `` ```ignore `` (never compiled or verified)
  across `oxiblas-core`, `oxiblas-matrix`, `oxiblas-lapack`, `oxiblas-ndarray`, and
  `oxiblas-sparse` are now real, executed doctests with concrete inputs and
  assertions on the results (the audit that closed this finding had only fixed 7 of
  the 64).
- `oxiblas-core`'s no_std test-code gating was incomplete for the x86_64 SIMD
  modules: `simd/x86_64/functions.rs` (a `thread_local!` override plus 16
  `is_x86_feature_detected!` call sites and a `println!`) and 4 direct-register
  test functions in `simd/complex.rs` compiled only under `std`, but were not
  `#[cfg(feature = "std")]`-gated — invisible to `--no-default-features` checks
  run on an aarch64 host, since that module tree is itself
  `#[cfg(target_arch = "x86_64")]`-gated and was never actually cross-compiled.
  Confirmed and fixed by adding `--target x86_64-apple-darwin` to the no_std
  verification command (see `TODO.md`); see `TODO.md` for the full file list.
- Two pre-existing test/feature-interaction bugs surfaced by the above
  verification pass (not caused by it): `simd::dispatch::tests::test_aarch64_neon_always_present`
  and `simd::multiver::tests::test_aarch64_neon_always_true` both asserted NEON
  is unconditionally present on AArch64 without accounting for `dispatch.rs`'s
  own `limited_to()` masking, which correctly reports `has_neon = false` under
  the `force-scalar` feature (always active under `--all-features`). Fixed by
  checking `SimdCapabilities::simd_ceiling_bytes() >= 16` before asserting,
  mirroring the already-correct sibling test `test_compute_respects_feature_ceiling`.

### Changed
- `PackedRef::new`, `PackedMut::new`, `BandedRef::new`, `BandedMut::new` are now
  `unsafe fn` (see Fixed above) — existing safe call sites inside this workspace
  were updated to `unsafe { ... }` blocks with `// SAFETY:` justifications;
  downstream callers constructing these types directly from raw pointers will need
  to do the same.
- Hoisted the last four per-crate dependency version pins
  (`oxiblas-ndarray`'s `faer`, `oxiblas-benchmarks`'s `ndarray-linalg`/`blas-src`/
  `cblas-sys`) into `[workspace.dependencies]` so every version lives in one place.
- `oxiblas-blas`'s crate-wide `#![allow(clippy::missing_safety_doc)]`,
  `clippy::ptr_as_ptr`, `clippy::transmute_ptr_to_ref`, `clippy::cast_possible_wrap`,
  `clippy::transmute_undefined_repr`, and `clippy::missing_transmute_annotations`
  lint suppressions were removed; the 74 CBLAS entry points that were missing a
  `# Safety` doc section now have one, and 8 NEON transmute call sites in
  `level1/nrm2.rs` now carry explicit `transmute::<From, To>()` turbofish
  annotations instead of relying on suppressed lints.
- Split 4 source files that had grown past the workspace's 2000-line limit into
  cohesive modules with the public API re-exported from the original path:
  `oxiblas-blas/src/level2/gemv.rs`, `oxiblas-blas/src/cblas/basic.rs`,
  `oxiblas-sparse/src/linalg/eigenvalue/tests.rs`, and
  `oxiblas-sparse/src/linalg/eigenvalue/special.rs`.

### Added
- `rustfmt.toml`, `clippy.toml` (MSRV pinned to the workspace's `rust-version`),
  `deny.toml` (COOLJAPAN dependency-ban list, `cargo deny check bans` clean),
  `SECURITY.md`, and `CONTRIBUTING.md` at the workspace root.
- `fuzz/` (an independent `cargo-fuzz` workspace, excluded from the main
  workspace so `cargo build`/`clippy --workspace` never need a nightly
  toolchain) with two real libFuzzer targets: `mtx_matrix_market` (the Matrix
  Market parser) and `mmap_header` (the `.oxiblas` memory-mapped-matrix header
  parser) — the two untrusted-input parsers in the workspace.

## [0.2.2] - 2026-07-18

A production-readiness hardening release. A systematic multi-agent audit swept the
entire workspace (`oxiblas-core`, `oxiblas-matrix`, `oxiblas-blas`, `oxiblas-lapack`,
`oxiblas-ndarray`, `oxiblas-sparse`), followed by adversarial re-verification of every
proposed fix, resulting in a large batch of correctness, honesty, and documentation
fixes. No public API breaking changes.

### Fixed
- Numerous numerical correctness bugs across BLAS, LAPACK, and sparse routines,
  including Hermitian/symmetric diagonal handling, SVD and eigensolver convergence
  and deflation, sparse factorization fill-in, incremental SVD, and IRAM restart
  logic
- Several fabricated or stub code paths replaced with real, verified algorithms
  (e.g. a dead-code MRRR eigensolver path, an untested symmetric divide-and-conquer
  EVD merge, and fake complex-routine aliases)
- Panics on edge-case inputs (empty, singular, or near-singular matrices) converted
  to proper `Result`-based error handling instead of `unwrap()`/`expect()`/`panic!()`
- Assorted silent-fallback and misleading-documentation issues that could mask
  incorrect results

### Changed
- Dependencies updated to latest compatible versions
- Documentation corrected in numerous places to accurately describe actual behavior

### Added
- `[package.metadata.docs.rs]` with `all-features = true` on every publishable
  crate so feature-gated APIs are visible on docs.rs

## [0.2.1] - 2026-03-16

### Fixed
- Fixed compilation error in sparse eigenvalue test imports
- Fixed unnecessary type casts in BLAS level 3 autotune module

### Changed
- Updated dependencies to latest versions

## [0.2.0] - 2026-03-06

### Added

#### LAPACK

- Recursive cache-oblivious factorizations: `Cholesky::compute_recursive()`, `Lu::compute_recursive()`, `Qr::compute_recursive()` - divide-and-conquer algorithms with automatic cache adaptation
- Parallel blocked factorizations: `Cholesky::compute_blocked_par()`, `Lu::compute_blocked_par()` (requires "parallel" feature) - multi-threaded Level 3 BLAS updates via rayon
- Complex bidiagonal reduction: `ComplexBidiagFactors` supporting `Complex64` and `Complex32` matrices
- Mixed-precision iterative refinement variants: `mixed_precision_solve` (LU), `mixed_precision_solve_cholesky`, `mixed_precision_solve_symmetric`, and `mixed_precision_solve_qr` - f32 factorization with f64 residual computation
- LAPACK integration test suite: 61 tests covering LU, Cholesky, QR, SVD, EVD, and solve operations in `tests/lapack_compat.rs`

#### BLAS

- Batched BLAS operations: `gemm_batched`, `gemm_strided_batched`, `axpy_batched`, `gemv_batched` with parallel variants (`gemm_batched_par`, etc.)
- Runtime auto-tuning infrastructure: `RuntimeAutoTuner` for dynamic block size selection, `gemm_auto_tuned()` convenience function (requires "runtime-tuning" feature)

#### Sparse

- Multifrontal sparse factorizations: `MultifrontalCholesky` and `MultifrontalLU` with elimination tree construction and supernodal aggregation
- Advanced sparse LU pivoting strategies: threshold pivoting (`SparseLuThreshold`), static pivoting (`SparseLuStaticPivot` - SuperLU-style), and Bunch-Kaufman LDL^T factorization (`SparseLdlt`)
- Standard test matrix generators: `laplacian_2d`, `laplacian_3d`, `tridiagonal`, `diagonal`, `arrow_matrix`, `random_spd`, `poisson_1d`
- Memory usage integration tests: 27 tests verifying sparse operation memory behavior

#### Core

- Runtime SIMD dispatch infrastructure: `SimdCapabilities` (runtime CPU feature detection), `SimdDispatcher`, `KernelSelector`, and `simd_dispatch!` macro for function multi-versioning
- no_std support for `oxiblas-core` and `oxiblas-matrix`: add `default-features = false` to use in embedded or no-std environments (requires `alloc`)
- Feature constants module in the main `oxiblas` crate: `oxiblas::features::{HAS_PARALLEL, HAS_SPARSE, HAS_F16, HAS_F128, HAS_RUNTIME_TUNING, ...}` for compile-time feature introspection

#### ndarray Integration

- Parallel GEMM: `matmul_par` for ndarray `Array2` (requires "parallel" feature)
- Sparse integration functions: `array2_to_csr`, `csr_to_array2`, `spmv_ndarray`, `sparse_solve_ndarray`

#### Performance Regression Framework

- Performance regression framework: `PerfBaseline`, `PerfMeasurement`, `RegressionChecker` types with JSON storage and configurable degradation threshold
- Performance regression CLI binary (`regress`): subcommands `capture`, `check`, `report`, `list` for CI-integrated throughput tracking
  - `regress capture [--output baseline.json]` — run all quick benchmarks (GEMM f64/f32, Cholesky) and save JSON baseline
  - `regress check [--baseline baseline.json] [--threshold 5.0]` — compare current performance vs baseline, PASS/FAIL per operation
  - `regress report [--baseline baseline.json]` — formatted table of baseline measurements
  - `regress list` — print all available benchmark names

#### BLAS

- SSE4.2 intermediate GEMM micro-kernels: `F64x2Sse` (F64×2, 4×4 tiles) and `F32x4Sse` (F32×4, 4×4 tiles) filling the gap between scalar and AVX2 on x86_64 CPUs without AVX2

#### Core

- NUMA-aware memory allocation: `NumaVec<T>`, `MatNuma<T>`, Linux real NUMA topology detection, NUMA-local allocation fallback
- Thread pool customization: `set_global_thread_pool`, `OxiblasThreadConfig`, `with_thread_count` for fine-grained parallel execution control

#### Documentation and Benchmarks

- Algorithm Selection Guide added to README: when to use blocked vs. recursive vs. parallel variants
- Performance comparison tables in README: LAPACK factorization speedups (Cholesky 6-10×, LU 14-23×, QR 3-7×)
- Library comparison table in README: OxiBLAS vs. ndarray-linalg vs. nalgebra vs. faer across key criteria
- Benchmark size variation suites: tiny/non-power-of-2/rectangular/large matrix benchmarks
- Precision benchmarks: f16/f32/f64/f128 throughput comparisons
- Cross-platform performance comparison: tested platforms, SIMD feature detection summary

### Fixed

- Blocked QR factorization (WY representation): corrected T matrix construction (was using T, must use T^T per DLARFT specification) and corrected block reflector application; 3-7× speedup for large matrices now fully realized

### Changed

- Refactored `oxiblas-core/src/scalar.rs` (2,846 lines) into 8 focused module files under `scalar/` directory; all files remain under the 2,000-line policy limit
- Retired `oxiblas-ffi` crate from workspace (Pure Rust ecosystem policy); crate directory preserved as deprecated archive
- Project statistics updated: ~169,900 lines of Rust across 359 files, 2,835 passing tests + 195 doctests

### Removed

- `oxiblas-ffi` removed from workspace members (`Cargo.toml` members list); the crate directory is retained as a deprecated archive but is no longer built or published

### Code Quality

- Zero `unwrap()` calls in all production code across the entire workspace
- All source files are under 2,000 lines (100% compliant with refactoring policy)
- 16 previously oversized files (51,890 lines total) refactored into 113 modules; v0.2.0 adds the `scalar.rs` refactoring for a total of 114 modules

## [0.1.2] - 2025-12-30

### Added

- **Complex Number Support in ndarray Integration**: Added comprehensive complex-specific LAPACK functions to `oxiblas-ndarray`:
  - `svd_complex_ndarray`: SVD decomposition for complex matrices using one-sided Jacobi algorithm
  - `qr_complex_ndarray`: QR decomposition for complex matrices (unitary Q)
  - `cholesky_hermitian_ndarray`: Cholesky decomposition for Hermitian positive definite matrices
  - `eig_hermitian_ndarray`: Eigenvalue decomposition for Hermitian matrices (real eigenvalues, complex eigenvectors)
  - `ComplexSvdResult` and `HermitianEvdResult` types for complex decomposition results

### Changed

- **Relaxed Trait Bounds**: Removed unnecessary `Real` trait bound from `solve` and `solve_multiple` functions in both `oxiblas-lapack` and `oxiblas-ndarray`, enabling proper support for complex number types (`Complex<f32>`, `Complex<f64>`)

### Fixed

- Minor code formatting improvements in symmetric eigenvalue decomposition tests

## [0.1.1] - 2025-12-29

### Fixed

- **Symmetric Eigenvalue Decomposition**: Fixed critical bug in QR algorithm for tridiagonal matrices where off-diagonal elements were incorrectly stored using `hypot(x, z)` (always positive) instead of `c * x - s * z` (preserves sign). This caused eigenvectors to be computed incorrectly for matrices requiring multiple QR iterations, while eigenvalues remained correct. The fix ensures proper accumulation of Givens rotations into the eigenvector matrix.

## [0.1.0] - 2025-12-27

### Initial Release

OxiBLAS 0.1.0 is the first public release of a pure Rust BLAS/LAPACK implementation.

#### Features

**Core Library (`oxiblas-core`)**
- SIMD layer with AVX2/FMA, AVX-512, and ARM NEON support
- Portable scalar fallback for all platforms
- Extended precision support: f16, f32, f64, f128 (quad precision)
- Complex number support: Complex32, Complex64
- Parallel execution with rayon (optional `parallel` feature)

**BLAS Operations (`oxiblas-blas`)**
- **Level 1** (11 operations): dot, axpy, nrm2, scal, iamax, swap, copy, rot, rotg, rotm, rotmg, asum
- **Level 2** (15 operations): gemv, ger, syr, spr, symv, sbmv, tbmv, tbsv, tpmv, tpsv, spmv, her, hpr, hbmv, hpmv
- **Level 3** (11 operations): gemm, syrk, trsm, syr2k, symm, hemm, herk, her2k
- Complete packed and banded matrix support
- Hermitian and symmetric operations for complex/real types

**LAPACK Operations (`oxiblas-lapack`)**
- **Factorizations**: LU (getrf), Cholesky (potrf), QR (geqrf, geqr2)
- **Decompositions**: SVD (gesvd), Eigenvalue (syev, geev), Schur (gees), Hessenberg (gehrd)
- **Linear Solvers**: Triangular (trtrs), General (gesv, getrs), Tridiagonal (gtsv, ptsv), Least squares (gels)
- **Advanced**: Condition number estimation, matrix inversion, balancing

**Sparse Linear Algebra (`oxiblas-sparse`)**
- **9 sparse formats**: CSR, CSC, COO, ELL, DIA, BSR, BSC, HYB, SELL-C-σ
- **Basic operations**: SpMV (generic + SIMD), SpMM, sparse addition, triangular solve
- **Iterative solvers** (10 variants):
  - CG, PCG (preconditioned conjugate gradient)
  - BiCGStab (stabilized bi-conjugate gradient)
  - GMRES, MINRES (minimal residual methods)
  - IDR(s), TFQMR, QMR (quasi-minimal residual variants)
  - Block-CG, Block-GMRES (multiple right-hand sides)
- **Eigenvalue solvers**: Lanczos, Arnoldi, IRAM (Implicitly Restarted Arnoldi)
- **SVD solvers**: Truncated SVD, Randomized SVD
- **Factorizations**: Sparse LU, Sparse QR, Sparse Cholesky
- **Preconditioners**: Jacobi, Gauss-Seidel, SOR, ILU0, ILUT, ILUTP, IC0, ICT, AMG, SPAI, AINV, Schwarz
- **Ordering algorithms**: RCM, AMD, Nested Dissection

**Matrix Utilities (`oxiblas-matrix`)**
- Flexible matrix types with multiple layouts (row-major, column-major)
- View and mutable view support
- Efficient memory management and slicing

**Extended Features**
- **Tensor operations**: Einstein summation (24 patterns), batched operations
- **Advanced summation**: Kahan (compensated), pairwise, superaccurate
- ~~**C FFI**: Drop-in replacement for C BLAS/LAPACK (`oxiblas-ffi`)~~ (RETIRED v0.2.0)
- **ndarray integration**: Seamless interop with rust-ndarray (`oxiblas-ndarray`)

**Benchmarks (`oxiblas-benchmarks`)**
- **12 comprehensive benchmark suites** (13 files, 121 benchmark functions)
- Criterion.rs framework with HTML reports and statistical analysis
- OpenBLAS comparison suite (optional feature)
- Complete coverage: BLAS L1/L2/L3, LAPACK, sparse operations, iterative/eigenvalue/SVD solvers
- Performance: DGEMM achieves 79% of OpenBLAS speed on large matrices

#### Performance

**macOS (Apple M3, ARM64 NEON):**
- **GEMM**: 79% of OpenBLAS performance (1024×1024 matrices)
- **Peak**: 20.9 GFLOPS on large matrices

**Linux x86_64 (Intel Xeon E5-2623 v4, AVX2/FMA):**
- **DGEMM f64**: 80-112% of OpenBLAS performance across all sizes
  - 1024×1024: **102% of OpenBLAS** (213 vs 208 GFLOPS) - **Faster than OpenBLAS!**
  - 256×256: 95% of OpenBLAS (220 vs 232 GFLOPS) - Peak performance
  - 512×512: 80% of OpenBLAS (194 vs 244 GFLOPS) - Optimization target
- **SGEMM f32**: 94-112% of OpenBLAS performance
  - 64×64: **112% of OpenBLAS** (253 vs 225 GFLOPS) - **Faster than OpenBLAS!**
  - 128×128: 100% of OpenBLAS (328 vs 328 GFLOPS) - Identical performance
- **Cache-aware tuning**: 13-20% performance improvement with platform-specific optimization
  - Fine-tuned blocking parameters: KC=192, MC=128 (optimized for 256KB L2 cache)
  - Increased prefetch distance: 12 iterations (Intel Xeon E5-2600 latency)

**General:**
- **GEMM-based operations**: 6-15× speedup over naive implementations
- **Cache-aware**: BLIS-style blocking for optimal memory hierarchy usage
- **Platform detection**: Linux sysfs, macOS sysctl, x86_64 CPUID fallback
- **Parallel**: Efficient multi-threading with rayon
- **SIMD**: Hand-tuned kernels for AVX2/FMA and ARM NEON

#### Documentation

- Comprehensive API documentation with examples
- README with quick start guide
- TODO.md tracking development priorities
- 121 benchmark functions demonstrating all operations

#### Code Quality

- ~156,000 lines of Rust code
- Zero clippy warnings policy enforced
- Modular architecture with 9 specialized crates
- Complete test coverage for all operations
- Safe Rust with minimal unsafe blocks (only in SIMD kernels)

#### Known Limitations

- SVD performance not yet optimized (future work)
- Some LAPACK operations may be slower than commercial libraries
- No GPU acceleration (CPU-only)
- Requires Rust 1.85+ for edition 2024 features

#### Dependencies

**Required**:
- num-complex 0.4
- num-traits 0.2
- bytemuck 1.21

**Optional**:
- rayon 1.10 (parallel feature)
- ndarray 0.17 (ndarray integration)
- nalgebra 0.33 (interop)
- half 2.4 (f16 support)
- twofloat 0.8 (f128 support)

#### Crate Structure

- `oxiblas` - Main convenience crate, re-exports all modules
- `oxiblas-core` - Core types, traits, SIMD layer
- `oxiblas-matrix` - Matrix types and utilities
- `oxiblas-blas` - BLAS Level 1/2/3 implementations
- `oxiblas-lapack` - LAPACK factorizations and solvers
- `oxiblas-sparse` - Sparse matrix formats and algorithms
- ~~`oxiblas-ffi` - C FFI bindings (CBLAS/LAPACKE compatible)~~ (RETIRED v0.2.0)
- `oxiblas-ndarray` - Integration with rust-ndarray
- `oxiblas-benchmarks` - Comprehensive performance benchmarks

---

## Release Checklist (v0.1.0 — historical)

This checklist tracked readiness for the original v0.1.0 release (2025-12-27) and is
complete; it is retained here for historical reference only and requires no further
action. See the dated entries above for what shipped in each subsequent release.

- [x] All tests pass
- [x] Zero clippy warnings
- [x] Documentation complete
- [x] Benchmarks comprehensive
- [x] LICENSE file (Apache-2.0)
- [x] README up to date
- [x] CHANGELOG created
- [x] Version 0.1.0 in all Cargo.toml
- [x] Examples functional
- [x] cargo publish --dry-run succeeds
- [x] Git tags created

---

[Unreleased]: https://github.com/cool-japan/oxiblas/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/cool-japan/oxiblas/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/cool-japan/oxiblas/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/cool-japan/oxiblas/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/cool-japan/oxiblas/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/cool-japan/oxiblas/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/cool-japan/oxiblas/releases/tag/v0.1.0
