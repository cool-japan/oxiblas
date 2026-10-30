# oxiblas-blas TODO

BLAS (Basic Linear Algebra Subprograms) implementations.

## Level 1 - Vector Operations

### Current Status
- [x] dot, dotc (dot product)
- [x] axpy (y = αx + y)
- [x] scal (x = αx)
- [x] copy, swap
- [x] nrm2, asum (norms)
- [x] iamax, iamin (index of max/min)
- [x] rot, rotg, rotm, rotmg (rotations)

### Missing
- [x] sdsdot - scaled dot product with extended precision accumulation
- [x] dsdot - dot product with extended precision
- [x] Complex variants optimization (cdotc, zdotc with conjugate) - SIMD optimized with NEON/AVX2, 4-way accumulation
- [x] Complex Level-1 natives completing the Netlib set (0.2.3, #15) - `cabs1`/`scabs1`/`dcabs1`, `cnrm2`/`scnrm2`/`dznrm2` (+ `cnrm2_strided`), `scal_real`/`csscal`/`zdscal`, `rotm_strided`, `zdrot_strided`
- [x] Netlib 3.12 increment and quick-return semantics (0.2.3, #10, #11, #12) - `inc == 0` and negative increments in nrm2/dot/axpy/copy/swap, complex scal/axpy quick returns, `cabs1` ranking in `iamax_strided`/`iamin_strided`, `crotg` per `zrotg.f90`; native `rotg`/`rotmg` aligned with `drotg.f90`/`drotmg.f` (#14)

### Optimizations
- [x] SIMD optimization for axpy (NEON/AVX2 with FMA, 8-16x unrolling)
- [x] SIMD optimization for nrm2 (Blue's algorithm for overflow/underflow protection)
- [x] Blocked dot product for large vectors (block size 1024/2048)
- [x] Parallel variants for large vectors (axpy_par, dot_par, nrm2_par, asum_par, scal_par with Rayon)

---

## Level 2 - Matrix-Vector Operations

### Current Status
- [x] gemv (general matrix-vector)
- [x] ger, gerc (rank-1 update)
- [x] trmv (triangular matrix-vector)
- [x] trsv (triangular solve)
- [x] syr, her (symmetric/Hermitian rank-1)
- [x] symv - Symmetric matrix-vector multiply
- [x] hemv - Hermitian matrix-vector multiply
- [x] syr2 - Symmetric rank-2 update
- [x] her2 - Hermitian rank-2 update
- [x] Reference quick returns, zero-column skips and Hermitian diagonal (0.2.3, #13, #14) - gemv/gemv_with_par/gemv_strided leave y untouched for an empty A; her/her2 write a real diagonal; ger/gerc/syr/spr/syr2/spr2/her/her2/hpr/hpr2 skip columns with an exactly-zero scaling element (ger/gerc column-oriented, reference association); hpr/hpr2 are bitwise equal to her/her2 for NaN-free input; her2/hpr2 form the second temp as conj(alpha*x(j)), as ZHER2/ZHPR2 do

### Banded Operations (Completed)
- [x] gbmv - General banded matrix-vector
- [x] sbmv - Symmetric banded matrix-vector
- [x] hbmv - Hermitian banded matrix-vector
- [x] tbmv - Triangular banded matrix-vector
- [x] tbsv - Triangular banded solve

### Packed Operations (Completed)
- [x] spmv - Symmetric packed matrix-vector
- [x] hpmv - Hermitian packed matrix-vector
- [x] tpmv - Triangular packed matrix-vector
- [x] tpsv - Triangular packed solve
- [x] spr - Symmetric packed rank-1
- [x] hpr - Hermitian packed rank-1
- [x] spr2 - Symmetric packed rank-2
- [x] hpr2 - Hermitian packed rank-2

### Optimizations
- [x] Blocked gemv for better cache utilization (MC=64, KC=256, 8-way unrolling)
- [x] Parallel gemv for large matrices - 8-way unrolling for NoTrans, 4-way for Trans/ConjTrans, tree-based parallel reduction, parallel beta scaling
- [x] SIMD-optimized trsv - blocked algorithm with 4-way unrolling (TRSV_BLOCK_SIZE=64)
- [x] Fused operations (gemv + add) - gemv_add, gemv_add_inplace, gemv_sum2 with blocked variants

---

## Level 3 - Matrix-Matrix Operations

### Current Status
- [x] gemm (general matrix-matrix) - AVX2/NEON optimized
- [x] trsm (triangular solve)
- [x] trmm (triangular multiply)
- [x] symm (symmetric matrix-matrix) - GEMM-optimized for n >= 32
- [x] hemm (Hermitian matrix-matrix) - 3M GEMM optimization available
- [x] syrk (symmetric rank-k)
- [x] herk (Hermitian rank-k)
- [x] syr2k (symmetric rank-2k)
- [x] her2k (Hermitian rank-2k)
- [x] Reference alpha == 0 / beta == 0 semantics (0.2.3, #9) - A/B are never read for alpha == 0 and C never for beta == 0 in gemm (small, blocked, parallel), gemm3m, Strassen, Winograd, cache-oblivious, batched, symm, hemm, syrk, syr2k, herk, her2k and trsm; herk/her2k quick return

### Optimizations
- [x] Larger micro-kernels for AVX-512 - f64: 16×6 (12 zmm accumulators), f32: 16×16 (16 zmm accumulators), with 2-way loop unrolling and FMA
- [x] Larger micro-kernels for AVX2 - f64: 8×6 (12 ymm accumulators) with 2-way unrolling, improved arithmetic intensity
- [x] NEON optimized micro-kernels for Apple Silicon - f64: 8×6 (24 accumulators) with 2-way unrolling, f32: 8×8 (16 accumulators) with 4-way unrolling, software prefetching
- [x] Better packing strategies - gemm_packing module with 4-way unrolling, prefetching, cache-line aware
- [x] Auto-tuning for block sizes - autotune module with runtime cache detection, gemm_auto/gemm_auto_with_par, adaptive blocking for aspect ratios
- [x] Asymmetric blocking for rectangular matrices - GemmBlocking::asymmetric with tall-thin, short-wide, inner-product, panel-panel strategies
- [x] Small matrix specializations (< 32) - unrolled kernels for 2x2, 3x3, 4x4, register blocking for 8x8, 16x16
- [x] Strassen algorithm for very large matrices - recursive algorithm with 7 products, O(n^2.807) complexity, parallel variant
- [x] Optimized blocking parameters for Apple Silicon L2 - f64: MC=512, KC=256 (1 MB pack_a fits in 4+ MB L2), f32: MC=512, KC=512
- [x] Macro-panel multiply prefetching - prefetch next A micro-panel while computing current, 128-byte cache line aware for Apple Silicon
- [x] Unified optimized packing in GEMM - using pack_a_optimized/pack_b_optimized with 4-way unrolling and software prefetching
- [x] TRMM-specific blocking parameters - smaller MC=256, KC=128 for f64 to reduce overhead in triangular operations
- [x] TRSM-specific blocking parameters - smaller MC=256, KC=128 for f64 to reduce overhead in triangular solve operations
- [x] SYMM via GEMM optimization - expands symmetric matrix to full dense and uses optimized GEMM for n >= 32
- [x] Balanced GEMM blocking (0.2.3, #4) - `balanced_block` fits every MC/KC/NC target to the problem: a dimension that fits is one block, a larger one is split into the fewest near-equal MR/NR/8-aligned blocks (no `[252, 4]`-style tails that re-pack A)
- [x] 2D (M×N) parallel GEMM driver (0.2.3, #16) - `level3/gemm_parallel.rs` (feature `parallel`), tiles aligned to the micro-tile grid, bitwise equal to `Par::Seq`; gate 128³ total / 80³ per thread
- [x] NEON small-GEMM dispatch on aarch64 (0.2.3, #16) - `gemm_small` sends f32/f64 problems with m, n >= 4 and column-major operands to the 4×4 NEON kernels
- [x] Blocked GEMM honours `col_stride` of NoTrans operands (0.2.3) - strided views such as `MatRef::transpose()` were read with a unit column stride by the fast packers

### Complex Optimizations
- [x] Optimized complex gemm (3M method) - uses real GEMM kernels for large matrices
- [x] Complex trsm optimization - trsm_c64/trsm_c32 using 3M GEMM for off-diagonal updates
- [x] Interleaved complex storage support - complex_interleaved module with conversions, dotc, axpy, scal, nrm2 for interleaved format

### Recently Added Variants
- [x] gemm3m - Complex gemm using 3 real multiplies (gemm3m_c32, gemm3m_c64)
- [x] gemmt - Triangular result update (upper/lower, with transpose support)

---

## Tensor Operations

### Current Status
- [x] Tensor3 - 3D tensor (rank-3 array) in row-major order
- [x] contract_2d - 2D tensor contraction (matrix multiply)
- [x] contract_3d_2d - 3D tensor with 2D matrix contraction
- [x] outer_product - Vector outer product
- [x] batched_matmul - Batched matrix multiplication
- [x] einsum - Einstein summation notation for tensor contractions

### Optimizations
- [x] contract_2d GEMM optimization - uses optimized GEMM for matrices >= 32, row-major to column-major transpose trick
- [x] batched_matmul GEMM optimization - uses optimized GEMM for matrices >= 16, row-major to column-major transpose trick
- [x] einsum GEMM optimization - matrix multiply patterns use optimized contract_2d internally

---

## Performance Optimizations (2025-12)

### Phase 1 - Core Optimizations
- [x] HEMM 3M GEMM optimization - enabled for n >= 64, uses 3 real GEMMs instead of 4
- [x] GEMV asymmetric blocking - increased MC=128, KC=512 for better cache utilization
- [x] DTRSM recursive blocking - divide-and-conquer for matrices >= 256, 2x better cache locality
- [x] AVX2 micro-kernel tuning - 4-way loop unrolling and software prefetching
- [x] Auto-tuning integration - automatic blocking parameter selection for matrices >= 512

### Phase 2 - Advanced Optimizations
- [x] SIMD GEMV inner loop - explicit NEON/AVX2 vectorization with FMA for f64
- [x] Parallel recursive DTRSM - Par::Rayon for GEMM updates when m,n >= 128
- [x] Larger GEMM micro-kernel (8x6) - AVX2: 12 ymm accumulators, NEON: 24 registers with 2-way unrolling
- [x] Packing buffer reuse - main GEMM buffers already reused across iterations
- [x] f32 AVX2 micro-kernel tuning - 4-way loop unrolling with software prefetching

---

## Performance Targets

| Operation | Size | Current | Target | Notes |
|-----------|------|---------|--------|-------|
| dgemm | 1000x1000 | ~80% MKL | 95% MKL | Micro-kernel tuning |
| sgemm | 1000x1000 | ~75% MKL | 90% MKL | FMA utilization |
| dgemv | 10000x10000 | ~75% MKL | 85% MKL | Improved blocking (MC=128, KC=512) |
| dtrsm | 1000x1000 | ~75% MKL | 85% MKL | Recursive blocking for >= 256 |

---

## CBLAS Interface

- [x] Level 1 CBLAS functions (cblas_ddot, cblas_sdot, cblas_dnrm2, cblas_snrm2, cblas_dasum, cblas_sasum, cblas_idamax, cblas_isamax, cblas_dscal, cblas_sscal, cblas_daxpy, cblas_saxpy, cblas_dcopy, cblas_scopy, cblas_dswap, cblas_sswap)
- [x] Level 2 CBLAS functions (cblas_dgemv, cblas_sgemv)
- [x] Level 3 CBLAS functions (cblas_dgemm, cblas_sgemm, cblas_zgemm, cblas_cgemm, cblas_dtrsm, cblas_strsm, cblas_dtrmm, cblas_strmm, cblas_dsyrk, cblas_ssyrk, cblas_dsyr2k, cblas_ssyr2k, cblas_dsymm, cblas_ssymm)
- [x] Complex CBLAS dot products (cblas_zdotu_sub, cblas_zdotc_sub, cblas_cdotu_sub, cblas_cdotc_sub)
- [x] Reference CBLAS entry-point coverage (0.2.3, #14) - 78 entry points added (16 Level 1, 48 Level 2, 14 Level 3); 152 `cblas_*` symbols exported: every routine declared in reference `cblas.h` 3.12.1 (3.12.0 plus `?gemmtr`) except the `cblas_xerbla` handler, with matching parameter counts and kinds. Not a full drop-in yet: `i?amax` return `i32` (header: `size_t`), the layout/trans/uplo/side/diag parameters are Rust enums (out-of-range values are undefined behaviour), there is no `_64` API, invalid arguments are a silent no-op instead of `cblas_xerbla`, and no C-linkable library is built by this workspace. Open items: root TODO.md, "Release Milestones" (v0.2.4 triangular kernels, v0.3.0 signatures)
  - Level 1: `cblas_{s,d}rotg`, `rotmg`, `rot`, `rotm`; `cblas_{c,z}rotg`; `cblas_csrot`, `cblas_zdrot`; `cblas_sdsdot`, `cblas_dsdot`; `cblas_scabs1`, `cblas_dcabs1`
  - Level 2: band (`?gbmv`, `?sbmv`, `?hbmv`, `?tbmv`, `?tbsv`), packed (`?spmv`, `?spr`, `?spr2`, `?hpmv`, `?hpr`, `?hpr2`, `?tpmv`, `?tpsv`) and complex dense (`?geru`, `?gerc`, `?her`, `?her2`, `?trmv`, `?trsv`)
  - Level 3: `?gemmtr` (all four precisions), `cblas_{c,z}symm`, `cblas_{c,z}syrk`, `cblas_{c,z}syr2k`, `cblas_{c,z}trmm`, `cblas_{c,z}trsm`
- [x] Fast path optimization for unit stride vectors (uses SIMD-optimized level1 functions)
- [x] Fast path optimization for NoTrans/NoTrans GEMM (uses SIMD-optimized level3::gemm)
- [x] Row-major and column-major layout support with proper transpose handling

---

## Testing Requirements

- [x] BLAS-TESTER compatibility - cblas module with extern "C" functions for Level 1/2/3, complex ops, row/col major support - 0.2.3: covers the entry points only; netlib's error-exit tests pass out-of-range enum values and expect `cblas_xerbla`, which does not exist here (invalid arguments are a silent no-op, enum values outside the `cblas.h` set are undefined behaviour; see the v0.3.0 milestone in the root TODO.md)
- [x] Accuracy tests (forward/backward error) - accuracy module with error metrics, reference implementations, DOT/GEMV/GEMM/NRM2 accuracy tests
- [x] Edge cases (1x1, empty, singular)
- [x] Stride handling tests
- [x] Memory bounds tests - comprehensive tests for empty/single/odd/prime/large vectors
- [x] Overflow/underflow tests - Blue's algorithm for nrm2, comprehensive tests for extreme values

---

## Benchmarks

- [x] Criterion benchmarks for all operations - Level 1/2/3, new features, advanced features (Strassen, asymmetric, parallel)
- [x] Comparison with OpenBLAS - comparison.rs with DGEMM, SGEMM, DTRSM, DSYRK, DGEMV, DDOT, DAXPY, DNRM2
- [ ] Comparison with MKL (where available) - optional, requires Intel MKL
- [x] Scaling tests (1 thread vs N threads) - bench_gemm_scaling, parallel vs sequential comparisons

---

## Code Organization

- [x] TRSM refactored into 5 modules (level3/trsm/)
- [x] CBLAS refactored into 3 modules (cblas/) - 0.2.3: since split further (the `basic/` directory, plus `level1_rot`, `level3_ext`, `banded`, `packed`, `complex_level2_ext` and `strided` from #14); see the module list in `cblas/mod.rs`

## Future Enhancements

- [x] Winograd algorithm for GEMM (reduces multiplications) - implemented in level3
- [x] Cache-oblivious GEMM algorithm option - recursive divide-and-conquer
- [x] BLAS-like extensions (batched operations with variable strides) - `gemm_batched`, `gemm_strided_batched`, `axpy_batched`, `gemv_batched` + parallel (v0.2.0)

---

## Version 0.2.1 (2026-03-16)

- **780 lib tests + 68 doctests passing, 0 stubs**
- All BLAS Level 1/2/3, tensor, CBLAS, and extended operations fully implemented and tested
