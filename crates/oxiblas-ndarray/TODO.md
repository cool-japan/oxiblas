# oxiblas-ndarray TODO

Integration with the ndarray crate.

## Conversions

### Current Status
- [x] Basic Array2 support
- [x] Zero-copy views where possible (column-major)
- [x] ArrayView / ArrayViewMut support
- [x] Strided array support (row-major handled via transposed view)
- [x] mat_ref_to_array2 for MatRef conversion

### Completed
- [x] Dynamic dimension support (ArrayD) - arrayd_to_mat, mat_to_arrayd, arrayd_to_array2, array2_to_arrayd

---

## BLAS Operations

- [x] GEMM using oxiblas backend
- [x] GEMV using oxiblas backend
- [x] Dot product using oxiblas
- [x] Norms using oxiblas (nrm2, asum, frobenius, norm_1, norm_inf, norm_max)
- [x] AXPY operation
- [x] Scale operation
- [x] Matrix-vector multiplication (matvec, matvec_t)
- [x] Matrix multiplication (matmul)

### Complex Number Support

- [x] Complex dot products (dotc_c64_ndarray, dotc_c32_ndarray - conjugate)
- [x] Complex unconjugated dot products (dotu_c64_ndarray, dotu_c32_ndarray)
- [x] Complex vector norms (nrm2_c64_ndarray, nrm2_c32_ndarray)
- [x] Complex L1 norms (asum_c64_ndarray, asum_c32_ndarray)
- [x] Complex AXPY (axpy_c64_ndarray, axpy_c32_ndarray)
- [x] Complex scale (scal_c64_ndarray, scal_c32_ndarray)
- [x] Hermitian transpose (conj_transpose_c64, conj_transpose_c32)
- [x] Complex Frobenius norm (frobenius_norm_c64, frobenius_norm_c32)
- [x] Complex matrix norms (norm_1_c64, norm_inf_c64, norm_max_c64, etc.)
- [x] Complex trace (trace_c64, trace_c32)
- [x] Complex identity matrices (eye_c64, eye_c32)

---

## LAPACK Operations

- [x] LU decomposition returning ndarray
- [x] QR decomposition returning ndarray
- [x] SVD returning ndarray
- [x] Eigenvalue decomposition returning ndarray (symmetric)
- [x] Linear solve returning ndarray
- [x] Least squares solve
- [x] Cholesky decomposition
- [x] Matrix inverse
- [x] Pseudo-inverse
- [x] Determinant
- [x] Condition number
- [x] Rank computation

### Advanced LAPACK Operations

- [x] Randomized SVD (rsvd_ndarray, rsvd_power_ndarray)
- [x] Low-rank approximation (low_rank_approx_ndarray)
- [x] Schur decomposition (schur_ndarray)
- [x] General eigenvalue decomposition (eig_ndarray, eigvals_ndarray)
- [x] Tridiagonal solvers (tridiag_solve_ndarray, tridiag_solve_spd_ndarray)
- [x] Batch tridiagonal solve (tridiag_solve_multiple_ndarray)

---

## Performance

- [x] Benchmark against ndarray (benches/ndarray_blas.rs)
- [x] Benchmark against faer (benches/faer_comparison.rs)
- [x] Memory layout optimization (column-major preference)
- [x] In-place operations (gemm_ndarray, gemv_ndarray, axpy_ndarray)

### Benchmark Results (Apple Silicon M-series)

**Matrix Multiplication (GEMM) - oxiblas vs faer:**

| Size | oxiblas | faer | Ratio |
|------|---------|------|-------|
| 32x32 | ~15µs | ~1.7µs | faer 8.8x faster |
| 64x64 | ~26µs | ~12µs | faer 2.1x faster |
| 128x128 | ~204µs | ~72µs | faer 2.8x faster |
| 256x256 | ~1.2ms | ~350µs | faer 3.4x faster |
| 512x512 | ~9.2ms | ~2.1ms | faer 4.4x faster |

**QR Decomposition - oxiblas vs faer:**

| Size | oxiblas | faer | Ratio |
|------|---------|------|-------|
| 64x64 | ~270µs | ~84µs | faer 3.2x faster |
| 128x128 | ~2.4ms | ~690µs | faer 3.5x faster |
| 256x256 | ~13ms | ~2.4ms | faer 5.4x faster |

**Analysis:**
- faer uses the highly optimized `gemm` crate with CPU-specific tuning
- oxiblas is pure Rust with portable SIMD (NEON/AVX2)
- faer is recommended for maximum performance on supported platforms
- oxiblas provides consistent cross-platform performance without external dependencies
- Both libraries are pure Rust without system BLAS/LAPACK requirements

---

## Compatibility

- [x] ndarray 0.16.x support
- [x] Feature parity with basic ndarray-linalg operations

---

## Testing

- [x] Unit tests (~94 passing)
- [x] Doc tests
- [x] Conversion roundtrip tests
- [x] Numerical accuracy tests
- [x] Memory layout tests
- [x] Large array tests
- [x] All decomposition tests

## Future Enhancements

- [ ] ndarray 0.17 support when released
- [ ] Support for ndarray's parallel feature
- [ ] Sparse ndarray integration
