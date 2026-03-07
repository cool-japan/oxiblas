//! Precision benchmarks comparing f32, f64 GEMM at multiple sizes,
//! f16 (half-precision) storage throughput, and mixed-precision LU
//! (f32 factorization with f64 residual correction).
//!
//! Throughput is reported via `Throughput::Elements` set to the number of
//! multiply-add operations (2 * m * n * k for square GEMM), giving a direct
//! FLOPS proxy in Criterion's HTML reports.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use oxiblas_blas::level3::gemm;
use oxiblas_lapack::solve::mixed_precision_solve_lu;
use oxiblas_matrix::Mat;
use std::hint::black_box;

// ---------------------------------------------------------------------------
// Helper constructors
// ---------------------------------------------------------------------------

fn square_f64(n: usize) -> Mat<f64> {
    let data: Vec<f64> = (0..n * n).map(|i| ((i % 97) as f64 + 1.0) * 0.01).collect();
    Mat::from_slice(n, n, &data)
}

fn square_f32(n: usize) -> Mat<f32> {
    let data: Vec<f32> = (0..n * n).map(|i| ((i % 97) as f32 + 1.0) * 0.01).collect();
    Mat::from_slice(n, n, &data)
}

/// Build a diagonally dominant n x n f64 matrix (guaranteed non-singular for LU).
fn diag_dominant_f64(n: usize) -> Mat<f64> {
    let mut a: Mat<f64> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = if i == j {
                (n as f64) + 1.0
            } else {
                ((i * 17 + j * 31) % 100) as f64 * 0.01
            };
        }
    }
    a
}

// ---------------------------------------------------------------------------
// Part 1: f32 vs f64 GEMM at 128, 256, 512, 1024
// ---------------------------------------------------------------------------

fn bench_gemm_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("precision_gemm_f32");

    for &n in &[128usize, 256, 512, 1024] {
        // 2*n^3 multiply-add operations.
        group.throughput(Throughput::Elements((2 * n * n * n) as u64));

        let a = square_f32(n);
        let b = square_f32(n);
        let mut out: Mat<f32> = Mat::zeros(n, n);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                gemm(
                    black_box(1.0_f32),
                    black_box(a.as_ref()),
                    black_box(b.as_ref()),
                    black_box(0.0_f32),
                    black_box(out.as_mut()),
                );
            });
        });
    }

    group.finish();
}

fn bench_gemm_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("precision_gemm_f64");

    for &n in &[128usize, 256, 512, 1024] {
        group.throughput(Throughput::Elements((2 * n * n * n) as u64));

        let a = square_f64(n);
        let b = square_f64(n);
        let mut out: Mat<f64> = Mat::zeros(n, n);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                gemm(
                    black_box(1.0_f64),
                    black_box(a.as_ref()),
                    black_box(b.as_ref()),
                    black_box(0.0_f64),
                    black_box(out.as_mut()),
                );
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Part 2: f16 precision benchmark
//
// The `gemm` kernel does not yet implement GemmKernel for `half::f16` because
// f16 SIMD accumulation requires explicit widening to f32.  The benchmark here
// measures the practical cost of the f16 storage round-trip that precedes a
// mixed-precision kernel: convert f16 → f32 slabs, run f32 GEMM, convert the
// f32 result back to f16.  This reflects real workloads (e.g. inference
// pipelines) and remains 100 % safe Rust using only the `half` crate already
// pulled in transitively by oxiblas-core's "f16" feature.
// ---------------------------------------------------------------------------

fn bench_f16_storage_gemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("precision_gemm_f16_storage");

    for &n in &[128usize, 256, 512] {
        // FLOPs are still counted against the core 2*n^3 multiply-adds.
        group.throughput(Throughput::Elements((2 * n * n * n) as u64));

        // Build f16-encoded matrices.
        let a_f16: Vec<half::f16> = (0..n * n)
            .map(|i| half::f16::from_f32(((i % 97) as f32 + 1.0) * 0.01))
            .collect();
        let b_f16: Vec<half::f16> = (0..n * n)
            .map(|i| half::f16::from_f32(((i % 89) as f32 + 1.0) * 0.01))
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                // Step 1: widen f16 → f32.
                let a_f32: Vec<f32> = black_box(a_f16.iter().map(|x| x.to_f32()).collect());
                let b_f32: Vec<f32> = black_box(b_f16.iter().map(|x| x.to_f32()).collect());

                let a_mat = Mat::from_slice(n, n, &a_f32);
                let b_mat = Mat::from_slice(n, n, &b_f32);
                let mut out_f32: Mat<f32> = Mat::zeros(n, n);

                // Step 2: f32 GEMM.
                gemm(
                    black_box(1.0_f32),
                    black_box(a_mat.as_ref()),
                    black_box(b_mat.as_ref()),
                    black_box(0.0_f32),
                    black_box(out_f32.as_mut()),
                );

                // Step 3: narrow f32 → f16 result.
                let _out_f16: Vec<half::f16> = black_box(
                    out_f32
                        .raw_data()
                        .iter()
                        .map(|&x| half::f16::from_f32(x))
                        .collect(),
                );
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Part 3: Mixed-precision LU (f32 factor + f64 residual)
// ---------------------------------------------------------------------------

fn bench_mixed_precision_lu(c: &mut Criterion) {
    let mut group = c.benchmark_group("precision_mixed_lu");

    // Sizes chosen so the benchmark runs in reasonable wall time.
    // LU factorization is O(2/3 * n^3); mixed-precision adds iterative refinement.
    for &n in &[64usize, 128, 256] {
        // Report throughput relative to the dominant LU factorization cost.
        group.throughput(Throughput::Elements((2 * n * n * n / 3) as u64));

        let a = diag_dominant_f64(n);

        // RHS: single column for simplicity.
        let b_data: Vec<f64> = (0..n).map(|i| (i % 10) as f64 + 1.0).collect();
        let b = Mat::from_slice(n, 1, &b_data);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                // mixed_precision_solve_lu: factorizes in f32, refines in f64.
                let _ = black_box(mixed_precision_solve_lu(
                    black_box(a.as_ref()),
                    black_box(b.as_ref()),
                ));
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion wiring
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_gemm_f32,
    bench_gemm_f64,
    bench_f16_storage_gemm,
    bench_mixed_precision_lu,
);
criterion_main!(benches);
