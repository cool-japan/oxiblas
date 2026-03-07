//! Size variation benchmarks for GEMM: covering tiny, non-power-of-2,
//! extremely rectangular, large, and f32 vs f64 comparison cases.
//!
//! Throughput is reported using `Throughput::Elements` set to the number of
//! multiply-accumulate operations (2 * m * n * k for GEMM), which Criterion
//! will display as elements/second — a direct FLOPS proxy.

use criterion::{
    BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group, criterion_main,
};
use oxiblas_blas::level3::gemm;
use oxiblas_matrix::Mat;
use std::hint::black_box;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a square f64 matrix of side `n` filled with a deterministic pattern.
fn square_f64(n: usize) -> Mat<f64> {
    let data: Vec<f64> = (0..n * n).map(|i| ((i % 97) as f64 + 1.0) * 0.01).collect();
    Mat::from_slice(n, n, &data)
}

/// Build a rectangular f64 matrix (rows x cols) with a deterministic pattern.
fn rect_f64(rows: usize, cols: usize) -> Mat<f64> {
    let data: Vec<f64> = (0..rows * cols)
        .map(|i| ((i % 97) as f64 + 1.0) * 0.01)
        .collect();
    Mat::from_slice(rows, cols, &data)
}

/// Build a square f32 matrix of side `n` with a deterministic pattern.
fn square_f32(n: usize) -> Mat<f32> {
    let data: Vec<f32> = (0..n * n).map(|i| ((i % 97) as f32 + 1.0) * 0.01).collect();
    Mat::from_slice(n, n, &data)
}

// ---------------------------------------------------------------------------
// Part 1: Very small matrices (overhead analysis)
// ---------------------------------------------------------------------------

fn bench_tiny_gemm_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_tiny_f64");

    // Sizes: 2, 4, 8 — power-of-2 micro-kernels, useful for overhead analysis.
    for &n in &[2usize, 4, 8] {
        // 2*n*n*n multiply-add FLOPs per GEMM.
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
// Part 2: Non-power-of-2 sizes
// ---------------------------------------------------------------------------

fn bench_nonpow2_gemm_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_nonpow2_f64");

    // 100, 300, 777 — exercising code paths that cannot rely on perfect tile alignment.
    for &n in &[100usize, 300, 777] {
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
// Part 3: Extremely rectangular matrices
// ---------------------------------------------------------------------------

/// Tall-thin: A is (m x k), B is (k x n) where m >> k (10000 x 10) * (10 x 10000).
fn bench_tall_thin_gemm_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_tall_thin_f64");

    // (10000 x 10) * (10 x 10000) => output is (10000 x 10000)
    // FLOPs = 2 * 10000 * 10000 * 10 = 2e9 — use a small sample size.
    let m = 10_000usize;
    let k = 10usize;
    let n = 10_000usize;

    group.throughput(Throughput::Elements((2 * m * k * n) as u64));
    group.sample_size(10);

    let a = rect_f64(m, k);
    let b = rect_f64(k, n);
    let mut out: Mat<f64> = Mat::zeros(m, n);

    group.bench_function("10000x10_x_10x10000", |bench| {
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

    group.finish();
}

/// Short-wide: A is (m x k), B is (k x n) where k >> m (10 x 10000) * (10000 x 10).
fn bench_short_wide_gemm_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_short_wide_f64");

    // (10 x 10000) * (10000 x 10) => output is (10 x 10)
    // FLOPs = 2 * 10 * 10 * 10000 = 2e6 — fast enough for default sample size.
    let m = 10usize;
    let k = 10_000usize;
    let n = 10usize;

    group.throughput(Throughput::Elements((2 * m * k * n) as u64));

    let a = rect_f64(m, k);
    let b = rect_f64(k, n);
    let mut out: Mat<f64> = Mat::zeros(m, n);

    group.bench_function("10x10000_x_10000x10", |bench| {
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

    group.finish();
}

// ---------------------------------------------------------------------------
// Part 4: Large matrices
// ---------------------------------------------------------------------------

fn bench_large_gemm_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_large_f64");

    // 2048 x 2048: ~17 billion FLOPs — use a very small sample size.
    let n = 2048usize;
    group.throughput(Throughput::Elements((2 * n * n * n) as u64));
    group.sample_size(10);
    group.sampling_mode(SamplingMode::Flat);

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

    group.finish();
}

// ---------------------------------------------------------------------------
// Part 5: f32 vs f64 comparison at multiple sizes
// ---------------------------------------------------------------------------

/// Compare f32 and f64 GEMM throughput at sizes 128, 512, 1024.
fn bench_f32_vs_f64_gemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_f32_vs_f64");

    for &n in &[128usize, 512, 1024] {
        let flops = (2 * n * n * n) as u64;

        // --- f64 ---
        group.throughput(Throughput::Elements(flops));
        {
            let a = square_f64(n);
            let b = square_f64(n);
            let mut out: Mat<f64> = Mat::zeros(n, n);

            group.bench_with_input(BenchmarkId::new("f64", n), &n, |bench, _| {
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

        // --- f32 ---
        group.throughput(Throughput::Elements(flops));
        {
            let a = square_f32(n);
            let b = square_f32(n);
            let mut out: Mat<f32> = Mat::zeros(n, n);

            group.bench_with_input(BenchmarkId::new("f32", n), &n, |bench, _| {
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
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion wiring
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_tiny_gemm_f64,
    bench_nonpow2_gemm_f64,
    bench_tall_thin_gemm_f64,
    bench_short_wide_gemm_f64,
    bench_large_gemm_f64,
    bench_f32_vs_f64_gemm,
);
criterion_main!(benches);
