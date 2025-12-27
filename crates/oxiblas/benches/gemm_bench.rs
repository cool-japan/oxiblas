//! GEMM Benchmarks
//!
//! Benchmarks for general matrix-matrix multiplication at various sizes.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use oxiblas::blas::level3::gemm;
use oxiblas::matrix::Mat;
use std::hint::black_box;

fn bench_gemm_square(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_square");

    for size in [32, 64, 128, 256, 512].iter() {
        let n = *size;
        let a: Mat<f64> = Mat::filled(n, n, 1.0);
        let b: Mat<f64> = Mat::filled(n, n, 1.0);
        let mut c_mat: Mat<f64> = Mat::zeros(n, n);

        group.bench_with_input(BenchmarkId::new("f64", n), &n, |bench, _| {
            bench.iter(|| {
                gemm(
                    black_box(1.0),
                    black_box(a.as_ref()),
                    black_box(b.as_ref()),
                    black_box(0.0),
                    black_box(c_mat.as_mut()),
                );
            });
        });
    }

    group.finish();
}

fn bench_gemm_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_f32");

    for size in [64, 128, 256, 512].iter() {
        let n = *size;
        let a: Mat<f32> = Mat::filled(n, n, 1.0);
        let b: Mat<f32> = Mat::filled(n, n, 1.0);
        let mut c_mat: Mat<f32> = Mat::zeros(n, n);

        group.bench_with_input(BenchmarkId::new("f32", n), &n, |bench, _| {
            bench.iter(|| {
                gemm(
                    black_box(1.0),
                    black_box(a.as_ref()),
                    black_box(b.as_ref()),
                    black_box(0.0),
                    black_box(c_mat.as_mut()),
                );
            });
        });
    }

    group.finish();
}

fn bench_gemm_rectangular(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_rectangular");

    // Tall-skinny: (512 x 64) * (64 x 512)
    {
        let a: Mat<f64> = Mat::filled(512, 64, 1.0);
        let b: Mat<f64> = Mat::filled(64, 512, 1.0);
        let mut c_mat: Mat<f64> = Mat::zeros(512, 512);

        group.bench_function("512x64_x_64x512", |bench| {
            bench.iter(|| {
                gemm(
                    black_box(1.0),
                    black_box(a.as_ref()),
                    black_box(b.as_ref()),
                    black_box(0.0),
                    black_box(c_mat.as_mut()),
                );
            });
        });
    }

    // Wide-short: (64 x 512) * (512 x 64)
    {
        let a: Mat<f64> = Mat::filled(64, 512, 1.0);
        let b: Mat<f64> = Mat::filled(512, 64, 1.0);
        let mut c_mat: Mat<f64> = Mat::zeros(64, 64);

        group.bench_function("64x512_x_512x64", |bench| {
            bench.iter(|| {
                gemm(
                    black_box(1.0),
                    black_box(a.as_ref()),
                    black_box(b.as_ref()),
                    black_box(0.0),
                    black_box(c_mat.as_mut()),
                );
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_gemm_square,
    bench_gemm_f32,
    bench_gemm_rectangular
);
criterion_main!(benches);
