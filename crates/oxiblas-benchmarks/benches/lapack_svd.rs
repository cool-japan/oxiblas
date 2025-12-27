//! Benchmarks for LAPACK SVD operations.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use oxiblas_lapack::svd::{Svd, SvdDc};
use oxiblas_matrix::Mat;
use std::hint::black_box;

fn bench_svd_standard(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd_standard");

    for size in [50, 100, 200].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        let a = Mat::from_slice(
            n,
            n,
            &(0..n * n).map(|i| i as f64 + 1.0).collect::<Vec<_>>(),
        );

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = Svd::compute(black_box(a.as_ref())).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_svd_divide_conquer(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd_divide_conquer");

    for size in [50, 100, 200].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        let a = Mat::from_slice(
            n,
            n,
            &(0..n * n).map(|i| i as f64 + 1.0).collect::<Vec<_>>(),
        );

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = SvdDc::compute(black_box(a.as_ref())).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_svd_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd_algorithm_comparison");

    let n = 100;
    let a = Mat::from_slice(
        n,
        n,
        &(0..n * n).map(|i| i as f64 + 1.0).collect::<Vec<_>>(),
    );

    group.bench_function("standard_gesvd", |b| {
        b.iter(|| {
            let _ = Svd::compute(black_box(a.as_ref())).unwrap();
        });
    });

    group.bench_function("divide_conquer_gesdd", |b| {
        b.iter(|| {
            let _ = SvdDc::compute(black_box(a.as_ref())).unwrap();
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_svd_standard,
    bench_svd_divide_conquer,
    bench_svd_comparison
);
criterion_main!(benches);
