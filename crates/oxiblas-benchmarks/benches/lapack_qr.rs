//! Benchmarks for LAPACK QR factorizations.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use oxiblas_lapack::qr::{CompleteOrthogonalDecomp, Lq, Qr, QrPivot, Rq};
use oxiblas_matrix::Mat;
use std::hint::black_box;

fn bench_qr_standard(c: &mut Criterion) {
    let mut group = c.benchmark_group("qr_standard");

    for size in [50, 100, 200, 500].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        let a = Mat::from_slice(
            n,
            n,
            &(0..n * n).map(|i| i as f64 + 1.0).collect::<Vec<_>>(),
        );

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = Qr::compute(black_box(a.as_ref())).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_qr_with_pivot(c: &mut Criterion) {
    let mut group = c.benchmark_group("qr_with_pivot");

    for size in [50, 100, 200, 500].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        let a = Mat::from_slice(
            n,
            n,
            &(0..n * n).map(|i| i as f64 + 1.0).collect::<Vec<_>>(),
        );

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = QrPivot::compute(black_box(a.as_ref())).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_lq_factorization(c: &mut Criterion) {
    let mut group = c.benchmark_group("lq_factorization");

    for size in [50, 100, 200, 500].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        let a = Mat::from_slice(
            n,
            n,
            &(0..n * n).map(|i| i as f64 + 1.0).collect::<Vec<_>>(),
        );

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = Lq::compute(black_box(a.as_ref())).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_rq_factorization(c: &mut Criterion) {
    let mut group = c.benchmark_group("rq_factorization");

    for size in [50, 100, 200, 500].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        let a = Mat::from_slice(
            n,
            n,
            &(0..n * n).map(|i| i as f64 + 1.0).collect::<Vec<_>>(),
        );

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = Rq::compute(black_box(a.as_ref())).unwrap();
            });
        });
    }

    group.finish();
}

fn bench_complete_orthogonal(c: &mut Criterion) {
    let mut group = c.benchmark_group("complete_orthogonal");

    for size in [50, 100, 200].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        // Create rank-deficient matrix
        let mut a_data = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a_data[i * n + j] = if i == j {
                    (i + 1) as f64
                } else if i < j {
                    (i * n + j) as f64 * 0.1
                } else {
                    0.0
                };
            }
        }
        let a = Mat::from_slice(n, n, &a_data);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = CompleteOrthogonalDecomp::compute(black_box(a.as_ref()), black_box(1e-10))
                    .unwrap();
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_qr_standard,
    bench_qr_with_pivot,
    bench_lq_factorization,
    bench_rq_factorization,
    bench_complete_orthogonal
);
criterion_main!(benches);
