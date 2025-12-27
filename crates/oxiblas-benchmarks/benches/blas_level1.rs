//! Benchmarks for BLAS Level 1 operations (vector-vector).

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use oxiblas_blas::level1::{asum, axpy, copy, dot, nrm2, rot, rotg, rotm, rotmg, scal, swap};
use std::hint::black_box;

fn bench_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");

    for size in [100, 1000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let x: Vec<f64> = (0..*size).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..*size).map(|i| (i + 1) as f64).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| black_box(dot(black_box(&x), black_box(&y))));
        });
    }

    group.finish();
}

fn bench_axpy(c: &mut Criterion) {
    let mut group = c.benchmark_group("axpy");

    for size in [100, 1000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let x: Vec<f64> = (0..*size).map(|i| i as f64).collect();
        let mut y: Vec<f64> = (0..*size).map(|i| (i + 1) as f64).collect();
        let alpha = 2.5;

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                axpy(black_box(alpha), black_box(&x), black_box(&mut y));
            });
        });
    }

    group.finish();
}

fn bench_scal(c: &mut Criterion) {
    let mut group = c.benchmark_group("scal");

    for size in [100, 1000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let mut x: Vec<f64> = (0..*size).map(|i| i as f64).collect();
        let alpha = 2.5;

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                scal(black_box(alpha), black_box(&mut x));
            });
        });
    }

    group.finish();
}

fn bench_nrm2(c: &mut Criterion) {
    let mut group = c.benchmark_group("nrm2");

    for size in [100, 1000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let x: Vec<f64> = (0..*size).map(|i| i as f64).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| black_box(nrm2(black_box(&x))));
        });
    }

    group.finish();
}

fn bench_asum(c: &mut Criterion) {
    let mut group = c.benchmark_group("asum");

    for size in [100, 1000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let x: Vec<f64> = (0..*size)
            .map(|i| if i % 2 == 0 { i as f64 } else { -(i as f64) })
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| black_box(asum(black_box(&x))));
        });
    }

    group.finish();
}

fn bench_swap(c: &mut Criterion) {
    let mut group = c.benchmark_group("swap");

    for size in [100, 1000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64 * 2));

        let mut x: Vec<f64> = (0..*size).map(|i| i as f64).collect();
        let mut y: Vec<f64> = (0..*size).map(|i| (i + 1000) as f64).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                swap(black_box(&mut x), black_box(&mut y));
            });
        });
    }

    group.finish();
}

fn bench_copy(c: &mut Criterion) {
    let mut group = c.benchmark_group("copy");

    for size in [100, 1000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let x: Vec<f64> = (0..*size).map(|i| i as f64).collect();
        let mut y: Vec<f64> = vec![0.0; *size];

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                copy(black_box(&x), black_box(&mut y));
            });
        });
    }

    group.finish();
}

fn bench_rot(c: &mut Criterion) {
    let mut group = c.benchmark_group("rot");

    for size in [100, 1000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64 * 2));

        let mut x: Vec<f64> = (0..*size).map(|i| i as f64).collect();
        let mut y: Vec<f64> = (0..*size).map(|i| (i + 1) as f64).collect();
        // cos(45°) and sin(45°)
        let c_val = std::f64::consts::FRAC_1_SQRT_2;
        let s_val = std::f64::consts::FRAC_1_SQRT_2;

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                rot(
                    black_box(c_val),
                    black_box(s_val),
                    black_box(&mut x),
                    black_box(&mut y),
                );
            });
        });
    }

    group.finish();
}

fn bench_rotg(c: &mut Criterion) {
    let mut group = c.benchmark_group("rotg");

    // rotg operates on scalars, so we measure throughput differently
    group.throughput(Throughput::Elements(1));

    let a = 3.0f64;
    let b = 4.0f64;

    group.bench_function("scalar", |bench| {
        bench.iter(|| {
            let _ = rotg(black_box(a), black_box(b));
        });
    });

    group.finish();
}

fn bench_rotm(c: &mut Criterion) {
    let mut group = c.benchmark_group("rotm");

    for size in [100, 1000, 10_000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64 * 2));

        let mut x: Vec<f64> = (0..*size).map(|i| i as f64).collect();
        let mut y: Vec<f64> = (0..*size).map(|i| (i + 1) as f64).collect();

        // Generate rotation parameters
        let mut d1 = 1.0;
        let mut d2 = 1.0;
        let mut x1 = 3.0;
        let y1 = 4.0;
        let params = rotmg(&mut d1, &mut d2, &mut x1, y1);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                rotm(black_box(&params), black_box(&mut x), black_box(&mut y));
            });
        });
    }

    group.finish();
}

fn bench_rotmg(c: &mut Criterion) {
    let mut group = c.benchmark_group("rotmg");

    // rotmg operates on scalars
    group.throughput(Throughput::Elements(1));

    group.bench_function("scalar", |bench| {
        bench.iter(|| {
            let mut d1 = 1.0f64;
            let mut d2 = 1.0f64;
            let mut x1 = 3.0f64;
            let y1 = 4.0f64;
            let _ = rotmg(
                black_box(&mut d1),
                black_box(&mut d2),
                black_box(&mut x1),
                black_box(y1),
            );
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_dot,
    bench_axpy,
    bench_scal,
    bench_nrm2,
    bench_asum,
    bench_swap,
    bench_copy,
    bench_rot,
    bench_rotg,
    bench_rotm,
    bench_rotmg
);
criterion_main!(benches);
