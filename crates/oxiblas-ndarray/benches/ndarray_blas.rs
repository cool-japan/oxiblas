//! Benchmarks for oxiblas-ndarray operations.
//!
//! Compares performance against pure ndarray operations.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ndarray::{Array1, Array2, ArrayD, IxDyn, ShapeBuilder};
use oxiblas_ndarray::prelude::*;
use std::hint::black_box;

// =============================================================================
// Matrix Multiplication Benchmarks
// =============================================================================

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    for size in [32, 64, 128, 256, 512].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n * n) as u64));

        // Create column-major arrays for best oxiblas performance
        let a: Array2<f64> = Array2::from_shape_fn((n, n).f(), |(i, j)| (i * n + j) as f64);
        let b: Array2<f64> = Array2::from_shape_fn((n, n).f(), |(i, j)| (i + j) as f64);

        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| black_box(matmul(&a, &b)));
        });

        group.bench_with_input(BenchmarkId::new("ndarray_dot", size), size, |bench, _| {
            bench.iter(|| black_box(a.dot(&b)));
        });
    }

    group.finish();
}

fn bench_matmul_rectangular(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_rectangular");

    for &(m, k, n) in [
        (64, 128, 64),
        (128, 64, 128),
        (256, 128, 64),
        (512, 64, 256),
    ]
    .iter()
    {
        let param = format!("{}x{}x{}", m, k, n);
        group.throughput(Throughput::Elements((m * k * n) as u64));

        let a: Array2<f64> = Array2::from_shape_fn((m, k).f(), |(i, j)| (i * k + j) as f64);
        let b: Array2<f64> = Array2::from_shape_fn((k, n).f(), |(i, j)| (i + j) as f64);

        group.bench_with_input(BenchmarkId::new("oxiblas", &param), &param, |bench, _| {
            bench.iter(|| black_box(matmul(&a, &b)));
        });

        group.bench_with_input(
            BenchmarkId::new("ndarray_dot", &param),
            &param,
            |bench, _| {
                bench.iter(|| black_box(a.dot(&b)));
            },
        );
    }

    group.finish();
}

// =============================================================================
// Matrix-Vector Multiplication Benchmarks
// =============================================================================

fn bench_matvec(c: &mut Criterion) {
    let mut group = c.benchmark_group("matvec");

    for size in [64, 128, 256, 512, 1024].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        let a: Array2<f64> = Array2::from_shape_fn((n, n).f(), |(i, j)| (i * n + j) as f64);
        let x: Array1<f64> = Array1::from_vec((0..n).map(|i| i as f64).collect());

        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| black_box(matvec(&a, &x)));
        });

        group.bench_with_input(BenchmarkId::new("ndarray_dot", size), size, |bench, _| {
            bench.iter(|| black_box(a.dot(&x)));
        });
    }

    group.finish();
}

// =============================================================================
// Vector Operations Benchmarks
// =============================================================================

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");

    for size in [1000, 10000, 100000, 1000000].iter() {
        let n = *size;
        group.throughput(Throughput::Elements(*size as u64));

        let x: Array1<f64> = Array1::from_vec((0..n).map(|i| i as f64).collect());
        let y: Array1<f64> = Array1::from_vec((0..n).map(|i| (i * 2) as f64).collect());

        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| black_box(dot_ndarray(&x, &y)));
        });

        group.bench_with_input(BenchmarkId::new("ndarray_dot", size), size, |bench, _| {
            bench.iter(|| black_box(x.dot(&y)));
        });
    }

    group.finish();
}

fn bench_nrm2(c: &mut Criterion) {
    let mut group = c.benchmark_group("nrm2");

    for size in [1000, 10000, 100000, 1000000].iter() {
        let n = *size;
        group.throughput(Throughput::Elements(*size as u64));

        let x: Array1<f64> = Array1::from_vec((0..n).map(|i| i as f64).collect());

        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| black_box(nrm2_ndarray(&x)));
        });

        group.bench_with_input(BenchmarkId::new("ndarray_fold", size), size, |bench, _| {
            bench.iter(|| black_box(x.iter().map(|v| v * v).sum::<f64>().sqrt()));
        });
    }

    group.finish();
}

// =============================================================================
// Conversion Benchmarks
// =============================================================================

fn bench_array2_to_mat(c: &mut Criterion) {
    let mut group = c.benchmark_group("array2_to_mat");

    for size in [64, 128, 256, 512, 1024].iter() {
        let n = *size;
        group.throughput(Throughput::Bytes((n * n * 8) as u64));

        // Column-major
        let col_major: Array2<f64> = Array2::from_shape_fn((n, n).f(), |(i, j)| (i * n + j) as f64);

        group.bench_with_input(BenchmarkId::new("col_major", size), size, |bench, _| {
            bench.iter(|| black_box(array2_to_mat(&col_major)));
        });

        // Row-major
        let row_major: Array2<f64> = Array2::from_shape_fn((n, n), |(i, j)| (i * n + j) as f64);

        group.bench_with_input(BenchmarkId::new("row_major", size), size, |bench, _| {
            bench.iter(|| black_box(array2_to_mat(&row_major)));
        });
    }

    group.finish();
}

fn bench_arrayd_conversions(c: &mut Criterion) {
    let mut group = c.benchmark_group("arrayd_conversions");

    for size in [64, 128, 256, 512].iter() {
        let n = *size;
        group.throughput(Throughput::Bytes((n * n * 8) as u64));

        let arr_d = ArrayD::from_shape_fn(IxDyn(&[n, n]), |idx| (idx[0] * n + idx[1]) as f64);

        group.bench_with_input(BenchmarkId::new("arrayd_to_mat", size), size, |bench, _| {
            bench.iter(|| black_box(arrayd_to_mat(&arr_d)));
        });

        group.bench_with_input(
            BenchmarkId::new("arrayd_to_array2", size),
            size,
            |bench, _| {
                bench.iter(|| black_box(arrayd_to_array2(&arr_d)));
            },
        );
    }

    group.finish();
}

// =============================================================================
// Norm Benchmarks
// =============================================================================

fn bench_frobenius_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("frobenius_norm");

    for size in [64, 128, 256, 512, 1024].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        let a: Array2<f64> = Array2::from_shape_fn((n, n).f(), |(i, j)| (i * n + j) as f64);

        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| black_box(frobenius_norm(&a)));
        });

        group.bench_with_input(BenchmarkId::new("ndarray_fold", size), size, |bench, _| {
            bench.iter(|| black_box(a.iter().map(|v| v * v).sum::<f64>().sqrt()));
        });
    }

    group.finish();
}

// =============================================================================
// Memory Layout Benchmarks
// =============================================================================

fn bench_layout_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("layout_conversion");

    for size in [64, 128, 256, 512].iter() {
        let n = *size;
        group.throughput(Throughput::Bytes((n * n * 8) as u64));

        let row_major: Array2<f64> = Array2::from_shape_fn((n, n), |(i, j)| (i * n + j) as f64);

        group.bench_with_input(
            BenchmarkId::new("to_column_major", size),
            size,
            |bench, _| {
                bench.iter(|| black_box(to_column_major(&row_major)));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_matmul,
    bench_matmul_rectangular,
    bench_matvec,
    bench_dot_product,
    bench_nrm2,
    bench_array2_to_mat,
    bench_arrayd_conversions,
    bench_frobenius_norm,
    bench_layout_conversion,
);
criterion_main!(benches);
