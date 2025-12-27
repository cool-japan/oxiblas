//! Benchmarks for newly implemented features.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use oxiblas_blas::level1::{dot_kahan, dot_pairwise, dsdot};
use oxiblas_blas::tensor::{Tensor3, batched_matmul, einsum, outer_product};
use std::hint::black_box;

fn bench_extended_precision_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("extended_precision_dot");

    for size in [1000, 10_000, 100_000].iter() {
        let n = *size;
        group.throughput(Throughput::Elements(n as u64));

        let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = (0..n).map(|i| (i + 1) as f64 * 0.1).collect();

        group.bench_with_input(BenchmarkId::new("dot_kahan", size), size, |bench, _| {
            bench.iter(|| black_box(dot_kahan(black_box(&x), black_box(&y))));
        });

        group.bench_with_input(BenchmarkId::new("dot_pairwise", size), size, |bench, _| {
            bench.iter(|| black_box(dot_pairwise(black_box(&x), black_box(&y))));
        });

        // Mixed precision
        let x_f32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let y_f32: Vec<f32> = y.iter().map(|&v| v as f32).collect();

        group.bench_with_input(BenchmarkId::new("dsdot", size), size, |bench, _| {
            bench.iter(|| black_box(dsdot(black_box(&x_f32), black_box(&y_f32))));
        });
    }

    group.finish();
}

fn bench_einsum_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("einsum");

    let sizes = [32, 64, 128];

    for &n in &sizes {
        // Matrix multiplication: "ij,jk->ik"
        let a = (0..n * n).map(|i| i as f64).collect::<Vec<_>>();
        let b = (0..n * n).map(|i| (i + 1) as f64).collect::<Vec<_>>();

        group.bench_with_input(BenchmarkId::new("matmul", n), &n, |bench, _| {
            bench.iter(|| {
                let _ = einsum(
                    black_box("ij,jk->ik"),
                    black_box(&a),
                    black_box(&[n, n]),
                    black_box(Some((&b, &[n, n]))),
                );
            });
        });

        // Outer product: "i,j->ij"
        let x = (0..n).map(|i| i as f64).collect::<Vec<_>>();
        let y = (0..n).map(|i| (i + 1) as f64).collect::<Vec<_>>();

        group.bench_with_input(BenchmarkId::new("outer_product", n), &n, |bench, _| {
            bench.iter(|| {
                let _ = einsum(
                    black_box("i,j->ij"),
                    black_box(&x),
                    black_box(&[n]),
                    black_box(Some((&y, &[n]))),
                );
            });
        });

        // Hadamard product: "ij,ij->ij"
        group.bench_with_input(BenchmarkId::new("hadamard", n), &n, |bench, _| {
            bench.iter(|| {
                let _ = einsum(
                    black_box("ij,ij->ij"),
                    black_box(&a),
                    black_box(&[n, n]),
                    black_box(Some((&b, &[n, n]))),
                );
            });
        });
    }

    group.finish();
}

fn bench_tensor_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_operations");

    for &batch_size in &[4, 8, 16] {
        let n = 64;
        let total_size = batch_size * n * n;

        group.throughput(Throughput::Elements(total_size as u64));

        let a: Tensor3<f64> = Tensor3::zeros(batch_size, n, n);
        let b: Tensor3<f64> = Tensor3::zeros(batch_size, n, n);

        group.bench_with_input(
            BenchmarkId::new("batched_matmul", format!("{}x{}x{}", batch_size, n, n)),
            &batch_size,
            |bench, _| {
                bench.iter(|| {
                    let _ = batched_matmul(black_box(&a), black_box(&b));
                });
            },
        );
    }

    group.finish();
}

fn bench_outer_product_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("outer_product_scaling");

    for &n in &[50, 100, 200, 500] {
        group.throughput(Throughput::Elements((n * n) as u64));

        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                let _ = outer_product(black_box(&x), black_box(&y));
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_extended_precision_dot,
    bench_einsum_patterns,
    bench_tensor_operations,
    bench_outer_product_scaling
);
criterion_main!(benches);
