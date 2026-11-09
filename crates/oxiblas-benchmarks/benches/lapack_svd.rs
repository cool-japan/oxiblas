//! Benchmarks for LAPACK SVD operations.
//!
//! Inputs are full-rank, deterministic pseudo-random matrices (fixed-seed LCG,
//! entries in `[-1, 1)`), so the timings exercise the real decomposition work
//! rather than null-space completion of a rank-deficient matrix.
//!
//! - `jacobi_svd` / `svd_jacobi`: one-sided Jacobi [`Svd`].
//! - `divide_conquer_gesdd` / `svd_divide_conquer`: divide-and-conquer [`SvdDc`].

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use oxiblas_lapack::svd::{Svd, SvdDc};
use oxiblas_matrix::Mat;
use std::hint::black_box;

/// Fixed seed so every run (and every build) sees the same matrices.
const SEED: u64 = 0x2545_F491_4F6C_DD1D;

/// Deterministic full-rank `n × n` matrix with entries in `[-1, 1)` (64-bit
/// LCG with Knuth's MMIX constants, top 53 bits).
fn random_matrix(n: usize) -> Mat<f64> {
    let mut state = SEED ^ n as u64;
    let mut a = Mat::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
            a[(i, j)] = 2.0 * unit - 1.0;
        }
    }
    a
}

fn bench_svd_jacobi(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd_jacobi");

    for size in [50, 100, 200].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        let a = random_matrix(n);

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

    for size in [50, 100, 200, 500].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        let a = random_matrix(n);

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

    let a = random_matrix(100);

    group.bench_function("jacobi_svd", |b| {
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
    bench_svd_jacobi,
    bench_svd_divide_conquer,
    bench_svd_comparison
);
criterion_main!(benches);
