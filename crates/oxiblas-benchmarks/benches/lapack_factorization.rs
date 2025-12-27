//! Benchmarks for LAPACK factorization operations (LU, Cholesky).

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use oxiblas_lapack::cholesky::Cholesky;
use oxiblas_lapack::lu::Lu;
use oxiblas_matrix::Mat;
use std::hint::black_box;

fn bench_lu_factorization(c: &mut Criterion) {
    let mut group = c.benchmark_group("lu_factorization");

    for size in [50, 100, 200, 500].iter() {
        let n = *size;
        // LU factorization: O(2/3 * n^3) FLOPs
        group.throughput(Throughput::Elements((2 * n * n * n / 3) as u64));

        // Create a diagonally dominant matrix to ensure non-singularity
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

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = Lu::compute(black_box(a.as_ref()));
            });
        });
    }

    group.finish();
}

fn bench_lu_blocked(c: &mut Criterion) {
    let mut group = c.benchmark_group("lu_blocked");

    for size in [100, 200, 500].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((2 * n * n * n / 3) as u64));

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

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = Lu::compute_blocked(black_box(a.as_ref()));
            });
        });
    }

    group.finish();
}

fn bench_lu_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("lu_solve");

    for size in [50, 100, 200, 500].iter() {
        let n = *size;
        // Forward + backward substitution: O(2 * n^2) FLOPs
        group.throughput(Throughput::Elements((2 * n * n) as u64));

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

        // Pre-compute the LU factorization
        let lu = Lu::compute(a.as_ref()).unwrap();

        // Create RHS
        let b_data: Vec<f64> = (0..n).map(|i| (i % 10) as f64).collect();
        let b = Mat::from_slice(n, 1, &b_data);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = lu.solve(black_box(b.as_ref()));
            });
        });
    }

    group.finish();
}

fn bench_cholesky_factorization(c: &mut Criterion) {
    let mut group = c.benchmark_group("cholesky_factorization");

    for size in [50, 100, 200, 500].iter() {
        let n = *size;
        // Cholesky factorization: O(1/3 * n^3) FLOPs
        group.throughput(Throughput::Elements((n * n * n / 3) as u64));

        // Create SPD matrix: A = B^T * B + n*I
        let mut a: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                let val = if i == j {
                    (n as f64) + ((i * 17) % 10) as f64
                } else {
                    ((i + j) % 10) as f64 * 0.1
                };
                a[(i, j)] = val;
                a[(j, i)] = val;
            }
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = Cholesky::compute(black_box(a.as_ref()));
            });
        });
    }

    group.finish();
}

fn bench_cholesky_blocked(c: &mut Criterion) {
    let mut group = c.benchmark_group("cholesky_blocked");

    for size in [100, 200, 500].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n * n / 3) as u64));

        let mut a: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                let val = if i == j {
                    (n as f64) + ((i * 17) % 10) as f64
                } else {
                    ((i + j) % 10) as f64 * 0.1
                };
                a[(i, j)] = val;
                a[(j, i)] = val;
            }
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = Cholesky::compute_blocked(black_box(a.as_ref()));
            });
        });
    }

    group.finish();
}

fn bench_cholesky_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("cholesky_solve");

    for size in [50, 100, 200, 500].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((2 * n * n) as u64));

        let mut a: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                let val = if i == j {
                    (n as f64) + ((i * 17) % 10) as f64
                } else {
                    ((i + j) % 10) as f64 * 0.1
                };
                a[(i, j)] = val;
                a[(j, i)] = val;
            }
        }

        let chol = Cholesky::compute(a.as_ref()).unwrap();
        let b_data: Vec<f64> = (0..n).map(|i| (i % 10) as f64).collect();
        let b = Mat::from_slice(n, 1, &b_data);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = chol.solve(black_box(b.as_ref()));
            });
        });
    }

    group.finish();
}

fn bench_lu_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("lu_inverse");

    for size in [50, 100, 200].iter() {
        let n = *size;
        // Matrix inversion via LU: O(2 * n^3) FLOPs
        group.throughput(Throughput::Elements((2 * n * n * n) as u64));

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

        let lu = Lu::compute(a.as_ref()).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = lu.inverse();
            });
        });
    }

    group.finish();
}

fn bench_cholesky_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("cholesky_inverse");

    for size in [50, 100, 200].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((2 * n * n * n) as u64));

        let mut a: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                let val = if i == j {
                    (n as f64) + ((i * 17) % 10) as f64
                } else {
                    ((i + j) % 10) as f64 * 0.1
                };
                a[(i, j)] = val;
                a[(j, i)] = val;
            }
        }

        let chol = Cholesky::compute(a.as_ref()).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = chol.inverse();
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_lu_factorization,
    bench_lu_blocked,
    bench_lu_solve,
    bench_cholesky_factorization,
    bench_cholesky_blocked,
    bench_cholesky_solve,
    bench_lu_inverse,
    bench_cholesky_inverse
);
criterion_main!(benches);
