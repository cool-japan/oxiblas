//! Benchmarks for LAPACK linear system solve operations.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use oxiblas_lapack::solve::{
    TriangularKind, lstsq, solve, solve_multiple, solve_triangular, tridiag_solve,
    tridiag_solve_spd,
};
use oxiblas_matrix::Mat;
use std::hint::black_box;

fn bench_solve_general(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve_general");

    for size in [50, 100, 200, 500].iter() {
        let n = *size;
        // General solve via LU: O(n^3) for factor + O(n^2) for solve
        group.throughput(Throughput::Elements((n * n * n) as u64));

        // Create a well-conditioned matrix
        let mut a_data = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a_data[i * n + j] = if i == j {
                    (n as f64) + 1.0
                } else {
                    ((i * 17 + j * 31) % 10) as f64 * 0.1
                };
            }
        }
        let a = Mat::from_slice(n, n, &a_data);
        let b_data: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.1).collect();
        let b = Mat::from_slice(n, 1, &b_data);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = solve(black_box(a.as_ref()), black_box(b.as_ref()));
            });
        });
    }

    group.finish();
}

fn bench_solve_multiple_rhs(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve_multiple_rhs");

    for size in [50, 100, 200].iter() {
        let n = *size;
        let nrhs = 10;
        // Multiple RHS solve: O(n^3) for factor + O(n^2 * nrhs) for solve
        group.throughput(Throughput::Elements((n * n * n + n * n * nrhs) as u64));

        let mut a_data = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                a_data[i * n + j] = if i == j {
                    (n as f64) + 1.0
                } else {
                    ((i * 17 + j * 31) % 10) as f64 * 0.1
                };
            }
        }
        let a = Mat::from_slice(n, n, &a_data);
        let b_data: Vec<f64> = (0..n * nrhs).map(|i| (i % 100) as f64 * 0.1).collect();
        let b = Mat::from_slice(n, nrhs, &b_data);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = solve_multiple(black_box(a.as_ref()), black_box(b.as_ref()));
            });
        });
    }

    group.finish();
}

fn bench_solve_triangular_upper(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve_triangular_upper");

    for size in [50, 100, 200, 500].iter() {
        let n = *size;
        // Triangular solve: O(n^2) via back substitution
        group.throughput(Throughput::Elements((n * n) as u64));

        // Create upper triangular matrix with good diagonal
        let mut a_data = vec![0.0f64; n * n];
        for i in 0..n {
            for j in i..n {
                a_data[i * n + j] = if i == j {
                    1.0 + (i % 10) as f64 * 0.1
                } else {
                    ((i * 17 + j * 31) % 10) as f64 * 0.1
                };
            }
        }
        let a = Mat::from_slice(n, n, &a_data);
        let b_data: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.1).collect();
        let b = Mat::from_slice(n, 1, &b_data);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = solve_triangular(
                    black_box(a.as_ref()),
                    black_box(b.as_ref()),
                    black_box(TriangularKind::Upper),
                );
            });
        });
    }

    group.finish();
}

fn bench_solve_triangular_lower(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve_triangular_lower");

    for size in [50, 100, 200, 500].iter() {
        let n = *size;
        // Triangular solve: O(n^2) via forward substitution
        group.throughput(Throughput::Elements((n * n) as u64));

        // Create lower triangular matrix with good diagonal
        let mut a_data = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..=i {
                a_data[i * n + j] = if i == j {
                    1.0 + (i % 10) as f64 * 0.1
                } else {
                    ((i * 17 + j * 31) % 10) as f64 * 0.1
                };
            }
        }
        let a = Mat::from_slice(n, n, &a_data);
        let b_data: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.1).collect();
        let b = Mat::from_slice(n, 1, &b_data);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = solve_triangular(
                    black_box(a.as_ref()),
                    black_box(b.as_ref()),
                    black_box(TriangularKind::Lower),
                );
            });
        });
    }

    group.finish();
}

fn bench_tridiag_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("tridiag_solve");

    for size in [100, 500, 1000, 5000, 10000].iter() {
        let n = *size;
        // Tridiagonal solve: O(n) via Thomas algorithm
        group.throughput(Throughput::Elements(n as u64));

        // Standard tridiagonal (like 1D Laplacian)
        let dl: Vec<f64> = vec![-1.0; n - 1];
        let d: Vec<f64> = vec![2.0; n];
        let du: Vec<f64> = vec![-1.0; n - 1];
        let b: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.01).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = tridiag_solve(black_box(&dl), black_box(&d), black_box(&du), black_box(&b));
            });
        });
    }

    group.finish();
}

fn bench_tridiag_solve_spd(c: &mut Criterion) {
    let mut group = c.benchmark_group("tridiag_solve_spd");

    for size in [100, 500, 1000, 5000, 10000].iter() {
        let n = *size;
        // SPD tridiagonal solve: O(n) via LDL^T factorization
        group.throughput(Throughput::Elements(n as u64));

        // SPD tridiagonal (symmetric)
        let d: Vec<f64> = vec![4.0; n];
        let e: Vec<f64> = vec![-1.0; n - 1];
        let b: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.01).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = tridiag_solve_spd(black_box(&d), black_box(&e), black_box(&b));
            });
        });
    }

    group.finish();
}

fn bench_lstsq(c: &mut Criterion) {
    let mut group = c.benchmark_group("lstsq");

    // Overdetermined systems with different aspect ratios
    for (m, n) in [(100, 50), (200, 50), (500, 100), (1000, 100)].iter() {
        let m = *m;
        let n = *n;
        // Least squares via QR: O(mn^2) for QR + O(n^2) for solve
        group.throughput(Throughput::Elements((m * n * n) as u64));

        // Create a well-conditioned matrix
        let a_data: Vec<f64> = (0..m * n)
            .map(|i| {
                let row = i / n;
                let col = i % n;
                if row == col {
                    1.0 + (row % 10) as f64 * 0.1
                } else {
                    ((row * 17 + col * 31) % 10) as f64 * 0.01
                }
            })
            .collect();
        let a = Mat::from_slice(m, n, &a_data);
        let b_data: Vec<f64> = (0..m).map(|i| (i % 100) as f64 * 0.1).collect();
        let b = Mat::from_slice(m, 1, &b_data);

        let label = format!("{}x{}", m, n);
        group.bench_with_input(BenchmarkId::from_parameter(&label), &(m, n), |bench, _| {
            bench.iter(|| {
                let _ = lstsq(black_box(a.as_ref()), black_box(b.as_ref()));
            });
        });
    }

    group.finish();
}

fn bench_solve_triangular_unit(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve_triangular_unit");

    for size in [50, 100, 200, 500].iter() {
        let n = *size;
        // Unit triangular solve: O(n^2) (no division by diagonal)
        group.throughput(Throughput::Elements((n * n) as u64));

        // Create unit lower triangular matrix
        let mut a_data = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..=i {
                a_data[i * n + j] = if i == j {
                    1.0 // Unit diagonal
                } else {
                    ((i * 17 + j * 31) % 10) as f64 * 0.1
                };
            }
        }
        let a = Mat::from_slice(n, n, &a_data);
        let b_data: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.1).collect();
        let b = Mat::from_slice(n, 1, &b_data);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = solve_triangular(
                    black_box(a.as_ref()),
                    black_box(b.as_ref()),
                    black_box(TriangularKind::UnitLower),
                );
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_solve_general,
    bench_solve_multiple_rhs,
    bench_solve_triangular_upper,
    bench_solve_triangular_lower,
    bench_tridiag_solve,
    bench_tridiag_solve_spd,
    bench_lstsq,
    bench_solve_triangular_unit
);
criterion_main!(benches);
