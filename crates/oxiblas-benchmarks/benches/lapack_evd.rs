//! Benchmarks for LAPACK eigenvalue decomposition operations.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use oxiblas_lapack::evd::{GeneralEvd, Hessenberg, Schur, SymmetricEvd, SymmetricEvdDc};
use oxiblas_matrix::Mat;
use std::hint::black_box;

fn bench_symmetric_evd(c: &mut Criterion) {
    let mut group = c.benchmark_group("symmetric_evd");

    for size in [50, 100, 200].iter() {
        let n = *size;
        // Symmetric EVD: O(n^3) FLOPs
        group.throughput(Throughput::Elements((n * n * n) as u64));

        // Create SPD matrix
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
                let _ = SymmetricEvd::compute(black_box(a.as_ref()));
            });
        });
    }

    group.finish();
}

fn bench_symmetric_evd_dc(c: &mut Criterion) {
    let mut group = c.benchmark_group("symmetric_evd_dc");

    for size in [50, 100, 200].iter() {
        let n = *size;
        // Divide and conquer: more efficient for larger matrices
        group.throughput(Throughput::Elements((n * n * n) as u64));

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
                let _ = SymmetricEvdDc::compute(black_box(a.as_ref()));
            });
        });
    }

    group.finish();
}

fn bench_symmetric_evd_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("symmetric_evd_large");

    for size in [300, 500].iter() {
        let n = *size;
        // Larger matrices for Symmetric EVD
        group.throughput(Throughput::Elements((n * n * n) as u64));

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
                let _ = SymmetricEvd::compute(black_box(a.as_ref()));
            });
        });
    }

    group.finish();
}

fn bench_general_evd(c: &mut Criterion) {
    let mut group = c.benchmark_group("general_evd");

    for size in [50, 100, 200].iter() {
        let n = *size;
        // General EVD: O(n^3) FLOPs
        group.throughput(Throughput::Elements((n * n * n) as u64));

        // Create a general (non-symmetric) matrix
        let mut a: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = ((i * 17 + j * 31) % 100) as f64 * 0.01;
                if i == j {
                    a[(i, j)] += 1.0; // Ensure non-singular
                }
            }
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = GeneralEvd::compute(black_box(a.as_ref()));
            });
        });
    }

    group.finish();
}

fn bench_general_eigenvalues_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("general_eigenvalues_only");

    for size in [50, 100, 200, 500].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        let mut a: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = ((i * 17 + j * 31) % 100) as f64 * 0.01;
                if i == j {
                    a[(i, j)] += 1.0;
                }
            }
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = GeneralEvd::eigenvalues_only(black_box(a.as_ref()));
            });
        });
    }

    group.finish();
}

fn bench_hessenberg(c: &mut Criterion) {
    let mut group = c.benchmark_group("hessenberg");

    for size in [50, 100, 200, 500].iter() {
        let n = *size;
        // Hessenberg reduction: O(10/3 * n^3) FLOPs
        group.throughput(Throughput::Elements((10 * n * n * n / 3) as u64));

        let mut a: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = ((i * 17 + j * 31) % 100) as f64 * 0.01;
            }
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = Hessenberg::compute(black_box(a.as_ref()));
            });
        });
    }

    group.finish();
}

fn bench_schur(c: &mut Criterion) {
    let mut group = c.benchmark_group("schur");

    for size in [50, 100, 200].iter() {
        let n = *size;
        // Schur decomposition: O(25 * n^3) FLOPs (Hessenberg + QR iteration)
        group.throughput(Throughput::Elements((25 * n * n * n) as u64));

        let mut a: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = ((i * 17 + j * 31) % 100) as f64 * 0.01;
                if i == j {
                    a[(i, j)] += 1.0;
                }
            }
        }

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = Schur::compute(black_box(a.as_ref()));
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_symmetric_evd,
    bench_symmetric_evd_dc,
    bench_symmetric_evd_large,
    bench_general_evd,
    bench_general_eigenvalues_only,
    bench_hessenberg,
    bench_schur
);
criterion_main!(benches);
