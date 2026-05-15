//! Benchmarks for advanced BLAS features and optimizations.
//!
//! This module benchmarks:
//! - Parallel Level 1 operations
//! - Strassen algorithm for large matrix multiplication
//! - Asymmetric blocking strategies for GEMM
//! - Complex TRSM optimization
//! - Scaling tests (sequential vs parallel)

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use num_complex::Complex64;
use oxiblas_blas::level1::parallel::{axpy_par, dot_par, nrm2_par, scal_par};
use oxiblas_blas::level1::{axpy, dot, nrm2, scal};
use oxiblas_blas::level3::{
    Diag, Side, Trans, Uplo, gemm, gemm_asymmetric, gemm_strassen, should_use_strassen, trsm,
};
use oxiblas_core::parallel::Par;
use oxiblas_matrix::Mat;

// =============================================================================
// Parallel Level 1 Benchmarks
// =============================================================================

fn bench_parallel_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_dot");

    for size in [10_000, 100_000, 1_000_000].iter() {
        let n = *size;
        group.throughput(Throughput::Elements(n as u64));

        let x: Vec<f64> = (0..n).map(|i| (i % 1000) as f64 * 0.001).collect();
        let y: Vec<f64> = (0..n).map(|i| ((i + 1) % 1000) as f64 * 0.001).collect();

        // Sequential
        group.bench_with_input(BenchmarkId::new("sequential", size), size, |bench, _| {
            bench.iter(|| black_box(dot(black_box(&x), black_box(&y))));
        });

        // Parallel
        group.bench_with_input(BenchmarkId::new("parallel", size), size, |bench, _| {
            bench.iter(|| black_box(dot_par(black_box(&x), black_box(&y), Par::Rayon)));
        });
    }

    group.finish();
}

fn bench_parallel_axpy(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_axpy");

    for size in [10_000, 100_000, 1_000_000].iter() {
        let n = *size;
        group.throughput(Throughput::Elements(n as u64));

        let x: Vec<f64> = (0..n).map(|i| (i % 1000) as f64 * 0.001).collect();
        let alpha = 2.5;

        // Sequential
        group.bench_with_input(BenchmarkId::new("sequential", size), size, |bench, _| {
            let mut y: Vec<f64> = vec![1.0; n];
            bench.iter(|| {
                axpy(black_box(alpha), black_box(&x), black_box(&mut y));
            });
        });

        // Parallel
        group.bench_with_input(BenchmarkId::new("parallel", size), size, |bench, _| {
            let mut y: Vec<f64> = vec![1.0; n];
            bench.iter(|| {
                axpy_par(
                    black_box(alpha),
                    black_box(&x),
                    black_box(&mut y),
                    Par::Rayon,
                );
            });
        });
    }

    group.finish();
}

fn bench_parallel_nrm2(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_nrm2");

    for size in [10_000, 100_000, 1_000_000].iter() {
        let n = *size;
        group.throughput(Throughput::Elements(n as u64));

        let x: Vec<f64> = (0..n).map(|i| (i % 1000) as f64 * 0.001).collect();

        // Sequential
        group.bench_with_input(BenchmarkId::new("sequential", size), size, |bench, _| {
            bench.iter(|| black_box(nrm2(black_box(&x))));
        });

        // Parallel
        group.bench_with_input(BenchmarkId::new("parallel", size), size, |bench, _| {
            bench.iter(|| black_box(nrm2_par(black_box(&x), Par::Rayon)));
        });
    }

    group.finish();
}

fn bench_parallel_scal(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_scal");

    for size in [10_000, 100_000, 1_000_000].iter() {
        let n = *size;
        group.throughput(Throughput::Elements(n as u64));

        let alpha = 2.5;

        // Sequential
        group.bench_with_input(BenchmarkId::new("sequential", size), size, |bench, _| {
            let mut x: Vec<f64> = vec![1.0; n];
            bench.iter(|| {
                scal(black_box(alpha), black_box(&mut x));
            });
        });

        // Parallel
        group.bench_with_input(BenchmarkId::new("parallel", size), size, |bench, _| {
            let mut x: Vec<f64> = vec![1.0; n];
            bench.iter(|| {
                scal_par(black_box(alpha), black_box(&mut x), Par::Rayon);
            });
        });
    }

    group.finish();
}

// =============================================================================
// Strassen Algorithm Benchmarks
// =============================================================================

fn bench_strassen_vs_standard(c: &mut Criterion) {
    let mut group = c.benchmark_group("strassen_vs_standard");

    // Only benchmark sizes where Strassen is effective
    for size in [256, 512, 1024].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n * n) as u64));

        let a_data: Vec<f64> = (0..n * n).map(|i| (i % 1000) as f64 * 0.001).collect();
        let a = Mat::from_slice(n, n, &a_data);
        let b_data: Vec<f64> = (0..n * n)
            .map(|i| ((i + 1) % 1000) as f64 * 0.001)
            .collect();
        let b = Mat::from_slice(n, n, &b_data);

        // Standard GEMM
        group.bench_with_input(BenchmarkId::new("standard", size), size, |bench, _| {
            let mut c_mat = Mat::zeros(n, n);
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

        // Strassen (only for larger sizes)
        if should_use_strassen(n, n, n) {
            group.bench_with_input(BenchmarkId::new("strassen", size), size, |bench, _| {
                let mut c_mat = Mat::zeros(n, n);
                bench.iter(|| {
                    gemm_strassen(
                        black_box(1.0),
                        black_box(a.as_ref()),
                        black_box(b.as_ref()),
                        black_box(0.0),
                        black_box(c_mat.as_mut()),
                    );
                });
            });
        }
    }

    group.finish();
}

// =============================================================================
// Asymmetric Blocking Benchmarks
// =============================================================================

fn bench_asymmetric_tall_thin(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_asymmetric_tall_thin");

    // Tall-thin matrices: M >> N, K
    for &(m, k, n) in &[(2048, 64, 64), (4096, 32, 32), (8192, 64, 64)] {
        let param = format!("{}x{}x{}", m, k, n);
        group.throughput(Throughput::Elements((m * k * n) as u64));

        let a_data: Vec<f64> = (0..m * k).map(|i| (i % 1000) as f64 * 0.001).collect();
        let a = Mat::from_slice(m, k, &a_data);
        let b_data: Vec<f64> = (0..k * n).map(|i| (i % 1000) as f64 * 0.001).collect();
        let b = Mat::from_slice(k, n, &b_data);

        // Standard GEMM
        group.bench_with_input(BenchmarkId::new("standard", &param), &param, |bench, _| {
            let mut c_mat = Mat::zeros(m, n);
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

        // Asymmetric blocking
        group.bench_with_input(
            BenchmarkId::new("asymmetric", &param),
            &param,
            |bench, _| {
                let mut c_mat = Mat::zeros(m, n);
                bench.iter(|| {
                    gemm_asymmetric(
                        black_box(1.0),
                        black_box(a.as_ref()),
                        black_box(b.as_ref()),
                        black_box(0.0),
                        black_box(c_mat.as_mut()),
                    );
                });
            },
        );
    }

    group.finish();
}

fn bench_asymmetric_short_wide(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_asymmetric_short_wide");

    // Short-wide matrices: N >> M, K
    for &(m, k, n) in &[(64, 64, 2048), (32, 32, 4096), (64, 64, 8192)] {
        let param = format!("{}x{}x{}", m, k, n);
        group.throughput(Throughput::Elements((m * k * n) as u64));

        let a_data: Vec<f64> = (0..m * k).map(|i| (i % 1000) as f64 * 0.001).collect();
        let a = Mat::from_slice(m, k, &a_data);
        let b_data: Vec<f64> = (0..k * n).map(|i| (i % 1000) as f64 * 0.001).collect();
        let b = Mat::from_slice(k, n, &b_data);

        // Standard GEMM
        group.bench_with_input(BenchmarkId::new("standard", &param), &param, |bench, _| {
            let mut c_mat = Mat::zeros(m, n);
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

        // Asymmetric blocking
        group.bench_with_input(
            BenchmarkId::new("asymmetric", &param),
            &param,
            |bench, _| {
                let mut c_mat = Mat::zeros(m, n);
                bench.iter(|| {
                    gemm_asymmetric(
                        black_box(1.0),
                        black_box(a.as_ref()),
                        black_box(b.as_ref()),
                        black_box(0.0),
                        black_box(c_mat.as_mut()),
                    );
                });
            },
        );
    }

    group.finish();
}

fn bench_asymmetric_inner_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_asymmetric_inner_product");

    // Inner product shape: K >> M, N
    for &(m, k, n) in &[(64, 2048, 64), (32, 4096, 32), (64, 8192, 64)] {
        let param = format!("{}x{}x{}", m, k, n);
        group.throughput(Throughput::Elements((m * k * n) as u64));

        let a_data: Vec<f64> = (0..m * k).map(|i| (i % 1000) as f64 * 0.001).collect();
        let a = Mat::from_slice(m, k, &a_data);
        let b_data: Vec<f64> = (0..k * n).map(|i| (i % 1000) as f64 * 0.001).collect();
        let b = Mat::from_slice(k, n, &b_data);

        // Standard GEMM
        group.bench_with_input(BenchmarkId::new("standard", &param), &param, |bench, _| {
            let mut c_mat = Mat::zeros(m, n);
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

        // Asymmetric blocking
        group.bench_with_input(
            BenchmarkId::new("asymmetric", &param),
            &param,
            |bench, _| {
                let mut c_mat = Mat::zeros(m, n);
                bench.iter(|| {
                    gemm_asymmetric(
                        black_box(1.0),
                        black_box(a.as_ref()),
                        black_box(b.as_ref()),
                        black_box(0.0),
                        black_box(c_mat.as_mut()),
                    );
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// TRSM Benchmarks
// =============================================================================

fn bench_trsm_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("trsm_f64");

    for size in [64, 128, 256, 512].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        // Create upper triangular matrix with non-zero diagonal
        let mut a_data = vec![0.0; n * n];
        for i in 0..n {
            for j in i..n {
                a_data[i + j * n] = if i == j {
                    (i + 1) as f64
                } else {
                    ((i * n + j) % 1000) as f64 * 0.01
                };
            }
        }
        let a = Mat::from_slice(n, n, &a_data);

        let b_data: Vec<f64> = (0..n * n).map(|i| (i % 1000) as f64 * 0.01).collect();
        let b = Mat::from_slice(n, n, &b_data);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = trsm(
                    black_box(Side::Left),
                    black_box(Uplo::Upper),
                    black_box(Trans::NoTrans),
                    black_box(Diag::NonUnit),
                    black_box(1.0),
                    black_box(a.as_ref()),
                    black_box(b.as_ref()),
                );
            });
        });
    }

    group.finish();
}

fn bench_trsm_complex(c: &mut Criterion) {
    let mut group = c.benchmark_group("trsm_complex");

    for size in [32, 64, 128, 256].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        // Create upper triangular complex matrix
        let mut a_data = vec![Complex64::new(0.0, 0.0); n * n];
        for i in 0..n {
            for j in i..n {
                a_data[i + j * n] = if i == j {
                    Complex64::new((i + 1) as f64, 0.0)
                } else {
                    Complex64::new(
                        ((i * n + j) % 100) as f64 * 0.01,
                        ((i + j) % 100) as f64 * 0.01,
                    )
                };
            }
        }
        let a = Mat::from_slice(n, n, &a_data);

        let b_data: Vec<Complex64> = (0..n * n)
            .map(|i| Complex64::new((i % 100) as f64 * 0.01, ((i + 1) % 100) as f64 * 0.01))
            .collect();
        let b = Mat::from_slice(n, n, &b_data);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = trsm(
                    black_box(Side::Left),
                    black_box(Uplo::Upper),
                    black_box(Trans::NoTrans),
                    black_box(Diag::NonUnit),
                    black_box(Complex64::new(1.0, 0.0)),
                    black_box(a.as_ref()),
                    black_box(b.as_ref()),
                );
            });
        });
    }

    group.finish();
}

// =============================================================================
// Scaling Tests (Thread Count Comparison)
// =============================================================================

fn bench_gemm_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_scaling");

    // Use a large matrix to see scaling effects
    let n = 512;
    group.throughput(Throughput::Elements((n * n * n) as u64));

    let a_data: Vec<f64> = (0..n * n).map(|i| (i % 1000) as f64 * 0.001).collect();
    let a = Mat::from_slice(n, n, &a_data);
    let b_data: Vec<f64> = (0..n * n)
        .map(|i| ((i + 1) % 1000) as f64 * 0.001)
        .collect();
    let b = Mat::from_slice(n, n, &b_data);

    // Sequential
    group.bench_function("sequential", |bench| {
        let mut c_mat = Mat::zeros(n, n);
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

    group.finish();
}

criterion_group!(
    benches,
    // Parallel Level 1
    bench_parallel_dot,
    bench_parallel_axpy,
    bench_parallel_nrm2,
    bench_parallel_scal,
    // Strassen
    bench_strassen_vs_standard,
    // Asymmetric blocking
    bench_asymmetric_tall_thin,
    bench_asymmetric_short_wide,
    bench_asymmetric_inner_product,
    // TRSM
    bench_trsm_f64,
    bench_trsm_complex,
    // Scaling
    bench_gemm_scaling,
);
criterion_main!(benches);
