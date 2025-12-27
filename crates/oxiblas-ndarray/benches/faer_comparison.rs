//! Benchmarks comparing oxiblas-ndarray vs faer.
//!
//! This benchmark suite compares performance between:
//! - OxiBLAS (pure Rust BLAS/LAPACK implementation)
//! - faer (pure Rust linear algebra library)
//!
//! Both libraries are pure Rust without external BLAS/LAPACK dependencies.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use faer::Mat;
use faer::prelude::*;
use ndarray::{Array1, Array2, ShapeBuilder};
use oxiblas_ndarray::prelude::*;
use std::hint::black_box;

// =============================================================================
// Matrix Multiplication (GEMM) Benchmarks
// =============================================================================

fn bench_gemm_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_comparison");

    for size in [32, 64, 128, 256, 512].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n * n) as u64));

        // OxiBLAS: Create column-major arrays for best performance
        let a_ndarray: Array2<f64> =
            Array2::from_shape_fn((n, n).f(), |(i, j)| (i * n + j) as f64 * 0.001);
        let b_ndarray: Array2<f64> =
            Array2::from_shape_fn((n, n).f(), |(i, j)| (i + j) as f64 * 0.001);

        // faer: Create matrices
        let a_faer: Mat<f64> = Mat::from_fn(n, n, |i, j| (i * n + j) as f64 * 0.001);
        let b_faer: Mat<f64> = Mat::from_fn(n, n, |i, j| (i + j) as f64 * 0.001);

        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| black_box(matmul(&a_ndarray, &b_ndarray)));
        });

        group.bench_with_input(BenchmarkId::new("faer", size), size, |bench, _| {
            bench.iter(|| black_box(&a_faer * &b_faer));
        });
    }

    group.finish();
}

fn bench_gemm_rectangular(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_rectangular_comparison");

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

        // OxiBLAS
        let a_ndarray: Array2<f64> =
            Array2::from_shape_fn((m, k).f(), |(i, j)| (i * k + j) as f64 * 0.001);
        let b_ndarray: Array2<f64> =
            Array2::from_shape_fn((k, n).f(), |(i, j)| (i + j) as f64 * 0.001);

        // faer
        let a_faer: Mat<f64> = Mat::from_fn(m, k, |i, j| (i * k + j) as f64 * 0.001);
        let b_faer: Mat<f64> = Mat::from_fn(k, n, |i, j| (i + j) as f64 * 0.001);

        group.bench_with_input(BenchmarkId::new("oxiblas", &param), &param, |bench, _| {
            bench.iter(|| black_box(matmul(&a_ndarray, &b_ndarray)));
        });

        group.bench_with_input(BenchmarkId::new("faer", &param), &param, |bench, _| {
            bench.iter(|| black_box(&a_faer * &b_faer));
        });
    }

    group.finish();
}

// =============================================================================
// Matrix-Vector Multiplication Benchmarks
// =============================================================================

fn bench_matvec_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("matvec_comparison");

    for size in [64, 128, 256, 512, 1024].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        // OxiBLAS
        let a_ndarray: Array2<f64> =
            Array2::from_shape_fn((n, n).f(), |(i, j)| (i * n + j) as f64 * 0.001);
        let x_ndarray: Array1<f64> = Array1::from_vec((0..n).map(|i| i as f64 * 0.01).collect());

        // faer
        let a_faer: Mat<f64> = Mat::from_fn(n, n, |i, j| (i * n + j) as f64 * 0.001);
        let x_faer: Mat<f64> = Mat::from_fn(n, 1, |i, _| i as f64 * 0.01);

        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| black_box(matvec(&a_ndarray, &x_ndarray)));
        });

        group.bench_with_input(BenchmarkId::new("faer", size), size, |bench, _| {
            bench.iter(|| black_box(&a_faer * &x_faer));
        });
    }

    group.finish();
}

// =============================================================================
// QR Decomposition Benchmarks
// =============================================================================

fn bench_qr_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("qr_decomposition");
    group.sample_size(50); // Fewer samples for expensive operations

    for size in [32, 64, 128, 256].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        // OxiBLAS: well-conditioned matrix
        let a_ndarray: Array2<f64> = Array2::from_shape_fn((n, n).f(), |(i, j)| {
            if i == j {
                n as f64 + 1.0
            } else {
                ((i + j) % 10) as f64 * 0.1
            }
        });

        // faer
        let a_faer: Mat<f64> = Mat::from_fn(n, n, |i, j| {
            if i == j {
                n as f64 + 1.0
            } else {
                ((i + j) % 10) as f64 * 0.1
            }
        });

        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| black_box(qr_ndarray(&a_ndarray)));
        });

        group.bench_with_input(BenchmarkId::new("faer", size), size, |bench, _| {
            bench.iter(|| black_box(a_faer.qr()));
        });
    }

    group.finish();
}

// =============================================================================
// SVD Benchmarks
// =============================================================================

fn bench_svd_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd_decomposition");
    group.sample_size(30); // SVD is expensive

    for size in [32, 64, 128].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        // OxiBLAS: well-conditioned matrix
        let a_ndarray: Array2<f64> = Array2::from_shape_fn((n, n).f(), |(i, j)| {
            if i == j {
                10.0 + i as f64
            } else {
                ((i * j) % 7) as f64 * 0.1
            }
        });

        // faer
        let a_faer: Mat<f64> = Mat::from_fn(n, n, |i, j| {
            if i == j {
                10.0 + i as f64
            } else {
                ((i * j) % 7) as f64 * 0.1
            }
        });

        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| black_box(svd_ndarray(&a_ndarray)));
        });

        group.bench_with_input(BenchmarkId::new("faer", size), size, |bench, _| {
            bench.iter(|| black_box(a_faer.svd()));
        });
    }

    group.finish();
}

fn bench_thin_svd_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("thin_svd_decomposition");
    group.sample_size(30);

    // Rectangular matrices (more rows than columns)
    for &(m, n) in [(128, 64), (256, 64), (512, 128)].iter() {
        let param = format!("{}x{}", m, n);
        group.throughput(Throughput::Elements((m * n) as u64));

        // OxiBLAS
        let a_ndarray: Array2<f64> = Array2::from_shape_fn((m, n).f(), |(i, j)| {
            if i == j {
                10.0 + i as f64
            } else {
                ((i * j) % 5) as f64 * 0.1
            }
        });

        // faer
        let a_faer: Mat<f64> = Mat::from_fn(m, n, |i, j| {
            if i == j {
                10.0 + i as f64
            } else {
                ((i * j) % 5) as f64 * 0.1
            }
        });

        group.bench_with_input(BenchmarkId::new("oxiblas", &param), &param, |bench, _| {
            bench.iter(|| black_box(svd_ndarray(&a_ndarray)));
        });

        group.bench_with_input(BenchmarkId::new("faer_thin", &param), &param, |bench, _| {
            bench.iter(|| black_box(a_faer.thin_svd()));
        });
    }

    group.finish();
}

// =============================================================================
// Eigenvalue Decomposition Benchmarks
// =============================================================================

fn bench_symmetric_eig_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("symmetric_eigenvalue");
    group.sample_size(30);

    for size in [32, 64, 128].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        // OxiBLAS: Create symmetric positive definite matrix
        let a_ndarray: Array2<f64> = {
            let mut a = Array2::from_shape_fn((n, n).f(), |(i, j)| {
                if i == j {
                    n as f64 + 10.0
                } else if i < j {
                    ((i + j) % 5) as f64 * 0.1
                } else {
                    0.0
                }
            });
            // Make symmetric
            for i in 0..n {
                for j in (i + 1)..n {
                    a[[j, i]] = a[[i, j]];
                }
            }
            a
        };

        // faer
        let a_faer: Mat<f64> = Mat::from_fn(n, n, |i, j| {
            if i == j {
                n as f64 + 10.0
            } else {
                ((i.min(j) + i.max(j)) % 5) as f64 * 0.1
            }
        });

        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| black_box(eig_symmetric(&a_ndarray)));
        });

        group.bench_with_input(BenchmarkId::new("faer", size), size, |bench, _| {
            bench.iter(|| black_box(a_faer.self_adjoint_eigen(faer::Side::Lower)));
        });
    }

    group.finish();
}

// =============================================================================
// Linear System Solve Benchmarks
// =============================================================================

fn bench_solve_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_solve");
    group.sample_size(50);

    for size in [32, 64, 128, 256].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        // OxiBLAS: Create well-conditioned matrix
        let a_ndarray: Array2<f64> = Array2::from_shape_fn((n, n).f(), |(i, j)| {
            if i == j {
                n as f64 + 5.0
            } else {
                ((i + j) % 7) as f64 * 0.1
            }
        });
        let b_ndarray: Array1<f64> = Array1::from_vec((0..n).map(|i| (i + 1) as f64).collect());

        // faer
        let a_faer: Mat<f64> = Mat::from_fn(n, n, |i, j| {
            if i == j {
                n as f64 + 5.0
            } else {
                ((i + j) % 7) as f64 * 0.1
            }
        });
        let b_faer: Mat<f64> = Mat::from_fn(n, 1, |i, _| (i + 1) as f64);

        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| black_box(solve_ndarray(&a_ndarray, &b_ndarray)));
        });

        group.bench_with_input(BenchmarkId::new("faer_lu", size), size, |bench, _| {
            bench.iter(|| {
                let lu = a_faer.partial_piv_lu();
                black_box(lu.solve(&b_faer))
            });
        });
    }

    group.finish();
}

fn bench_solve_multiple_rhs(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve_multiple_rhs");
    group.sample_size(50);

    for &(n, nrhs) in [(64, 10), (128, 10), (256, 20)].iter() {
        let param = format!("{}x{}_rhs{}", n, n, nrhs);
        group.throughput(Throughput::Elements((n * n + n * nrhs) as u64));

        // OxiBLAS
        let a_ndarray: Array2<f64> = Array2::from_shape_fn((n, n).f(), |(i, j)| {
            if i == j {
                n as f64 + 5.0
            } else {
                ((i + j) % 7) as f64 * 0.1
            }
        });
        let b_ndarray: Array2<f64> =
            Array2::from_shape_fn((n, nrhs).f(), |(i, j)| (i * nrhs + j + 1) as f64);

        // faer
        let a_faer: Mat<f64> = Mat::from_fn(n, n, |i, j| {
            if i == j {
                n as f64 + 5.0
            } else {
                ((i + j) % 7) as f64 * 0.1
            }
        });
        let b_faer: Mat<f64> = Mat::from_fn(n, nrhs, |i, j| (i * nrhs + j + 1) as f64);

        group.bench_with_input(BenchmarkId::new("oxiblas", &param), &param, |bench, _| {
            bench.iter(|| black_box(solve_multiple_ndarray(&a_ndarray, &b_ndarray)));
        });

        group.bench_with_input(BenchmarkId::new("faer_lu", &param), &param, |bench, _| {
            bench.iter(|| {
                let lu = a_faer.partial_piv_lu();
                black_box(lu.solve(&b_faer))
            });
        });
    }

    group.finish();
}

// =============================================================================
// Cholesky Decomposition Benchmarks
// =============================================================================

fn bench_cholesky_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("cholesky_decomposition");
    group.sample_size(50);

    for size in [32, 64, 128, 256].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        // Create symmetric positive definite matrix
        // A = B * B^T + n * I (guarantees positive definiteness)
        let a_ndarray: Array2<f64> = {
            let mut a = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                for j in 0..=i {
                    let val: f64 = (0..n)
                        .map(|k| ((i + k) % 7 + 1) as f64 * ((j + k) % 5 + 1) as f64 * 0.01)
                        .sum();
                    a[[i, j]] = val;
                    a[[j, i]] = val;
                }
                a[[i, i]] += n as f64 + 10.0; // Ensure positive definiteness
            }
            // Ensure column-major for oxiblas
            Array2::from_shape_fn((n, n).f(), |(i, j)| a[[i, j]])
        };

        let a_faer: Mat<f64> = Mat::from_fn(n, n, |i, j| {
            let base: f64 = (0..n)
                .map(|k| ((i + k) % 7 + 1) as f64 * ((j + k) % 5 + 1) as f64 * 0.01)
                .sum();
            if i == j { base + n as f64 + 10.0 } else { base }
        });

        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| black_box(cholesky_ndarray(&a_ndarray)));
        });

        group.bench_with_input(BenchmarkId::new("faer", size), size, |bench, _| {
            bench.iter(|| black_box(a_faer.llt(faer::Side::Lower)));
        });
    }

    group.finish();
}

// =============================================================================
// LU Decomposition Benchmarks
// =============================================================================

fn bench_lu_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("lu_decomposition");
    group.sample_size(50);

    for size in [32, 64, 128, 256].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        // Well-conditioned matrix
        let a_ndarray: Array2<f64> = Array2::from_shape_fn((n, n).f(), |(i, j)| {
            if i == j {
                n as f64 + 5.0
            } else {
                ((i * j + 1) % 9) as f64 * 0.1
            }
        });

        let a_faer: Mat<f64> = Mat::from_fn(n, n, |i, j| {
            if i == j {
                n as f64 + 5.0
            } else {
                ((i * j + 1) % 9) as f64 * 0.1
            }
        });

        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| black_box(lu_ndarray(&a_ndarray)));
        });

        group.bench_with_input(
            BenchmarkId::new("faer_partial_piv", size),
            size,
            |bench, _| {
                bench.iter(|| black_box(a_faer.partial_piv_lu()));
            },
        );

        group.bench_with_input(BenchmarkId::new("faer_full_piv", size), size, |bench, _| {
            bench.iter(|| black_box(a_faer.full_piv_lu()));
        });
    }

    group.finish();
}

// =============================================================================
// Matrix Inverse Benchmarks
// =============================================================================

fn bench_inverse_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_inverse");
    group.sample_size(50);

    for size in [32, 64, 128, 256].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        // Well-conditioned matrix
        let a_ndarray: Array2<f64> = Array2::from_shape_fn((n, n).f(), |(i, j)| {
            if i == j {
                n as f64 + 10.0
            } else {
                ((i + j) % 5) as f64 * 0.05
            }
        });

        let a_faer: Mat<f64> = Mat::from_fn(n, n, |i, j| {
            if i == j {
                n as f64 + 10.0
            } else {
                ((i + j) % 5) as f64 * 0.05
            }
        });

        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| black_box(inv_ndarray(&a_ndarray)));
        });

        group.bench_with_input(BenchmarkId::new("faer_lu_solve", size), size, |bench, _| {
            let identity: Mat<f64> = Mat::identity(n, n);
            bench.iter(|| {
                let lu = a_faer.partial_piv_lu();
                black_box(lu.solve(&identity))
            });
        });
    }

    group.finish();
}

// =============================================================================
// Vector Norms Benchmarks
// =============================================================================

fn bench_norm_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_norm");

    for size in [1000, 10000, 100000, 1000000].iter() {
        let n = *size;
        group.throughput(Throughput::Elements(n as u64));

        // OxiBLAS
        let x_ndarray: Array1<f64> = Array1::from_vec((0..n).map(|i| (i as f64).sin()).collect());

        // faer (column vector)
        let x_faer: Mat<f64> = Mat::from_fn(n, 1, |i, _| (i as f64).sin());

        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| black_box(nrm2_ndarray(&x_ndarray)));
        });

        group.bench_with_input(BenchmarkId::new("faer", size), size, |bench, _| {
            bench.iter(|| black_box(x_faer.norm_l2()));
        });
    }

    group.finish();
}

fn bench_matrix_frobenius_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("frobenius_norm_comparison");

    for size in [64, 128, 256, 512, 1024].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        // OxiBLAS
        let a_ndarray: Array2<f64> =
            Array2::from_shape_fn((n, n).f(), |(i, j)| ((i * n + j) as f64).sin());

        // faer
        let a_faer: Mat<f64> = Mat::from_fn(n, n, |i, j| ((i * n + j) as f64).sin());

        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| black_box(frobenius_norm(&a_ndarray)));
        });

        group.bench_with_input(BenchmarkId::new("faer", size), size, |bench, _| {
            bench.iter(|| black_box(a_faer.norm_l2()));
        });
    }

    group.finish();
}

// =============================================================================
// Determinant Benchmarks
// =============================================================================

fn bench_determinant_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("determinant");
    group.sample_size(50);

    for size in [32, 64, 128].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        // Well-conditioned matrix
        let a_ndarray: Array2<f64> = Array2::from_shape_fn((n, n).f(), |(i, j)| {
            if i == j {
                2.0 + (i as f64) * 0.01
            } else {
                ((i + j) % 5) as f64 * 0.01
            }
        });

        let a_faer: Mat<f64> = Mat::from_fn(n, n, |i, j| {
            if i == j {
                2.0 + (i as f64) * 0.01
            } else {
                ((i + j) % 5) as f64 * 0.01
            }
        });

        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| black_box(det_ndarray(&a_ndarray)));
        });

        // Note: faer doesn't have a direct determinant method
        // LU decomposition is typically used for determinant calculation
        group.bench_with_input(BenchmarkId::new("faer_lu", size), size, |bench, _| {
            bench.iter(|| black_box(a_faer.partial_piv_lu()));
        });
    }

    group.finish();
}

// =============================================================================
// Randomized SVD Benchmarks (oxiblas-specific feature)
// =============================================================================

fn bench_rsvd(c: &mut Criterion) {
    let mut group = c.benchmark_group("randomized_svd");
    group.sample_size(20);

    for &(m, n, k) in [(512, 256, 50), (1024, 512, 100), (2048, 512, 50)].iter() {
        let param = format!("{}x{}_k{}", m, n, k);
        group.throughput(Throughput::Elements((m * n) as u64));

        // Low-rank matrix for rSVD
        let a_ndarray: Array2<f64> = Array2::from_shape_fn((m, n).f(), |(i, j)| {
            // Create matrix with decaying singular values
            let mut sum = 0.0;
            for r in 0..k.min(20) {
                sum += (1.0 / ((r + 1) as f64).sqrt())
                    * ((i as f64 * r as f64 * 0.1).sin())
                    * ((j as f64 * r as f64 * 0.1).cos());
            }
            sum
        });

        let a_faer: Mat<f64> = Mat::from_fn(m, n, |i, j| {
            let mut sum = 0.0;
            for r in 0..k.min(20) {
                sum += (1.0 / ((r + 1) as f64).sqrt())
                    * ((i as f64 * r as f64 * 0.1).sin())
                    * ((j as f64 * r as f64 * 0.1).cos());
            }
            sum
        });

        group.bench_with_input(
            BenchmarkId::new("oxiblas_rsvd", &param),
            &param,
            |bench, _| {
                bench.iter(|| black_box(rsvd_ndarray(&a_ndarray, k)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("faer_full_svd", &param),
            &param,
            |bench, _| {
                bench.iter(|| black_box(a_faer.thin_svd()));
            },
        );
    }

    group.finish();
}

// =============================================================================
// Criterion Groups
// =============================================================================

criterion_group!(
    benches,
    // BLAS operations
    bench_gemm_comparison,
    bench_gemm_rectangular,
    bench_matvec_comparison,
    bench_norm_comparison,
    bench_matrix_frobenius_norm,
    // LAPACK decompositions
    bench_qr_comparison,
    bench_svd_comparison,
    bench_thin_svd_comparison,
    bench_symmetric_eig_comparison,
    bench_lu_comparison,
    bench_cholesky_comparison,
    // Linear systems
    bench_solve_comparison,
    bench_solve_multiple_rhs,
    // Other operations
    bench_inverse_comparison,
    bench_determinant_comparison,
    bench_rsvd,
);

criterion_main!(benches);
