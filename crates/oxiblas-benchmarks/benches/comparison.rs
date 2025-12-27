//! Comparison benchmarks: OxiBLAS vs OpenBLAS.
//!
//! This benchmark suite directly compares OxiBLAS performance against OpenBLAS
//! for common BLAS operations. Run with:
//!
//! ```bash
//! cargo bench --package oxiblas-benchmarks --bench comparison --features compare-openblas
//! ```
//!
//! ## Benchmarks Included
//!
//! ### Level 1 (Vector Operations)
//! - DDOT (f64 dot product)
//! - DAXPY (f64 vector addition)
//! - DNRM2 (f64 vector norm)
//!
//! ### Level 2 (Matrix-Vector Operations)
//! - DGEMV (f64 matrix-vector multiplication)
//!
//! ### Level 3 (Matrix-Matrix Operations)
//! - DGEMM (f64 matrix multiplication) - Square and rectangular matrices
//! - SGEMM (f32 matrix multiplication)
//! - DTRSM (f64 triangular solve with multiple RHS)
//! - DSYRK (f64 symmetric rank-k update)
//!
//! ## Results Interpretation
//!
//! Criterion will show performance comparison as:
//! - Time: lower is better
//! - Throughput: higher is better
//! - Performance ratio: OxiBLAS time / OpenBLAS time
//!   - Ratio < 1.0: OxiBLAS is faster
//!   - Ratio > 1.0: OpenBLAS is faster
//!   - Ratio ≈ 1.0: Similar performance

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

// OxiBLAS imports
use oxiblas_blas::level1::{axpy, dot, nrm2};
use oxiblas_blas::level2::{GemvTrans, gemv};
use oxiblas_blas::level3::{Diag, Side, Trans, Uplo, gemm, syrk, trsm_in_place};
use oxiblas_matrix::Mat;

// OpenBLAS imports via ndarray
use ndarray::{Array1, Array2};

/// Benchmark DGEMM: C = alpha * A * B + beta * C
fn bench_gemm_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_comparison");

    for size in [64, 128, 256, 512, 1024].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n * n) as u64));

        // OxiBLAS setup
        let a_data: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.01).collect();
        let b_data: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.01).collect();
        let a_oxi = Mat::from_slice(n, n, &a_data);
        let b_oxi = Mat::from_slice(n, n, &b_data);
        let mut c_oxi = Mat::zeros(n, n);

        // OpenBLAS setup via ndarray
        let a_ndarray = Array2::from_shape_vec((n, n), a_data.clone()).unwrap();
        let b_ndarray = Array2::from_shape_vec((n, n), b_data.clone()).unwrap();

        // Benchmark OxiBLAS
        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| {
                gemm(
                    black_box(1.0),
                    black_box(a_oxi.as_ref()),
                    black_box(b_oxi.as_ref()),
                    black_box(0.0),
                    black_box(c_oxi.as_mut()),
                );
            });
        });

        // Benchmark OpenBLAS (via ndarray)
        group.bench_with_input(BenchmarkId::new("openblas", size), size, |bench, _| {
            bench.iter(|| {
                let _result = black_box(&a_ndarray).dot(black_box(&b_ndarray));
            });
        });
    }

    group.finish();
}

/// Benchmark DGEMM with rectangular matrices
fn bench_gemm_rectangular_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_rectangular_comparison");

    for &(m, k, n) in [(128, 256, 128), (256, 512, 256), (512, 256, 512)].iter() {
        let param = format!("{}x{}x{}", m, k, n);
        group.throughput(Throughput::Elements((m * k * n) as u64));

        // OxiBLAS setup
        let a_data: Vec<f64> = (0..m * k).map(|i| (i as f64) * 0.01).collect();
        let b_data: Vec<f64> = (0..k * n).map(|i| (i as f64) * 0.01).collect();
        let a_oxi = Mat::from_slice(m, k, &a_data);
        let b_oxi = Mat::from_slice(k, n, &b_data);
        let mut c_oxi = Mat::zeros(m, n);

        // OpenBLAS setup
        let a_ndarray = Array2::from_shape_vec((m, k), a_data.clone()).unwrap();
        let b_ndarray = Array2::from_shape_vec((k, n), b_data.clone()).unwrap();

        // Benchmark OxiBLAS
        group.bench_with_input(BenchmarkId::new("oxiblas", &param), &param, |bench, _| {
            bench.iter(|| {
                gemm(
                    black_box(1.0),
                    black_box(a_oxi.as_ref()),
                    black_box(b_oxi.as_ref()),
                    black_box(0.0),
                    black_box(c_oxi.as_mut()),
                );
            });
        });

        // Benchmark OpenBLAS
        group.bench_with_input(BenchmarkId::new("openblas", &param), &param, |bench, _| {
            bench.iter(|| {
                let _result = black_box(&a_ndarray).dot(black_box(&b_ndarray));
            });
        });
    }

    group.finish();
}

/// Benchmark DGEMV: y = alpha * A * x + beta * y
fn bench_gemv_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemv_comparison");

    for size in [100, 500, 1000, 5000, 10000].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        // OxiBLAS setup
        let a_data: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.001).collect();
        let x_data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();
        let a_oxi = Mat::from_slice(n, n, &a_data);
        let x_oxi = x_data.clone();
        let mut y_oxi = vec![0.0; n];

        // OpenBLAS setup
        let a_ndarray = Array2::from_shape_vec((n, n), a_data.clone()).unwrap();
        let x_ndarray = Array1::from_vec(x_data.clone());

        // Benchmark OxiBLAS
        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| {
                gemv(
                    black_box(GemvTrans::NoTrans),
                    black_box(1.0),
                    black_box(a_oxi.as_ref()),
                    black_box(&x_oxi),
                    black_box(0.0),
                    black_box(&mut y_oxi),
                );
            });
        });

        // Benchmark OpenBLAS
        group.bench_with_input(BenchmarkId::new("openblas", size), size, |bench, _| {
            bench.iter(|| {
                let _result = black_box(&a_ndarray).dot(black_box(&x_ndarray));
            });
        });
    }

    group.finish();
}

/// Benchmark DDOT: dot product of two vectors
fn bench_dot_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_comparison");

    for size in [1000, 10000, 100000, 1000000].iter() {
        let n = *size;
        group.throughput(Throughput::Elements(n as u64));

        let x_data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        let y_data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.02).collect();

        // OxiBLAS
        let x_oxi = x_data.clone();
        let y_oxi = y_data.clone();

        // OpenBLAS
        let x_ndarray = Array1::from_vec(x_data.clone());
        let y_ndarray = Array1::from_vec(y_data.clone());

        // Benchmark OxiBLAS
        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| {
                let _result = dot(black_box(&x_oxi), black_box(&y_oxi));
            });
        });

        // Benchmark OpenBLAS
        group.bench_with_input(BenchmarkId::new("openblas", size), size, |bench, _| {
            bench.iter(|| {
                let _result = black_box(&x_ndarray).dot(black_box(&y_ndarray));
            });
        });
    }

    group.finish();
}

/// Benchmark DAXPY: y = alpha * x + y
fn bench_axpy_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("axpy_comparison");

    for size in [1000, 10000, 100000, 1000000].iter() {
        let n = *size;
        group.throughput(Throughput::Elements(n as u64));

        let x_data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        let y_data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.02).collect();

        // OxiBLAS
        let x_oxi = x_data.clone();
        let mut y_oxi = y_data.clone();

        // OpenBLAS
        let x_ndarray = Array1::from_vec(x_data.clone());
        let mut y_ndarray = Array1::from_vec(y_data.clone());

        // Benchmark OxiBLAS
        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| {
                axpy(black_box(2.5), black_box(&x_oxi), black_box(&mut y_oxi));
            });
        });

        // Benchmark OpenBLAS
        group.bench_with_input(BenchmarkId::new("openblas", size), size, |bench, _| {
            bench.iter(|| {
                y_ndarray.scaled_add(black_box(2.5), black_box(&x_ndarray));
            });
        });
    }

    group.finish();
}

/// Benchmark DNRM2: Euclidean norm of a vector
fn bench_nrm2_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("nrm2_comparison");

    for size in [1000, 10000, 100000, 1000000].iter() {
        let n = *size;
        group.throughput(Throughput::Elements(n as u64));

        let x_data: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();

        // OxiBLAS
        let x_oxi = x_data.clone();

        // OpenBLAS
        let x_ndarray = Array1::from_vec(x_data.clone());

        // Benchmark OxiBLAS
        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| {
                let _result = nrm2(black_box(&x_oxi));
            });
        });

        // Benchmark OpenBLAS
        group.bench_with_input(BenchmarkId::new("openblas", size), size, |bench, _| {
            bench.iter(|| {
                // Use BLAS dnrm2 via cblas-sys directly
                let _result = unsafe {
                    cblas_sys::cblas_dnrm2(x_ndarray.len() as i32, x_ndarray.as_ptr(), 1)
                };
                black_box(_result);
            });
        });
    }

    group.finish();
}

/// Benchmark SGEMM (single precision): C = alpha * A * B + beta * C
fn bench_sgemm_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("sgemm_comparison");

    for size in [64, 128, 256, 512, 1024].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n * n) as u64));

        // OxiBLAS setup
        let a_data: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.01).collect();
        let a_oxi: Mat<f32> = Mat::from_slice(n, n, &a_data);
        let b_oxi: Mat<f32> = Mat::from_slice(n, n, &b_data);
        let mut c_oxi: Mat<f32> = Mat::zeros(n, n);

        // OpenBLAS setup via ndarray
        let a_ndarray = Array2::<f32>::from_shape_vec((n, n), a_data.clone()).unwrap();
        let b_ndarray = Array2::<f32>::from_shape_vec((n, n), b_data.clone()).unwrap();

        // Benchmark OxiBLAS
        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| {
                gemm(
                    black_box(1.0f32),
                    black_box(a_oxi.as_ref()),
                    black_box(b_oxi.as_ref()),
                    black_box(0.0f32),
                    black_box(c_oxi.as_mut()),
                );
            });
        });

        // Benchmark OpenBLAS (via ndarray)
        group.bench_with_input(BenchmarkId::new("openblas", size), size, |bench, _| {
            bench.iter(|| {
                let _result = black_box(&a_ndarray).dot(black_box(&b_ndarray));
            });
        });
    }

    group.finish();
}

/// Benchmark DTRSM: Solve op(A) * X = alpha * B for X
fn bench_trsm_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("trsm_comparison");

    for size in [64, 128, 256, 512].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n * n) as u64));

        // Create a lower triangular matrix with positive diagonal (for stability)
        let a_data: Vec<f64> = (0..n * n)
            .map(|idx| {
                let i = idx % n;
                let j = idx / n;
                if i >= j {
                    if i == j { (i + 1) as f64 } else { 0.1 }
                } else {
                    0.0
                }
            })
            .collect();
        let b_data: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.01 + 1.0).collect();

        // OxiBLAS setup
        let a_oxi = Mat::from_slice(n, n, &a_data);
        let b_oxi = Mat::from_slice(n, n, &b_data);
        let mut x_oxi = Mat::from_slice(n, n, &b_data);

        // Benchmark OxiBLAS
        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| {
                // Reset x to b
                for j in 0..n {
                    for i in 0..n {
                        x_oxi[(i, j)] = b_oxi[(i, j)];
                    }
                }
                let _ = trsm_in_place(
                    black_box(Side::Left),
                    black_box(Uplo::Lower),
                    black_box(Trans::NoTrans),
                    black_box(Diag::NonUnit),
                    black_box(a_oxi.as_ref()),
                    black_box(x_oxi.as_mut()),
                );
            });
        });

        // Benchmark OpenBLAS (via cblas_sys)
        group.bench_with_input(BenchmarkId::new("openblas", size), size, |bench, _| {
            let mut x_blas = b_data.clone();
            bench.iter(|| {
                // Reset x to b
                x_blas.copy_from_slice(&b_data);
                unsafe {
                    cblas_sys::cblas_dtrsm(
                        cblas_sys::CBLAS_LAYOUT::CblasColMajor,
                        cblas_sys::CBLAS_SIDE::CblasLeft,
                        cblas_sys::CBLAS_UPLO::CblasLower,
                        cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                        cblas_sys::CBLAS_DIAG::CblasNonUnit,
                        n as i32,
                        n as i32,
                        1.0,
                        a_data.as_ptr(),
                        n as i32,
                        x_blas.as_mut_ptr(),
                        n as i32,
                    );
                }
            });
        });
    }

    group.finish();
}

/// Benchmark DSYRK: C = alpha * A * A^T + beta * C (symmetric rank-k update)
fn bench_syrk_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("syrk_comparison");

    for size in [64, 128, 256, 512].iter() {
        let n = *size;
        let k = *size;
        group.throughput(Throughput::Elements((n * n * k) as u64));

        let a_data: Vec<f64> = (0..n * k).map(|i| (i as f64) * 0.01).collect();
        let c_data: Vec<f64> = vec![0.0; n * n];

        // OxiBLAS setup
        let a_oxi = Mat::from_slice(n, k, &a_data);
        let mut c_oxi = Mat::from_slice(n, n, &c_data);

        // Benchmark OxiBLAS
        group.bench_with_input(BenchmarkId::new("oxiblas", size), size, |bench, _| {
            bench.iter(|| {
                // Reset C to zero
                for j in 0..n {
                    for i in 0..n {
                        c_oxi[(i, j)] = 0.0;
                    }
                }
                let _ = syrk(
                    black_box(Uplo::Lower),
                    black_box(Trans::NoTrans),
                    black_box(1.0),
                    black_box(a_oxi.as_ref()),
                    black_box(0.0),
                    black_box(c_oxi.as_mut()),
                );
            });
        });

        // Benchmark OpenBLAS (via cblas_sys)
        group.bench_with_input(BenchmarkId::new("openblas", size), size, |bench, _| {
            let mut c_blas = c_data.clone();
            bench.iter(|| {
                // Reset C to zero
                c_blas.fill(0.0);
                unsafe {
                    cblas_sys::cblas_dsyrk(
                        cblas_sys::CBLAS_LAYOUT::CblasColMajor,
                        cblas_sys::CBLAS_UPLO::CblasLower,
                        cblas_sys::CBLAS_TRANSPOSE::CblasNoTrans,
                        n as i32,
                        k as i32,
                        1.0,
                        a_data.as_ptr(),
                        n as i32,
                        0.0,
                        c_blas.as_mut_ptr(),
                        n as i32,
                    );
                }
            });
        });
    }

    group.finish();
}

criterion_group!(
    comparison_benches,
    bench_gemm_comparison,
    bench_gemm_rectangular_comparison,
    bench_sgemm_comparison,
    bench_trsm_comparison,
    bench_syrk_comparison,
    bench_gemv_comparison,
    bench_dot_comparison,
    bench_axpy_comparison,
    bench_nrm2_comparison,
);

criterion_main!(comparison_benches);
