//! Benchmarks for BLAS Level 3 operations (matrix-matrix).

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use num_complex::Complex64;
use oxiblas_blas::level3::{
    Diag, Side, Trans, TrmmDiag, TrmmSide, TrmmTrans, TrmmUplo, Uplo, gemm, gemm3m_c64, hemm_c64,
    her2k, herk, symm, syr2k, syrk, trmm, trsm,
};
use oxiblas_matrix::Mat;
use std::hint::black_box;

fn bench_gemm_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_f64");

    for size in [32, 64, 128, 256, 512].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n * n) as u64));

        let a_data: Vec<f64> = (0..n * n).map(|i| i as f64).collect();
        let a = Mat::from_slice(n, n, &a_data);
        let b_data: Vec<f64> = (0..n * n).map(|i| i as f64).collect();
        let b = Mat::from_slice(n, n, &b_data);
        let mut c = Mat::zeros(n, n);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                gemm(
                    black_box(1.0),
                    black_box(a.as_ref()),
                    black_box(b.as_ref()),
                    black_box(0.0),
                    black_box(c.as_mut()),
                );
            });
        });
    }

    group.finish();
}

fn bench_gemm_rectangular(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm_rectangular");

    for &(m, k, n) in [(64, 128, 64), (128, 64, 128), (256, 128, 64)].iter() {
        let param = format!("{}x{}x{}", m, k, n);
        group.throughput(Throughput::Elements((m * k * n) as u64));

        let a_data: Vec<f64> = (0..m * k).map(|i| i as f64).collect();
        let a = Mat::from_slice(m, k, &a_data);
        let b_data: Vec<f64> = (0..k * n).map(|i| i as f64).collect();
        let b = Mat::from_slice(k, n, &b_data);
        let mut c = Mat::zeros(m, n);

        group.bench_with_input(BenchmarkId::from_parameter(&param), &param, |bench, _| {
            bench.iter(|| {
                gemm(
                    black_box(1.0),
                    black_box(a.as_ref()),
                    black_box(b.as_ref()),
                    black_box(0.0),
                    black_box(c.as_mut()),
                );
            });
        });
    }

    group.finish();
}

fn bench_gemm3m_complex(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemm3m_complex");

    for size in [32, 64, 128, 256].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n * n) as u64));

        let a_data: Vec<Complex64> = (0..n * n).map(|i| Complex64::new(i as f64, 1.0)).collect();
        let a = Mat::from_slice(n, n, &a_data);
        let b_data: Vec<Complex64> = (0..n * n).map(|i| Complex64::new(i as f64, 0.5)).collect();
        let b = Mat::from_slice(n, n, &b_data);
        let mut c = Mat::zeros(n, n);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                gemm3m_c64(
                    black_box(Complex64::new(1.0, 0.0)),
                    black_box(a.as_ref()),
                    black_box(b.as_ref()),
                    black_box(Complex64::new(0.0, 0.0)),
                    black_box(c.as_mut()),
                );
            });
        });
    }

    group.finish();
}

fn bench_trmm(c: &mut Criterion) {
    let mut group = c.benchmark_group("trmm");

    for size in [64, 128, 256, 512].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        // Upper triangular matrix
        let mut a_data = vec![0.0; n * n];
        for i in 0..n {
            for j in i..n {
                a_data[i * n + j] = (i * n + j) as f64;
            }
        }
        let a = Mat::from_slice(n, n, &a_data);

        let b_data: Vec<f64> = (0..n * n).map(|i| i as f64).collect();
        let b = Mat::from_slice(n, n, &b_data);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = trmm(
                    black_box(TrmmSide::Left),
                    black_box(TrmmUplo::Upper),
                    black_box(TrmmTrans::NoTrans),
                    black_box(TrmmDiag::NonUnit),
                    black_box(1.0),
                    black_box(a.as_ref()),
                    black_box(b.as_ref()),
                )
                .unwrap();
            });
        });
    }

    group.finish();
}

fn bench_syrk(c: &mut Criterion) {
    let mut group = c.benchmark_group("syrk_f64");

    for size in [64, 128, 256, 512].iter() {
        let n = *size;
        let k = n / 2;
        // SYRK: C = alpha * A * A^T + beta * C
        // FLOPs: n^2 * k (symmetric rank-k update)
        group.throughput(Throughput::Elements((n * n * k) as u64));

        let a_data: Vec<f64> = (0..n * k).map(|i| (i % 100) as f64 * 0.01).collect();
        let a = Mat::from_slice(n, k, &a_data);
        let mut c = Mat::zeros(n, n);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = syrk(
                    black_box(Uplo::Lower),
                    black_box(Trans::NoTrans),
                    black_box(1.0),
                    black_box(a.as_ref()),
                    black_box(0.0),
                    black_box(c.as_mut()),
                );
            });
        });
    }

    group.finish();
}

fn bench_trsm(c: &mut Criterion) {
    let mut group = c.benchmark_group("trsm_f64");

    for size in [64, 128, 256, 512].iter() {
        let n = *size;
        // TRSM: solve A * X = B or X * A = B for X
        // FLOPs: n^2 * n (triangular solve with n RHS columns)
        group.throughput(Throughput::Elements((n * n * n) as u64));

        // Create a lower triangular matrix with non-zero diagonal
        let mut a_data = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                a_data[i * n + j] = if i == j { (i + 1) as f64 } else { 0.5 };
            }
        }
        let a = Mat::from_slice(n, n, &a_data);

        let b_data: Vec<f64> = (0..n * n).map(|i| (i % 100) as f64 * 0.01).collect();
        let b = Mat::from_slice(n, n, &b_data);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = trsm(
                    black_box(Side::Left),
                    black_box(Uplo::Lower),
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

fn bench_syr2k(c: &mut Criterion) {
    let mut group = c.benchmark_group("syr2k_f64");

    for size in [64, 128, 256, 512].iter() {
        let n = *size;
        let k = n / 2;
        // SYR2K: C = alpha * A * B^T + alpha * B * A^T + beta * C
        // FLOPs: 2 * n^2 * k (symmetric rank-2k update)
        group.throughput(Throughput::Elements((2 * n * n * k) as u64));

        let a_data: Vec<f64> = (0..n * k).map(|i| (i % 100) as f64 * 0.01).collect();
        let a = Mat::from_slice(n, k, &a_data);
        let b_data: Vec<f64> = (0..n * k).map(|i| ((i + 50) % 100) as f64 * 0.01).collect();
        let b = Mat::from_slice(n, k, &b_data);
        let mut c_mat = Mat::zeros(n, n);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = syr2k(
                    black_box(Uplo::Lower),
                    black_box(Trans::NoTrans),
                    black_box(1.0),
                    black_box(a.as_ref()),
                    black_box(b.as_ref()),
                    black_box(0.0),
                    black_box(c_mat.as_mut()),
                );
            });
        });
    }

    group.finish();
}

fn bench_symm(c: &mut Criterion) {
    let mut group = c.benchmark_group("symm_f64");

    for size in [64, 128, 256, 512].iter() {
        let n = *size;
        // SYMM: C = alpha * A * B + beta * C (A is symmetric)
        // FLOPs: 2 * n^2 * n = 2n^3
        group.throughput(Throughput::Elements((2 * n * n * n) as u64));

        // Create symmetric matrix A
        let mut a_data = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let val = ((i * n + j) % 100) as f64 * 0.01;
                a_data[i * n + j] = val;
                a_data[j * n + i] = val;
            }
        }
        let a = Mat::from_slice(n, n, &a_data);

        let b_data: Vec<f64> = (0..n * n).map(|i| (i % 100) as f64 * 0.01).collect();
        let b = Mat::from_slice(n, n, &b_data);
        let mut c_mat = Mat::zeros(n, n);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = symm(
                    black_box(Side::Left),
                    black_box(Uplo::Lower),
                    black_box(1.0),
                    black_box(a.as_ref()),
                    black_box(b.as_ref()),
                    black_box(0.0),
                    black_box(c_mat.as_mut()),
                );
            });
        });
    }

    group.finish();
}

fn bench_hemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("hemm_c64");

    for size in [64, 128, 256].iter() {
        let n = *size;
        // HEMM: C = alpha * A * B + beta * C (A is Hermitian)
        // FLOPs: 8 * n^2 * n (complex multiply-add)
        group.throughput(Throughput::Elements((8 * n * n * n) as u64));

        // Create Hermitian matrix A
        let mut a_data = vec![Complex64::new(0.0, 0.0); n * n];
        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    a_data[i * n + j] = Complex64::new((i % 10) as f64, 0.0);
                } else {
                    let val = Complex64::new((i % 10) as f64 * 0.1, (j % 10) as f64 * 0.1);
                    a_data[i * n + j] = val;
                    a_data[j * n + i] = val.conj();
                }
            }
        }
        let a = Mat::from_slice(n, n, &a_data);

        let b_data: Vec<Complex64> = (0..n * n)
            .map(|i| Complex64::new((i % 100) as f64 * 0.01, (i % 50) as f64 * 0.01))
            .collect();
        let b = Mat::from_slice(n, n, &b_data);
        let mut c_mat = Mat::zeros(n, n);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = hemm_c64(
                    black_box(Side::Left),
                    black_box(Uplo::Lower),
                    black_box(Complex64::new(1.0, 0.0)),
                    black_box(a.as_ref()),
                    black_box(b.as_ref()),
                    black_box(Complex64::new(0.0, 0.0)),
                    black_box(c_mat.as_mut()),
                );
            });
        });
    }

    group.finish();
}

fn bench_herk(c: &mut Criterion) {
    let mut group = c.benchmark_group("herk_f64");

    for size in [64, 128, 256].iter() {
        let n = *size;
        let k = n / 2;
        // HERK: C = alpha * A * A^H + beta * C (f64 version)
        // FLOPs: n^2 * k (rank-k update)
        group.throughput(Throughput::Elements((n * n * k) as u64));

        let a_data: Vec<f64> = (0..n * k).map(|i| (i % 100) as f64 * 0.01).collect();
        let a = Mat::from_slice(n, k, &a_data);
        let mut c_mat: Mat<f64> = Mat::zeros(n, n);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = herk(
                    black_box(Uplo::Lower),
                    black_box(Trans::NoTrans),
                    black_box(1.0),
                    black_box(a.as_ref()),
                    black_box(0.0),
                    black_box(c_mat.as_mut()),
                );
            });
        });
    }

    group.finish();
}

fn bench_her2k(c: &mut Criterion) {
    let mut group = c.benchmark_group("her2k_f64");

    for size in [64, 128, 256].iter() {
        let n = *size;
        let k = n / 2;
        // HER2K: C = alpha * A * B^H + conj(alpha) * B * A^H + beta * C (f64 version)
        // FLOPs: 2 * n^2 * k (rank-2k update)
        group.throughput(Throughput::Elements((2 * n * n * k) as u64));

        let a_data: Vec<f64> = (0..n * k).map(|i| (i % 100) as f64 * 0.01).collect();
        let a = Mat::from_slice(n, k, &a_data);
        let b_data: Vec<f64> = (0..n * k).map(|i| ((i + 50) % 100) as f64 * 0.01).collect();
        let b = Mat::from_slice(n, k, &b_data);
        let mut c_mat: Mat<f64> = Mat::zeros(n, n);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = her2k(
                    black_box(Uplo::Lower),
                    black_box(Trans::NoTrans),
                    black_box(1.0),
                    black_box(a.as_ref()),
                    black_box(b.as_ref()),
                    black_box(0.0),
                    black_box(c_mat.as_mut()),
                );
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_gemm_f64,
    bench_gemm_rectangular,
    bench_gemm3m_complex,
    bench_trmm,
    bench_syrk,
    bench_trsm,
    bench_syr2k,
    bench_symm,
    bench_hemm,
    bench_herk,
    bench_her2k
);
criterion_main!(benches);
