//! Benchmarks for BLAS Level 2 operations (matrix-vector).

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use num_complex::Complex64;
use oxiblas_blas::level2::{
    GemvTrans, HbmvUplo, HerUplo, HpmvUplo, HprUplo, SbmvUplo, SpmvUplo, SprUplo, SymvUplo,
    SyrUplo, TbmvDiag, TbmvTrans, TbmvUplo, TbsvDiag, TbsvTrans, TbsvUplo, TpmvDiag, TpmvTrans,
    TpmvUplo, TpsvDiag, TpsvTrans, TpsvUplo, gemv, ger, hbmv, her, hpmv, hpr, sbmv, spmv, spr,
    symv, syr, tbmv, tbsv, tpmv, tpsv,
};
use oxiblas_matrix::Mat;
use std::hint::black_box;

/// Helper function to create a complex number.
fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn bench_gemv(c: &mut Criterion) {
    let mut group = c.benchmark_group("gemv");

    for size in [50, 100, 200, 500].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        let data: Vec<f64> = (0..n * n).map(|i| i as f64).collect();
        let a = Mat::from_slice(n, n, &data);
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut y: Vec<f64> = vec![0.0; n];

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                gemv(
                    black_box(GemvTrans::NoTrans),
                    black_box(1.0),
                    black_box(a.as_ref()),
                    black_box(&x),
                    black_box(0.0),
                    black_box(&mut y),
                );
            });
        });
    }

    group.finish();
}

fn bench_ger(c: &mut Criterion) {
    let mut group = c.benchmark_group("ger");

    for size in [50, 100, 200, 500].iter() {
        let n = *size;
        group.throughput(Throughput::Elements((n * n) as u64));

        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let mut a = Mat::zeros(n, n);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                ger(
                    black_box(1.0),
                    black_box(&x),
                    black_box(&y),
                    black_box(a.as_mut()),
                );
            });
        });
    }

    group.finish();
}

fn bench_syr(c: &mut Criterion) {
    let mut group = c.benchmark_group("syr");

    for size in [50, 100, 200, 500].iter() {
        let n = *size;
        // SYR: A = alpha * x * x^T + A (only upper or lower triangle)
        group.throughput(Throughput::Elements((n * (n + 1) / 2) as u64));

        let x: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.01).collect();
        let mut a = Mat::zeros(n, n);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = syr(
                    black_box(SyrUplo::Lower),
                    black_box(1.0),
                    black_box(&x),
                    black_box(a.as_mut()),
                );
            });
        });
    }

    group.finish();
}

fn bench_spr(c: &mut Criterion) {
    let mut group = c.benchmark_group("spr");

    for size in [50, 100, 200, 500].iter() {
        let n = *size;
        // SPR: A = alpha * x * x^T + A (packed storage)
        let packed_size = n * (n + 1) / 2;
        group.throughput(Throughput::Elements(packed_size as u64));

        let x: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.01).collect();
        let mut ap: Vec<f64> = vec![0.0; packed_size];

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = spr(
                    black_box(SprUplo::Lower),
                    black_box(n),
                    black_box(1.0),
                    black_box(&x),
                    black_box(&mut ap),
                );
            });
        });
    }

    group.finish();
}

fn bench_symv(c: &mut Criterion) {
    let mut group = c.benchmark_group("symv");

    for size in [50, 100, 200, 500].iter() {
        let n = *size;
        // SYMV: y = alpha * A * x + beta * y (A is symmetric)
        group.throughput(Throughput::Elements((n * n) as u64));

        // Create symmetric matrix
        let mut a_data = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let val = ((i + j) % 100) as f64 * 0.01;
                a_data[i * n + j] = val;
                a_data[j * n + i] = val;
            }
        }
        let a = Mat::from_slice(n, n, &a_data);
        let x: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.01).collect();
        let mut y: Vec<f64> = vec![0.0; n];

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = symv(
                    black_box(SymvUplo::Lower),
                    black_box(1.0),
                    black_box(a.as_ref()),
                    black_box(&x),
                    black_box(0.0),
                    black_box(&mut y),
                );
            });
        });
    }

    group.finish();
}

fn bench_sbmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("sbmv");

    for size in [100, 200, 500].iter() {
        let n = *size;
        let k = 10usize.min(n - 1); // bandwidth
        // SBMV: y = alpha * A * x + beta * y (A is symmetric banded)
        group.throughput(Throughput::Elements((n * (2 * k + 1)) as u64));

        // Create banded storage as Mat (k+1 rows, n columns)
        let ab_data: Vec<f64> = (0..(k + 1) * n).map(|i| (i % 100) as f64 * 0.01).collect();
        let ab = Mat::from_slice(k + 1, n, &ab_data);
        let x: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.01).collect();
        let mut y: Vec<f64> = vec![0.0; n];

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = sbmv(
                    black_box(SbmvUplo::Lower),
                    black_box(n),
                    black_box(k),
                    black_box(1.0),
                    black_box(ab.as_ref()),
                    black_box(&x),
                    black_box(0.0),
                    black_box(&mut y),
                );
            });
        });
    }

    group.finish();
}

fn bench_tbmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("tbmv");

    for size in [100, 200, 500].iter() {
        let n = *size;
        let k = 10usize.min(n - 1); // bandwidth
        // TBMV: x = A * x (A is triangular banded)
        group.throughput(Throughput::Elements((n * (k + 1)) as u64));

        // Create banded storage as Mat (k+1 rows, n columns)
        let ab_data: Vec<f64> = (0..(k + 1) * n)
            .map(|i| (i % 100) as f64 * 0.01 + 1.0)
            .collect();
        let ab = Mat::from_slice(k + 1, n, &ab_data);
        let mut x: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.01).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = tbmv(
                    black_box(TbmvUplo::Upper),
                    black_box(TbmvTrans::NoTrans),
                    black_box(TbmvDiag::NonUnit),
                    black_box(n),
                    black_box(k),
                    black_box(ab.as_ref()),
                    black_box(&mut x),
                );
            });
        });
    }

    group.finish();
}

fn bench_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv");

    for size in [50, 100, 200, 500].iter() {
        let n = *size;
        // SPMV: y = alpha * A * x + beta * y (A is symmetric packed)
        let packed_size = n * (n + 1) / 2;
        group.throughput(Throughput::Elements(packed_size as u64));

        let ap: Vec<f64> = (0..packed_size).map(|i| (i % 100) as f64 * 0.01).collect();
        let x: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.01).collect();
        let mut y: Vec<f64> = vec![0.0; n];

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = spmv(
                    black_box(SpmvUplo::Lower),
                    black_box(n),
                    black_box(1.0),
                    black_box(&ap),
                    black_box(&x),
                    black_box(0.0),
                    black_box(&mut y),
                );
            });
        });
    }

    group.finish();
}

fn bench_tbsv(c: &mut Criterion) {
    let mut group = c.benchmark_group("tbsv");

    for size in [100, 200, 500].iter() {
        let n = *size;
        let k = 10usize.min(n - 1); // bandwidth
        // TBSV: solve A * x = b (A is triangular banded)
        group.throughput(Throughput::Elements((n * (k + 1)) as u64));

        // Create banded storage as Mat (k+1 rows, n columns)
        // Ensure diagonal elements are non-zero for solve
        let ab_data: Vec<f64> = (0..(k + 1) * n)
            .map(|i| (i % 100) as f64 * 0.01 + 1.0)
            .collect();
        let ab = Mat::from_slice(k + 1, n, &ab_data);
        let mut x: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.01).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = tbsv(
                    black_box(TbsvUplo::Upper),
                    black_box(TbsvTrans::NoTrans),
                    black_box(TbsvDiag::NonUnit),
                    black_box(n),
                    black_box(k),
                    black_box(ab.as_ref()),
                    black_box(&mut x),
                );
            });
        });
    }

    group.finish();
}

fn bench_tpmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("tpmv");

    for size in [50, 100, 200, 500].iter() {
        let n = *size;
        // TPMV: x = A * x (A is triangular packed)
        let packed_size = n * (n + 1) / 2;
        group.throughput(Throughput::Elements(packed_size as u64));

        let ap: Vec<f64> = (0..packed_size)
            .map(|i| (i % 100) as f64 * 0.01 + 1.0)
            .collect();
        let mut x: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.01).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = tpmv(
                    black_box(TpmvUplo::Upper),
                    black_box(TpmvTrans::NoTrans),
                    black_box(TpmvDiag::NonUnit),
                    black_box(n),
                    black_box(&ap),
                    black_box(&mut x),
                );
            });
        });
    }

    group.finish();
}

fn bench_tpsv(c: &mut Criterion) {
    let mut group = c.benchmark_group("tpsv");

    for size in [50, 100, 200, 500].iter() {
        let n = *size;
        // TPSV: solve A * x = b (A is triangular packed)
        let packed_size = n * (n + 1) / 2;
        group.throughput(Throughput::Elements(packed_size as u64));

        // Ensure diagonal elements are non-zero for solve
        let ap: Vec<f64> = (0..packed_size)
            .map(|i| (i % 100) as f64 * 0.01 + 1.0)
            .collect();
        let mut x: Vec<f64> = (0..n).map(|i| (i % 100) as f64 * 0.01).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = tpsv(
                    black_box(TpsvUplo::Upper),
                    black_box(TpsvTrans::NoTrans),
                    black_box(TpsvDiag::NonUnit),
                    black_box(n),
                    black_box(&ap),
                    black_box(&mut x),
                );
            });
        });
    }

    group.finish();
}

fn bench_her(cr: &mut Criterion) {
    let mut group = cr.benchmark_group("her");

    for size in [50, 100, 200, 500].iter() {
        let n = *size;
        // HER: A = alpha * x * x^H + A (only upper or lower triangle)
        group.throughput(Throughput::Elements((n * (n + 1) / 2) as u64));

        let x: Vec<Complex64> = (0..n)
            .map(|i| c((i % 100) as f64 * 0.01, (i % 50) as f64 * 0.01))
            .collect();
        let mut a_data: Vec<Complex64> = vec![c(0.0, 0.0); n * n];
        // Initialize with Hermitian matrix (diagonal real)
        for i in 0..n {
            a_data[i * n + i] = c((i + 1) as f64, 0.0);
        }
        let mut a = Mat::from_slice(n, n, &a_data);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = her(
                    black_box(HerUplo::Lower),
                    black_box(1.0),
                    black_box(&x),
                    black_box(a.as_mut()),
                );
            });
        });
    }

    group.finish();
}

fn bench_hpr(cr: &mut Criterion) {
    let mut group = cr.benchmark_group("hpr");

    for size in [50, 100, 200, 500].iter() {
        let n = *size;
        // HPR: A = alpha * x * x^H + A (packed storage)
        let packed_size = n * (n + 1) / 2;
        group.throughput(Throughput::Elements(packed_size as u64));

        let x: Vec<Complex64> = (0..n)
            .map(|i| c((i % 100) as f64 * 0.01, (i % 50) as f64 * 0.01))
            .collect();
        let mut ap: Vec<Complex64> = vec![c(0.0, 0.0); packed_size];

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = hpr(
                    black_box(HprUplo::Lower),
                    black_box(n),
                    black_box(1.0),
                    black_box(&x),
                    black_box(&mut ap),
                );
            });
        });
    }

    group.finish();
}

fn bench_hbmv(cr: &mut Criterion) {
    let mut group = cr.benchmark_group("hbmv");

    for size in [100, 200, 500].iter() {
        let n = *size;
        let k = 10usize.min(n - 1); // bandwidth
        // HBMV: y = alpha * A * x + beta * y (A is Hermitian banded)
        group.throughput(Throughput::Elements((n * (2 * k + 1)) as u64));

        // Create banded storage as Mat (k+1 rows, n columns)
        let ab_data: Vec<Complex64> = (0..(k + 1) * n)
            .map(|i| c((i % 100) as f64 * 0.01, (i % 50) as f64 * 0.005))
            .collect();
        let ab = Mat::from_slice(k + 1, n, &ab_data);
        let x: Vec<Complex64> = (0..n).map(|i| c((i % 100) as f64 * 0.01, 0.0)).collect();
        let mut y: Vec<Complex64> = vec![c(0.0, 0.0); n];

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = hbmv(
                    black_box(HbmvUplo::Upper),
                    black_box(n),
                    black_box(k),
                    black_box(c(1.0, 0.0)),
                    black_box(ab.as_ref()),
                    black_box(&x),
                    black_box(c(0.0, 0.0)),
                    black_box(&mut y),
                );
            });
        });
    }

    group.finish();
}

fn bench_hpmv(cr: &mut Criterion) {
    let mut group = cr.benchmark_group("hpmv");

    for size in [50, 100, 200, 500].iter() {
        let n = *size;
        // HPMV: y = alpha * A * x + beta * y (A is Hermitian packed)
        let packed_size = n * (n + 1) / 2;
        group.throughput(Throughput::Elements(packed_size as u64));

        let ap: Vec<Complex64> = (0..packed_size)
            .map(|i| c((i % 100) as f64 * 0.01, (i % 50) as f64 * 0.005))
            .collect();
        let x: Vec<Complex64> = (0..n).map(|i| c((i % 100) as f64 * 0.01, 0.0)).collect();
        let mut y: Vec<Complex64> = vec![c(0.0, 0.0); n];

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |bench, _| {
            bench.iter(|| {
                let _ = hpmv(
                    black_box(HpmvUplo::Lower),
                    black_box(n),
                    black_box(c(1.0, 0.0)),
                    black_box(&ap),
                    black_box(&x),
                    black_box(c(0.0, 0.0)),
                    black_box(&mut y),
                );
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches, bench_gemv, bench_ger, bench_syr, bench_spr, bench_symv, bench_sbmv, bench_tbmv,
    bench_spmv, bench_tbsv, bench_tpmv, bench_tpsv, bench_her, bench_hpr, bench_hbmv, bench_hpmv
);
criterion_main!(benches);
