//! Complex GEMM (3M method) benchmark
//!
//! Measures performance of complex matrix multiplication using the 3M method
//! that leverages optimized real GEMM kernels.

use num_complex::{Complex32, Complex64};
use oxiblas::prelude::*;
use oxiblas_blas::level3::{gemm3m_c32, gemm3m_c64};
use std::time::Instant;

/// Naive complex GEMM for comparison (4 real multiplications per complex multiplication)
fn gemm_naive_c64(a: &Mat<Complex64>, b: &Mat<Complex64>, c: &mut Mat<Complex64>) {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

    for i in 0..m {
        for j in 0..n {
            let mut sum = Complex64::new(0.0, 0.0);
            for p in 0..k {
                sum += a[(i, p)] * b[(p, j)];
            }
            c[(i, j)] = sum;
        }
    }
}

fn gemm_naive_c32(a: &Mat<Complex32>, b: &Mat<Complex32>, c: &mut Mat<Complex32>) {
    let m = a.nrows();
    let k = a.ncols();
    let n = b.ncols();

    for i in 0..m {
        for j in 0..n {
            let mut sum = Complex32::new(0.0, 0.0);
            for p in 0..k {
                sum += a[(i, p)] * b[(p, j)];
            }
            c[(i, j)] = sum;
        }
    }
}

fn bench_complex_gemm_c64(sizes: &[(usize, &str)], n_warmup: usize, n_samples: usize) {
    println!("\n=== Complex64 ZGEMM (3M Method) ===");
    println!(
        "{:>8} {:>12} {:>12} {:>12}",
        "Size", "Naive", "3M Method", "Speedup"
    );
    println!("{}", "-".repeat(50));

    for &(n, label) in sizes {
        // Create complex matrices with values
        let mut a: Mat<Complex64> = Mat::zeros(n, n);
        let mut b: Mat<Complex64> = Mat::zeros(n, n);

        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = Complex64::new((i + j) as f64 * 0.001, (i * j) as f64 * 0.0001);
                b[(i, j)] = Complex64::new((i + 1) as f64 * 0.002, (j + 1) as f64 * 0.0002);
            }
        }

        // Warmup naive
        for _ in 0..n_warmup {
            let mut c: Mat<Complex64> = Mat::zeros(n, n);
            gemm_naive_c64(&a, &b, &mut c);
            std::hint::black_box(&c);
        }

        // Measure naive
        let mut naive_times = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let mut c: Mat<Complex64> = Mat::zeros(n, n);
            let start = Instant::now();
            gemm_naive_c64(&a, &b, &mut c);
            naive_times.push(start.elapsed().as_secs_f64());
            std::hint::black_box(&c);
        }
        naive_times.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let naive_median = naive_times[n_samples / 2];

        // Warmup 3M
        for _ in 0..n_warmup {
            let mut c: Mat<Complex64> = Mat::zeros(n, n);
            gemm3m_c64(
                Complex64::new(1.0, 0.0),
                a.as_ref(),
                b.as_ref(),
                Complex64::new(0.0, 0.0),
                c.as_mut(),
            );
            std::hint::black_box(&c);
        }

        // Measure 3M
        let mut opt_times = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let mut c: Mat<Complex64> = Mat::zeros(n, n);
            let start = Instant::now();
            gemm3m_c64(
                Complex64::new(1.0, 0.0),
                a.as_ref(),
                b.as_ref(),
                Complex64::new(0.0, 0.0),
                c.as_mut(),
            );
            opt_times.push(start.elapsed().as_secs_f64());
            std::hint::black_box(&c);
        }
        opt_times.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let opt_median = opt_times[n_samples / 2];

        // Complex GEMM has 8 real flops per complex multiply-add
        // For C = A * B where all are n×n: 8 * n³ flops
        let flops = 8.0 * (n as f64).powi(3);
        let naive_gflops = flops / naive_median / 1e9;
        let opt_gflops = flops / opt_median / 1e9;
        let speedup = naive_median / opt_median;

        println!(
            "{:>8} {:>9.2} G/s {:>9.2} G/s {:>10.2}x",
            label, naive_gflops, opt_gflops, speedup
        );
    }
}

fn bench_complex_gemm_c32(sizes: &[(usize, &str)], n_warmup: usize, n_samples: usize) {
    println!("\n=== Complex32 CGEMM (3M Method) ===");
    println!(
        "{:>8} {:>12} {:>12} {:>12}",
        "Size", "Naive", "3M Method", "Speedup"
    );
    println!("{}", "-".repeat(50));

    for &(n, label) in sizes {
        // Create complex matrices with values
        let mut a: Mat<Complex32> = Mat::zeros(n, n);
        let mut b: Mat<Complex32> = Mat::zeros(n, n);

        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = Complex32::new((i + j) as f32 * 0.001, (i * j) as f32 * 0.0001);
                b[(i, j)] = Complex32::new((i + 1) as f32 * 0.002, (j + 1) as f32 * 0.0002);
            }
        }

        // Warmup naive
        for _ in 0..n_warmup {
            let mut c: Mat<Complex32> = Mat::zeros(n, n);
            gemm_naive_c32(&a, &b, &mut c);
            std::hint::black_box(&c);
        }

        // Measure naive
        let mut naive_times = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let mut c: Mat<Complex32> = Mat::zeros(n, n);
            let start = Instant::now();
            gemm_naive_c32(&a, &b, &mut c);
            naive_times.push(start.elapsed().as_secs_f64());
            std::hint::black_box(&c);
        }
        naive_times.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let naive_median = naive_times[n_samples / 2];

        // Warmup 3M
        for _ in 0..n_warmup {
            let mut c: Mat<Complex32> = Mat::zeros(n, n);
            gemm3m_c32(
                Complex32::new(1.0, 0.0),
                a.as_ref(),
                b.as_ref(),
                Complex32::new(0.0, 0.0),
                c.as_mut(),
            );
            std::hint::black_box(&c);
        }

        // Measure 3M
        let mut opt_times = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let mut c: Mat<Complex32> = Mat::zeros(n, n);
            let start = Instant::now();
            gemm3m_c32(
                Complex32::new(1.0, 0.0),
                a.as_ref(),
                b.as_ref(),
                Complex32::new(0.0, 0.0),
                c.as_mut(),
            );
            opt_times.push(start.elapsed().as_secs_f64());
            std::hint::black_box(&c);
        }
        opt_times.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let opt_median = opt_times[n_samples / 2];

        // Complex GEMM has 8 real flops per complex multiply-add
        let flops = 8.0 * (n as f64).powi(3);
        let naive_gflops = flops / naive_median / 1e9;
        let opt_gflops = flops / opt_median / 1e9;
        let speedup = naive_median / opt_median;

        println!(
            "{:>8} {:>9.2} G/s {:>9.2} G/s {:>10.2}x",
            label, naive_gflops, opt_gflops, speedup
        );
    }
}

fn main() {
    println!("==============================================");
    println!("   OxiBLAS Complex GEMM (3M) Benchmark");
    println!("==============================================");
    println!();
    println!("The 3M method uses 3 optimized real GEMM operations");
    println!("with NEON SIMD, cache blocking, and prefetching.");

    let sizes = [(256, "256"), (512, "512"), (1024, "1024")];

    bench_complex_gemm_c64(&sizes, 3, 5);
    bench_complex_gemm_c32(&sizes, 3, 5);

    println!("\nNote: Complex GEMM counts 8 real flops per complex multiply-add.");
}
