//! Parallel GEMM scaling benchmark
//!
//! This example requires the `parallel` feature:
//! ```bash
//! cargo run --example parallel_bench --features parallel
//! ```

use oxiblas::prelude::*;
use std::time::Instant;

fn bench_gemm_f64(sizes: &[(usize, &str)], n_warmup: usize, n_samples: usize) {
    println!("\n=== f64 DGEMM Parallel Scaling ===");
    println!(
        "{:>8} {:>12} {:>12} {:>12}",
        "Size", "Sequential", "Parallel", "Speedup"
    );
    println!("{}", "-".repeat(50));

    for &(n, label) in sizes {
        // Create matrices
        let a: Mat<f64> = Mat::filled(n, n, 1.0);
        let b: Mat<f64> = Mat::filled(n, n, 1.0);

        // Warmup
        for _ in 0..n_warmup {
            let mut c: Mat<f64> = Mat::zeros(n, n);
            gemm_with_par(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut(), Par::Seq);
        }

        // Measure sequential
        let mut seq_times = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let mut c: Mat<f64> = Mat::zeros(n, n);
            let start = Instant::now();
            gemm_with_par(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut(), Par::Seq);
            seq_times.push(start.elapsed().as_secs_f64());
            std::hint::black_box(&c);
        }
        seq_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let seq_median = seq_times[n_samples / 2];

        // Parallel benchmark

        // Warmup
        for _ in 0..n_warmup {
            let mut c: Mat<f64> = Mat::zeros(n, n);
            gemm_with_par(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut(), Par::Rayon);
        }

        // Measure parallel
        let mut par_times = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let mut c: Mat<f64> = Mat::zeros(n, n);
            let start = Instant::now();
            gemm_with_par(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut(), Par::Rayon);
            par_times.push(start.elapsed().as_secs_f64());
            std::hint::black_box(&c);
        }
        par_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let par_median = par_times[n_samples / 2];

        let flops = 2.0 * (n as f64).powi(3);
        let seq_gflops = flops / seq_median / 1e9;
        let par_gflops = flops / par_median / 1e9;
        let speedup = seq_median / par_median;

        println!(
            "{:>8} {:>9.2} G/s {:>9.2} G/s {:>10.2}x",
            label, seq_gflops, par_gflops, speedup
        );
    }
}

fn bench_gemm_f32(sizes: &[(usize, &str)], n_warmup: usize, n_samples: usize) {
    println!("\n=== f32 SGEMM Parallel Scaling ===");
    println!(
        "{:>8} {:>12} {:>12} {:>12}",
        "Size", "Sequential", "Parallel", "Speedup"
    );
    println!("{}", "-".repeat(50));

    for &(n, label) in sizes {
        // Create matrices
        let a: Mat<f32> = Mat::filled(n, n, 1.0);
        let b: Mat<f32> = Mat::filled(n, n, 1.0);

        // Warmup
        for _ in 0..n_warmup {
            let mut c: Mat<f32> = Mat::zeros(n, n);
            gemm_with_par(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut(), Par::Seq);
        }

        // Measure sequential
        let mut seq_times = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let mut c: Mat<f32> = Mat::zeros(n, n);
            let start = Instant::now();
            gemm_with_par(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut(), Par::Seq);
            seq_times.push(start.elapsed().as_secs_f64());
            std::hint::black_box(&c);
        }
        seq_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let seq_median = seq_times[n_samples / 2];

        // Parallel benchmark

        // Warmup
        for _ in 0..n_warmup {
            let mut c: Mat<f32> = Mat::zeros(n, n);
            gemm_with_par(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut(), Par::Rayon);
        }

        // Measure parallel
        let mut par_times = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let mut c: Mat<f32> = Mat::zeros(n, n);
            let start = Instant::now();
            gemm_with_par(1.0, a.as_ref(), b.as_ref(), 0.0, c.as_mut(), Par::Rayon);
            par_times.push(start.elapsed().as_secs_f64());
            std::hint::black_box(&c);
        }
        par_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let par_median = par_times[n_samples / 2];

        let flops = 2.0 * (n as f64).powi(3);
        let seq_gflops = flops / seq_median / 1e9;
        let par_gflops = flops / par_median / 1e9;
        let speedup = seq_median / par_median;

        println!(
            "{:>8} {:>9.2} G/s {:>9.2} G/s {:>10.2}x",
            label, seq_gflops, par_gflops, speedup
        );
    }
}

fn main() {
    println!("==============================================");
    println!("   OxiBLAS Parallel GEMM Scaling Benchmark");
    println!("==============================================");

    let sizes = [(512, "512"), (1024, "1024"), (2048, "2048")];

    // f64 benchmark
    bench_gemm_f64(&sizes, 5, 10);

    // f32 benchmark
    bench_gemm_f32(&sizes, 5, 10);
}
