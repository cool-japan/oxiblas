//! TRMM/TRSM benchmark
//!
//! Compares naive vs optimized TRMM (GEMM-based) and TRSM operations.

use oxiblas::prelude::*;
use oxiblas_blas::level3::trmm::{TrmmDiag, TrmmSide, TrmmTrans, TrmmUplo, trmm};
use oxiblas_blas::level3::trsm::{Diag, Side, Trans, Uplo, trsm};
use std::time::Instant;

/// Naive TRMM for comparison (Left, Lower, NoTrans: B = α·L·B)
fn trmm_naive(l: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    let m = b.nrows();
    let n = b.ncols();
    let mut result: Mat<f64> = Mat::zeros(m, n);

    for j in 0..n {
        for i in 0..m {
            let mut sum = 0.0;
            // Lower triangular: only elements at or below diagonal
            for k in 0..=i {
                sum += l[(i, k)] * b[(k, j)];
            }
            result[(i, j)] = sum;
        }
    }
    result
}

fn bench_trmm(sizes: &[(usize, &str)], n_warmup: usize, n_samples: usize) {
    println!("\n=== f64 TRMM (Triangular Matrix-Matrix Multiply: B = α·L·B) ===");
    println!(
        "{:>8} {:>12} {:>12} {:>12}",
        "Size", "Naive", "Optimized", "Speedup"
    );
    println!("{}", "-".repeat(50));

    for &(n, label) in sizes {
        // Create lower triangular matrix L
        let mut l: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                l[(i, j)] = 0.001 * (i + j + 1) as f64;
            }
        }

        // Create matrix B
        let mut b: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                b[(i, j)] = 0.002 * (i + j + 1) as f64;
            }
        }

        // Warmup naive
        for _ in 0..n_warmup {
            let _ = trmm_naive(&l, &b);
        }

        // Measure naive
        let mut naive_times = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let start = Instant::now();
            let _ = trmm_naive(&l, &b);
            naive_times.push(start.elapsed().as_secs_f64());
        }
        naive_times.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let naive_median = naive_times[n_samples / 2];

        // Warmup optimized
        for _ in 0..n_warmup {
            let _ = trmm(
                TrmmSide::Left,
                TrmmUplo::Lower,
                TrmmTrans::NoTrans,
                TrmmDiag::NonUnit,
                1.0,
                l.as_ref(),
                b.as_ref(),
            );
        }

        // Measure optimized
        let mut opt_times = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let start = Instant::now();
            let _ = trmm(
                TrmmSide::Left,
                TrmmUplo::Lower,
                TrmmTrans::NoTrans,
                TrmmDiag::NonUnit,
                1.0,
                l.as_ref(),
                b.as_ref(),
            );
            opt_times.push(start.elapsed().as_secs_f64());
        }
        opt_times.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let opt_median = opt_times[n_samples / 2];

        // TRMM has ~n²·m flops (for left multiply with m×n result)
        // For square matrices: ~n³ (but only half of A is used)
        let flops = 2.0 * (n as f64) * (n as f64) * (n as f64);
        let naive_gflops = flops / naive_median / 1e9;
        let opt_gflops = flops / opt_median / 1e9;
        let speedup = naive_median / opt_median;

        println!(
            "{:>8} {:>9.2} G/s {:>9.2} G/s {:>10.2}x",
            label, naive_gflops, opt_gflops, speedup
        );
    }
}

/// Naive TRSM for comparison (Left, Lower, NoTrans: L·X = B)
fn trsm_naive(l: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    let m = b.nrows();
    let n = b.ncols();
    let mut x: Mat<f64> = Mat::zeros(m, n);

    // Copy B to X
    for j in 0..n {
        for i in 0..m {
            x[(i, j)] = b[(i, j)];
        }
    }

    // Forward substitution
    for j in 0..n {
        for i in 0..m {
            let mut sum = x[(i, j)];
            for k in 0..i {
                sum -= l[(i, k)] * x[(k, j)];
            }
            x[(i, j)] = sum / l[(i, i)];
        }
    }
    x
}

fn bench_trsm(sizes: &[(usize, &str)], n_warmup: usize, n_samples: usize) {
    println!("\n=== f64 TRSM (Triangular Solve: L·X = B) ===");
    println!(
        "{:>8} {:>12} {:>12} {:>12}",
        "Size", "Naive", "Blocked", "Speedup"
    );
    println!("{}", "-".repeat(50));

    for &(n, label) in sizes {
        // Create lower triangular matrix L (diagonally dominant for stability)
        let mut l: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..i {
                l[(i, j)] = 0.001 * (i + j + 1) as f64;
            }
            // Diagonal should be larger than sum of row for stability
            l[(i, i)] = (n as f64) * 0.01;
        }

        // Create matrix B
        let mut b: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                b[(i, j)] = 0.002 * (i + j + 1) as f64;
            }
        }

        // Warmup naive
        for _ in 0..n_warmup {
            let _ = trsm_naive(&l, &b);
        }

        // Measure naive
        let mut naive_times = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let start = Instant::now();
            let _ = trsm_naive(&l, &b);
            naive_times.push(start.elapsed().as_secs_f64());
        }
        naive_times.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let naive_median = naive_times[n_samples / 2];

        // Warmup blocked
        for _ in 0..n_warmup {
            let _ = trsm(
                Side::Left,
                Uplo::Lower,
                Trans::NoTrans,
                Diag::NonUnit,
                1.0,
                l.as_ref(),
                b.as_ref(),
            );
        }

        // Measure blocked
        let mut blocked_times = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let start = Instant::now();
            let _ = trsm(
                Side::Left,
                Uplo::Lower,
                Trans::NoTrans,
                Diag::NonUnit,
                1.0,
                l.as_ref(),
                b.as_ref(),
            );
            blocked_times.push(start.elapsed().as_secs_f64());
        }
        blocked_times.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let blocked_median = blocked_times[n_samples / 2];

        // TRSM has ~n²·m flops for solving L·X = B where X is m×n
        let flops = (n as f64) * (n as f64) * (n as f64);
        let naive_gflops = flops / naive_median / 1e9;
        let blocked_gflops = flops / blocked_median / 1e9;
        let speedup = naive_median / blocked_median;

        println!(
            "{:>8} {:>9.2} G/s {:>9.2} G/s {:>10.2}x",
            label, naive_gflops, blocked_gflops, speedup
        );
    }
}

fn main() {
    println!("==============================================");
    println!("   OxiBLAS TRMM/TRSM Benchmark");
    println!("==============================================");
    println!();
    println!("TRMM: B = α·op(A)·B or B = α·B·op(A) where A is triangular");
    println!("TRSM: Solve A·X = α·B or X·A = α·B where A is triangular");
    println!();
    println!("TRMM is optimized by expanding triangular matrix to full");
    println!("matrix and using the optimized GEMM kernel.");

    let sizes = [(128, "128"), (256, "256"), (512, "512"), (1024, "1024")];

    bench_trmm(&sizes, 2, 5);
    bench_trsm(&sizes, 2, 5);

    println!();
    println!("Note: TRMM optimized for n >= 32 using GEMM-based approach.");
    println!("      TRSM optimized for n >= 64 using blocked algorithm with GEMM.");
}
