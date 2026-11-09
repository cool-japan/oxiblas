//! LU Factorization benchmark
//!
//! Compares unblocked vs blocked LU factorization performance.

use oxiblas::prelude::*;
use oxiblas_lapack::lu::Lu;
use oxiblas_lapack::workspace::optimal_block_size_lu;
use std::time::Instant;

fn lu_naive(a: &Mat<f64>) -> Mat<f64> {
    let n = a.nrows();
    let mut lu: Mat<f64> = Mat::zeros(n, n);

    // Copy A
    for i in 0..n {
        for j in 0..n {
            lu[(i, j)] = a[(i, j)];
        }
    }

    // Doolittle algorithm (no pivoting for benchmark)
    for k in 0..n {
        let pivot = lu[(k, k)];
        for i in (k + 1)..n {
            let mult = lu[(i, k)] / pivot;
            lu[(i, k)] = mult;
            for j in (k + 1)..n {
                lu[(i, j)] -= mult * lu[(k, j)];
            }
        }
    }
    lu
}

/// Median of the samples (sorted in place).
fn median(times: &mut [f64]) -> f64 {
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

fn bench_lu(sizes: &[(usize, &str)], n_warmup: usize, n_samples: usize) {
    println!("\n=== f64 LU Factorization Benchmark ===");
    println!(
        "{:>8} {:>5} {:>12} {:>12} {:>12} {:>12}",
        "Size", "nb", "Naive", "Unblocked", "Blocked", "Speedup"
    );
    println!("{}", "-".repeat(68));

    for &(n, label) in sizes {
        // Create diagonally dominant matrix
        let mut a: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = ((i * 17 + j * 31) % 100) as f64 / 100.0;
                if i == j {
                    a[(i, j)] += (n as f64) * 0.5;
                }
            }
        }

        // Warmup naive
        for _ in 0..n_warmup {
            let _ = lu_naive(&a);
        }

        // Measure naive
        let mut naive_times = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let start = Instant::now();
            let _ = lu_naive(&a);
            naive_times.push(start.elapsed().as_secs_f64());
        }
        let naive_median = median(&mut naive_times);

        // Warmup unblocked
        for _ in 0..n_warmup {
            let _ = Lu::compute(a.as_ref());
        }

        // Measure unblocked
        let mut unblocked_times = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let start = Instant::now();
            let _ = Lu::compute(a.as_ref());
            unblocked_times.push(start.elapsed().as_secs_f64());
        }
        let unblocked_median = median(&mut unblocked_times);

        // Warmup blocked
        for _ in 0..n_warmup {
            let _ = Lu::compute_blocked(a.as_ref());
        }

        // Measure blocked
        let mut blocked_times = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let start = Instant::now();
            let _ = Lu::compute_blocked(a.as_ref());
            blocked_times.push(start.elapsed().as_secs_f64());
        }
        let blocked_median = median(&mut blocked_times);

        // LU has ~(2/3)n³ flops
        let flops = 2.0 * (n as f64).powi(3) / 3.0;
        let naive_gflops = flops / naive_median / 1e9;
        let unblocked_gflops = flops / unblocked_median / 1e9;
        let blocked_gflops = flops / blocked_median / 1e9;
        let speedup = unblocked_median / blocked_median;

        // The panel width `Lu::compute_blocked` uses for this size.
        let nb = optimal_block_size_lu(n, n);

        println!(
            "{:>8} {:>5} {:>9.2} G/s {:>9.2} G/s {:>9.2} G/s {:>9.2}x",
            label, nb, naive_gflops, unblocked_gflops, blocked_gflops, speedup
        );
    }
}

fn main() {
    println!("==============================================");
    println!("   OxiBLAS LU Factorization Benchmark");
    println!("==============================================");
    println!();
    println!("LU decomposition: PA = LU with partial pivoting");
    println!("Blocked algorithm uses GEMM/TRSM for cache efficiency.");
    println!();
    println!("Naive: Simple O(n³) without pivoting (for reference)");
    println!("Unblocked: With pivoting, column-oriented rank-1 updates");
    println!("Blocked: With pivoting, recursive panel + in-place TRSM/GEMM updates,");
    println!("         panel width nb from optimal_block_size_lu: n if n < 64,");
    println!("         32 if n < 256, 64 if n < 1024, 128 otherwise (nb column below)");
    println!("The test matrices are diagonally dominant: no row interchanges occur.");

    let sizes = [
        (64, "64"),
        (128, "128"),
        (256, "256"),
        (512, "512"),
        (768, "768"),
        (1024, "1024"),
    ];

    bench_lu(&sizes, 2, 5);

    println!();
    println!("Note: Blocked algorithm is most effective for n >= 64.");
    println!("      Speedup increases with matrix size due to better cache usage.");
}
