//! LU Factorization benchmark
//!
//! Compares unblocked vs blocked LU factorization performance.

use oxiblas::prelude::*;
use oxiblas_lapack::lu::Lu;
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

fn bench_lu(sizes: &[(usize, &str)], n_warmup: usize, n_samples: usize) {
    println!("\n=== f64 LU Factorization Benchmark ===");
    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>12}",
        "Size", "Naive", "Unblocked", "Blocked", "Speedup"
    );
    println!("{}", "-".repeat(62));

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
        naive_times.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let naive_median = naive_times[n_samples / 2];

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
        unblocked_times.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let unblocked_median = unblocked_times[n_samples / 2];

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
        blocked_times.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let blocked_median = blocked_times[n_samples / 2];

        // LU has ~(2/3)n³ flops
        let flops = 2.0 * (n as f64).powi(3) / 3.0;
        let naive_gflops = flops / naive_median / 1e9;
        let unblocked_gflops = flops / unblocked_median / 1e9;
        let blocked_gflops = flops / blocked_median / 1e9;
        let speedup = unblocked_median / blocked_median;

        println!(
            "{:>8} {:>9.2} G/s {:>9.2} G/s {:>9.2} G/s {:>9.2}x",
            label, naive_gflops, unblocked_gflops, blocked_gflops, speedup
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
    println!("Unblocked: With pivoting, row-by-row updates");
    println!("Blocked: With pivoting, panel + GEMM updates (nb=64)");

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
