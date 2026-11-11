//! Cholesky Factorization benchmark
//!
//! Compares unblocked vs blocked Cholesky factorization performance.

use oxiblas::prelude::*;
use oxiblas_lapack::cholesky::Cholesky;
use oxiblas_lapack::workspace::optimal_block_size_cholesky;
use std::time::Instant;

fn cholesky_naive(a: &Mat<f64>) -> Mat<f64> {
    let n = a.nrows();
    let mut l: Mat<f64> = Mat::zeros(n, n);

    // Copy lower triangular part
    for i in 0..n {
        for j in 0..=i {
            l[(i, j)] = a[(i, j)];
        }
    }

    // Cholesky-Banachiewicz algorithm
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;

            if j == i {
                // Diagonal element
                for k in 0..j {
                    sum += l[(j, k)] * l[(j, k)];
                }
                l[(i, j)] = (l[(i, j)] - sum).sqrt();
            } else {
                // Off-diagonal element
                for k in 0..j {
                    sum += l[(i, k)] * l[(j, k)];
                }
                l[(i, j)] = (l[(i, j)] - sum) / l[(j, j)];
            }
        }
    }
    l
}

/// Median of the samples (sorted in place).
fn median(times: &mut [f64]) -> f64 {
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

/// Create a random-ish SPD matrix
fn create_spd_matrix(n: usize) -> Mat<f64> {
    let mut a: Mat<f64> = Mat::zeros(n, n);

    // Create random matrix
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] = ((i * 17 + j * 31 + 7) % 100) as f64 / 100.0;
        }
    }

    // Make it SPD: B = A^T * A + n*I
    let mut spd: Mat<f64> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[(k, i)] * a[(k, j)];
            }
            spd[(i, j)] = sum;
            if i == j {
                spd[(i, j)] += (n as f64) * 0.5; // Ensure positive definiteness
            }
        }
    }
    spd
}

fn bench_cholesky(sizes: &[(usize, &str)], n_warmup: usize, n_samples: usize) {
    println!("\n=== f64 Cholesky Factorization Benchmark ===");
    println!(
        "{:>8} {:>5} {:>12} {:>12} {:>12} {:>12}",
        "Size", "nb", "Naive", "Unblocked", "Blocked", "Speedup"
    );
    println!("{}", "-".repeat(68));

    for &(n, label) in sizes {
        let a = create_spd_matrix(n);

        // Warmup naive
        for _ in 0..n_warmup {
            let _ = cholesky_naive(&a);
        }

        // Measure naive
        let mut naive_times = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let start = Instant::now();
            let _ = cholesky_naive(&a);
            naive_times.push(start.elapsed().as_secs_f64());
        }
        let naive_median = median(&mut naive_times);

        // Warmup unblocked
        for _ in 0..n_warmup {
            let _ = Cholesky::compute(a.as_ref());
        }

        // Measure unblocked
        let mut unblocked_times = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let start = Instant::now();
            let _ = Cholesky::compute(a.as_ref());
            unblocked_times.push(start.elapsed().as_secs_f64());
        }
        let unblocked_median = median(&mut unblocked_times);

        // Warmup blocked
        for _ in 0..n_warmup {
            let _ = Cholesky::compute_blocked(a.as_ref());
        }

        // Measure blocked
        let mut blocked_times = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let start = Instant::now();
            let _ = Cholesky::compute_blocked(a.as_ref());
            blocked_times.push(start.elapsed().as_secs_f64());
        }
        let blocked_median = median(&mut blocked_times);

        // Cholesky has ~(1/3)n³ flops (half of LU)
        let flops = (n as f64).powi(3) / 3.0;
        let naive_gflops = flops / naive_median / 1e9;
        let unblocked_gflops = flops / unblocked_median / 1e9;
        let blocked_gflops = flops / blocked_median / 1e9;
        let speedup = unblocked_median / blocked_median;

        // The block size `Cholesky::compute_blocked` uses for this size.
        let nb = optimal_block_size_cholesky(n);

        println!(
            "{:>8} {:>5} {:>9.2} G/s {:>9.2} G/s {:>9.2} G/s {:>9.2}x",
            label, nb, naive_gflops, unblocked_gflops, blocked_gflops, speedup
        );
    }
}

fn main() {
    println!("==============================================");
    println!("   OxiBLAS Cholesky Factorization Benchmark");
    println!("==============================================");
    println!();
    println!("Cholesky decomposition: A = LL^T for SPD matrices");
    println!("Blocked algorithm uses TRSM, GEMM and triangle-only SYRK for cache efficiency.");
    println!();
    println!("Naive: Simple O(n³/3) Cholesky-Banachiewicz");
    println!("Unblocked: Same algorithm with tolerance checks");
    println!("Blocked: Diagonal block + TRSM, then GEMM/SYRK updates of the lower triangle,");
    println!("         block size nb from optimal_block_size_cholesky: n if n < 64,");
    println!("         32 if n < 256, 64 if n < 1024, 128 otherwise (nb column below)");

    let sizes = [
        (64, "64"),
        (128, "128"),
        (256, "256"),
        (512, "512"),
        (768, "768"),
        (1024, "1024"),
    ];

    bench_cholesky(&sizes, 2, 5);

    println!();
    println!("Note: Blocked algorithm is most effective for n >= 64.");
    println!("      Cholesky is ~2x faster than LU (half the work).");
}
