//! SYRK/SYR2K benchmark
//!
//! Compares naive vs optimized (GEMM-based) symmetric rank-k and rank-2k updates.

use oxiblas::prelude::*;
use oxiblas_blas::level3::syr2k::syr2k;
use oxiblas_blas::level3::syrk::syrk;
use oxiblas_blas::level3::trsm::{Trans, Uplo};
use std::time::Instant;

/// Naive SYRK for comparison (NoTrans: C = α·A·A^T + β·C)
fn syrk_naive(a: &Mat<f64>, c: &mut Mat<f64>, uplo: Uplo, trans: Trans) {
    let n = c.nrows();
    let k = match trans {
        Trans::NoTrans => a.ncols(),
        Trans::Trans | Trans::ConjTrans => a.nrows(),
    };

    match trans {
        Trans::NoTrans => {
            // C = A·A^T
            match uplo {
                Uplo::Lower => {
                    for j in 0..n {
                        for i in j..n {
                            let mut sum = 0.0;
                            for l in 0..k {
                                sum += a[(i, l)] * a[(j, l)];
                            }
                            c[(i, j)] = sum;
                        }
                    }
                }
                Uplo::Upper => {
                    for j in 0..n {
                        for i in 0..=j {
                            let mut sum = 0.0;
                            for l in 0..k {
                                sum += a[(i, l)] * a[(j, l)];
                            }
                            c[(i, j)] = sum;
                        }
                    }
                }
            }
        }
        Trans::Trans | Trans::ConjTrans => {
            // C = A^T·A
            match uplo {
                Uplo::Lower => {
                    for j in 0..n {
                        for i in j..n {
                            let mut sum = 0.0;
                            for l in 0..k {
                                sum += a[(l, i)] * a[(l, j)];
                            }
                            c[(i, j)] = sum;
                        }
                    }
                }
                Uplo::Upper => {
                    for j in 0..n {
                        for i in 0..=j {
                            let mut sum = 0.0;
                            for l in 0..k {
                                sum += a[(l, i)] * a[(l, j)];
                            }
                            c[(i, j)] = sum;
                        }
                    }
                }
            }
        }
    }
}

/// Naive SYR2K for comparison (NoTrans: C = α·A·B^T + α·B·A^T + β·C)
fn syr2k_naive(a: &Mat<f64>, b: &Mat<f64>, c: &mut Mat<f64>, uplo: Uplo, trans: Trans) {
    let n = c.nrows();
    let k = match trans {
        Trans::NoTrans => a.ncols(),
        Trans::Trans | Trans::ConjTrans => a.nrows(),
    };

    match trans {
        Trans::NoTrans => {
            // C = A·B^T + B·A^T
            match uplo {
                Uplo::Lower => {
                    for j in 0..n {
                        for i in j..n {
                            let mut sum = 0.0;
                            for l in 0..k {
                                sum += a[(i, l)] * b[(j, l)] + b[(i, l)] * a[(j, l)];
                            }
                            c[(i, j)] = sum;
                        }
                    }
                }
                Uplo::Upper => {
                    for j in 0..n {
                        for i in 0..=j {
                            let mut sum = 0.0;
                            for l in 0..k {
                                sum += a[(i, l)] * b[(j, l)] + b[(i, l)] * a[(j, l)];
                            }
                            c[(i, j)] = sum;
                        }
                    }
                }
            }
        }
        Trans::Trans | Trans::ConjTrans => {
            // C = A^T·B + B^T·A
            match uplo {
                Uplo::Lower => {
                    for j in 0..n {
                        for i in j..n {
                            let mut sum = 0.0;
                            for l in 0..k {
                                sum += a[(l, i)] * b[(l, j)] + b[(l, i)] * a[(l, j)];
                            }
                            c[(i, j)] = sum;
                        }
                    }
                }
                Uplo::Upper => {
                    for j in 0..n {
                        for i in 0..=j {
                            let mut sum = 0.0;
                            for l in 0..k {
                                sum += a[(l, i)] * b[(l, j)] + b[(l, i)] * a[(l, j)];
                            }
                            c[(i, j)] = sum;
                        }
                    }
                }
            }
        }
    }
}

fn bench_syrk(sizes: &[(usize, &str)], n_warmup: usize, n_samples: usize) {
    println!("\n=== f64 SYRK (Symmetric Rank-K Update: C = α·A·A^T + β·C) ===");
    println!(
        "{:>8} {:>12} {:>12} {:>12}",
        "Size", "Naive", "Optimized", "Speedup"
    );
    println!("{}", "-".repeat(50));

    for &(n, label) in sizes {
        let k = n; // Square case: A is n×n

        // Create matrix A
        let mut a: Mat<f64> = Mat::zeros(n, k);
        for i in 0..n {
            for j in 0..k {
                a[(i, j)] = 0.001 * (i + j + 1) as f64;
            }
        }

        // Warmup naive
        for _ in 0..n_warmup {
            let mut c: Mat<f64> = Mat::zeros(n, n);
            syrk_naive(&a, &mut c, Uplo::Lower, Trans::NoTrans);
            std::hint::black_box(&c);
        }

        // Measure naive
        let mut naive_times = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let mut c: Mat<f64> = Mat::zeros(n, n);
            let start = Instant::now();
            syrk_naive(&a, &mut c, Uplo::Lower, Trans::NoTrans);
            naive_times.push(start.elapsed().as_secs_f64());
            std::hint::black_box(&c);
        }
        naive_times.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let naive_median = naive_times[n_samples / 2];

        // Warmup optimized
        for _ in 0..n_warmup {
            let mut c: Mat<f64> = Mat::zeros(n, n);
            syrk(
                Uplo::Lower,
                Trans::NoTrans,
                1.0,
                a.as_ref(),
                0.0,
                c.as_mut(),
            )
            .unwrap();
            std::hint::black_box(&c);
        }

        // Measure optimized
        let mut opt_times = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let mut c: Mat<f64> = Mat::zeros(n, n);
            let start = Instant::now();
            syrk(
                Uplo::Lower,
                Trans::NoTrans,
                1.0,
                a.as_ref(),
                0.0,
                c.as_mut(),
            )
            .unwrap();
            opt_times.push(start.elapsed().as_secs_f64());
            std::hint::black_box(&c);
        }
        opt_times.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let opt_median = opt_times[n_samples / 2];

        // SYRK has n²k flops (half of GEMM due to symmetry, but we compute full)
        // Using 2 * n² * k for comparison with GEMM-like operation
        let flops = 2.0 * (n as f64) * (n as f64) * (k as f64);
        let naive_gflops = flops / naive_median / 1e9;
        let opt_gflops = flops / opt_median / 1e9;
        let speedup = naive_median / opt_median;

        println!(
            "{:>8} {:>9.2} G/s {:>9.2} G/s {:>10.2}x",
            label, naive_gflops, opt_gflops, speedup
        );
    }
}

fn bench_syr2k(sizes: &[(usize, &str)], n_warmup: usize, n_samples: usize) {
    println!("\n=== f64 SYR2K (Symmetric Rank-2K Update: C = α·A·B^T + α·B·A^T + β·C) ===");
    println!(
        "{:>8} {:>12} {:>12} {:>12}",
        "Size", "Naive", "Optimized", "Speedup"
    );
    println!("{}", "-".repeat(50));

    for &(n, label) in sizes {
        let k = n; // Square case

        // Create matrices A and B
        let mut a: Mat<f64> = Mat::zeros(n, k);
        let mut b: Mat<f64> = Mat::zeros(n, k);
        for i in 0..n {
            for j in 0..k {
                a[(i, j)] = 0.001 * (i + j + 1) as f64;
                b[(i, j)] = 0.002 * (i + j + 1) as f64;
            }
        }

        // Warmup naive
        for _ in 0..n_warmup {
            let mut c: Mat<f64> = Mat::zeros(n, n);
            syr2k_naive(&a, &b, &mut c, Uplo::Lower, Trans::NoTrans);
            std::hint::black_box(&c);
        }

        // Measure naive
        let mut naive_times = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let mut c: Mat<f64> = Mat::zeros(n, n);
            let start = Instant::now();
            syr2k_naive(&a, &b, &mut c, Uplo::Lower, Trans::NoTrans);
            naive_times.push(start.elapsed().as_secs_f64());
            std::hint::black_box(&c);
        }
        naive_times.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let naive_median = naive_times[n_samples / 2];

        // Warmup optimized
        for _ in 0..n_warmup {
            let mut c: Mat<f64> = Mat::zeros(n, n);
            syr2k(
                Uplo::Lower,
                Trans::NoTrans,
                1.0,
                a.as_ref(),
                b.as_ref(),
                0.0,
                c.as_mut(),
            )
            .unwrap();
            std::hint::black_box(&c);
        }

        // Measure optimized
        let mut opt_times = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let mut c: Mat<f64> = Mat::zeros(n, n);
            let start = Instant::now();
            syr2k(
                Uplo::Lower,
                Trans::NoTrans,
                1.0,
                a.as_ref(),
                b.as_ref(),
                0.0,
                c.as_mut(),
            )
            .unwrap();
            opt_times.push(start.elapsed().as_secs_f64());
            std::hint::black_box(&c);
        }
        opt_times.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let opt_median = opt_times[n_samples / 2];

        // SYR2K has 2×n²k flops (two matrix products)
        let flops = 4.0 * (n as f64) * (n as f64) * (k as f64);
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
    println!("   OxiBLAS SYRK/SYR2K Benchmark");
    println!("==============================================");
    println!();
    println!("SYRK/SYR2K optimized by using optimized GEMM kernel.");
    println!("SYRK: C = α·A·A^T + β·C (or A^T·A variant)");
    println!("SYR2K: C = α·A·B^T + α·B·A^T + β·C (or transposed variant)");

    let sizes = [(128, "128"), (256, "256"), (512, "512"), (1024, "1024")];

    bench_syrk(&sizes, 2, 5);
    bench_syr2k(&sizes, 2, 5);

    println!();
    println!("Note: Flop counts are 2n²k for SYRK, 4n²k for SYR2K.");
}
