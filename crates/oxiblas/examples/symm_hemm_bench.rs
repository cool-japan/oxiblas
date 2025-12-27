//! SYMM/HEMM benchmark
//!
//! Compares naive vs optimized (GEMM-based) symmetric and Hermitian matrix multiplication.

use num_complex::Complex64;
use oxiblas::prelude::*;
use oxiblas_blas::level3::hemm::hemm_c64;
use oxiblas_blas::level3::symm::symm;
use oxiblas_blas::level3::trsm::{Side, Uplo};
use std::time::Instant;

/// Naive SYMM for comparison
fn symm_naive(a: &Mat<f64>, b: &Mat<f64>, c: &mut Mat<f64>, side: Side, uplo: Uplo) {
    let m = c.nrows();
    let n = c.ncols();
    let ka = a.nrows();

    let get_a = |i: usize, j: usize| -> f64 {
        match uplo {
            Uplo::Lower => {
                if i >= j {
                    a[(i, j)]
                } else {
                    a[(j, i)]
                }
            }
            Uplo::Upper => {
                if i <= j {
                    a[(i, j)]
                } else {
                    a[(j, i)]
                }
            }
        }
    };

    match side {
        Side::Left => {
            for j in 0..n {
                for i in 0..m {
                    let mut sum = 0.0;
                    for k in 0..ka {
                        sum += get_a(i, k) * b[(k, j)];
                    }
                    c[(i, j)] = sum;
                }
            }
        }
        Side::Right => {
            for j in 0..n {
                for i in 0..m {
                    let mut sum = 0.0;
                    for k in 0..ka {
                        sum += b[(i, k)] * get_a(k, j);
                    }
                    c[(i, j)] = sum;
                }
            }
        }
    }
}

/// Naive HEMM for comparison
fn hemm_naive(
    a: &Mat<Complex64>,
    b: &Mat<Complex64>,
    c: &mut Mat<Complex64>,
    side: Side,
    uplo: Uplo,
) {
    let m = c.nrows();
    let n = c.ncols();
    let ka = a.nrows();

    let get_a = |i: usize, j: usize| -> Complex64 {
        match uplo {
            Uplo::Lower => {
                if i >= j {
                    a[(i, j)]
                } else {
                    a[(j, i)].conj()
                }
            }
            Uplo::Upper => {
                if i <= j {
                    a[(i, j)]
                } else {
                    a[(j, i)].conj()
                }
            }
        }
    };

    match side {
        Side::Left => {
            for j in 0..n {
                for i in 0..m {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for k in 0..ka {
                        sum += get_a(i, k) * b[(k, j)];
                    }
                    c[(i, j)] = sum;
                }
            }
        }
        Side::Right => {
            for j in 0..n {
                for i in 0..m {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for k in 0..ka {
                        sum += b[(i, k)] * get_a(k, j);
                    }
                    c[(i, j)] = sum;
                }
            }
        }
    }
}

fn bench_symm(sizes: &[(usize, &str)], n_warmup: usize, n_samples: usize) {
    println!("\n=== f64 SYMM (Symmetric Matrix Multiply) ===");
    println!(
        "{:>8} {:>12} {:>12} {:>12}",
        "Size", "Naive", "Optimized", "Speedup"
    );
    println!("{}", "-".repeat(50));

    for &(n, label) in sizes {
        // Create symmetric matrix A (lower triangle)
        let mut a: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                a[(i, j)] = 0.001 * (i + j + 1) as f64;
            }
        }

        // Create general matrix B
        let mut b: Mat<f64> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                b[(i, j)] = 0.002 * (i + j + 1) as f64;
            }
        }

        // Warmup naive
        for _ in 0..n_warmup {
            let mut c: Mat<f64> = Mat::zeros(n, n);
            symm_naive(&a, &b, &mut c, Side::Left, Uplo::Lower);
            std::hint::black_box(&c);
        }

        // Measure naive
        let mut naive_times = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let mut c: Mat<f64> = Mat::zeros(n, n);
            let start = Instant::now();
            symm_naive(&a, &b, &mut c, Side::Left, Uplo::Lower);
            naive_times.push(start.elapsed().as_secs_f64());
            std::hint::black_box(&c);
        }
        naive_times.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let naive_median = naive_times[n_samples / 2];

        // Warmup optimized
        for _ in 0..n_warmup {
            let mut c: Mat<f64> = Mat::zeros(n, n);
            symm(
                Side::Left,
                Uplo::Lower,
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
            symm(
                Side::Left,
                Uplo::Lower,
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

        // SYMM has same flop count as GEMM: 2 * n^3
        let flops = 2.0 * (n as f64).powi(3);
        let naive_gflops = flops / naive_median / 1e9;
        let opt_gflops = flops / opt_median / 1e9;
        let speedup = naive_median / opt_median;

        println!(
            "{:>8} {:>9.2} G/s {:>9.2} G/s {:>10.2}x",
            label, naive_gflops, opt_gflops, speedup
        );
    }
}

fn bench_hemm(sizes: &[(usize, &str)], n_warmup: usize, n_samples: usize) {
    println!("\n=== Complex64 HEMM (Hermitian Matrix Multiply) ===");
    println!(
        "{:>8} {:>12} {:>12} {:>12}",
        "Size", "Naive", "Optimized", "Speedup"
    );
    println!("{}", "-".repeat(50));

    for &(n, label) in sizes {
        // Create Hermitian matrix A (lower triangle)
        let mut a: Mat<Complex64> = Mat::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = Complex64::new((i + 1) as f64, 0.0); // Real diagonal
            for j in 0..i {
                a[(i, j)] = Complex64::new(0.01 * (i + j) as f64, 0.001 * (i * j) as f64);
            }
        }

        // Create general matrix B
        let mut b: Mat<Complex64> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                b[(i, j)] = Complex64::new(0.01 * (i + j + 1) as f64, 0.0);
            }
        }

        // Warmup naive
        for _ in 0..n_warmup {
            let mut c: Mat<Complex64> = Mat::zeros(n, n);
            hemm_naive(&a, &b, &mut c, Side::Left, Uplo::Lower);
            std::hint::black_box(&c);
        }

        // Measure naive
        let mut naive_times = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let mut c: Mat<Complex64> = Mat::zeros(n, n);
            let start = Instant::now();
            hemm_naive(&a, &b, &mut c, Side::Left, Uplo::Lower);
            naive_times.push(start.elapsed().as_secs_f64());
            std::hint::black_box(&c);
        }
        naive_times.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let naive_median = naive_times[n_samples / 2];

        // Warmup optimized
        for _ in 0..n_warmup {
            let mut c: Mat<Complex64> = Mat::zeros(n, n);
            hemm_c64(
                Side::Left,
                Uplo::Lower,
                Complex64::new(1.0, 0.0),
                a.as_ref(),
                b.as_ref(),
                Complex64::new(0.0, 0.0),
                c.as_mut(),
            )
            .unwrap();
            std::hint::black_box(&c);
        }

        // Measure optimized
        let mut opt_times = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let mut c: Mat<Complex64> = Mat::zeros(n, n);
            let start = Instant::now();
            hemm_c64(
                Side::Left,
                Uplo::Lower,
                Complex64::new(1.0, 0.0),
                a.as_ref(),
                b.as_ref(),
                Complex64::new(0.0, 0.0),
                c.as_mut(),
            )
            .unwrap();
            opt_times.push(start.elapsed().as_secs_f64());
            std::hint::black_box(&c);
        }
        opt_times.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let opt_median = opt_times[n_samples / 2];

        // Complex HEMM has 8 real flops per complex multiply-add: 8 * n^3
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
    println!("   OxiBLAS SYMM/HEMM Benchmark");
    println!("==============================================");
    println!();
    println!("SYMM/HEMM optimized by expanding symmetric/Hermitian");
    println!("matrix to full and using optimized GEMM.");

    let sizes = [(128, "128"), (256, "256"), (512, "512"), (1024, "1024")];

    bench_symm(&sizes, 2, 5);
    bench_hemm(&sizes, 2, 5);

    println!();
    println!("Note: SYMM uses 2n³ flops, HEMM uses 8n³ flops (complex).");
}
