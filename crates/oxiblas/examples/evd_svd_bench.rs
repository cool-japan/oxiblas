//! Eigenvalue / SVD benchmark and accuracy harness.
//!
//! Times `SymmetricEvd`, `SymmetricEvdDc`, `Svd` (one-sided Jacobi) and `SvdDc`
//! on deterministic pseudo-random f64 matrices and reports accuracy residuals.
//!
//! Usage:
//!
//! ```text
//! cargo run --release -p oxiblas --example evd_svd_bench -- [--reps N] [SIZE ...]
//! ```
//!
//! - `SIZE ...` defaults to `200 500 1000 2000`.
//! - The repetition count defaults to 3; override it with `--reps N` or the
//!   `OXIBLAS_BENCH_REPS` environment variable. The median time is reported.
//!
//! Inputs: a fixed-seed LCG fills a square matrix `R`. The EVD routines get the
//! symmetric `A = (R + R^T) / 2`, the SVD routines get `R` itself.
//!
//! Accuracy columns:
//! - EVD: `||A V - V L||_F / ||A||_F` (resid) and `||V^T V - I||_F` (orth).
//! - SVD: `||A - U S V^T||_F / ||A||_F` (resid), `||U^T U - I||_F` (orthU) and
//!   `||V^T V - I||_F` (orthV).

use oxiblas::prelude::*;
use std::time::Instant;

/// Fixed seed so every run (and every build) sees the same matrices.
const SEED: u64 = 0x2545_F491_4F6C_DD1D;
const DEFAULT_SIZES: [usize; 4] = [200, 500, 1000, 2000];
const DEFAULT_REPS: usize = 3;

type BenchResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Minimal 64-bit LCG (Knuth MMIX constants); top bits mapped to `[-1, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // 53 high-quality bits -> [0, 1) -> [-1, 1).
        let unit = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        2.0 * unit - 1.0
    }
}

fn random_square(n: usize, rng: &mut Lcg) -> Mat<f64> {
    let mut r = Mat::<f64>::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            r[(i, j)] = rng.next_f64();
        }
    }
    r
}

fn symmetrize(r: &Mat<f64>) -> Mat<f64> {
    let n = r.nrows();
    let mut a = Mat::<f64>::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            a[(i, j)] = 0.5 * (r[(i, j)] + r[(j, i)]);
        }
    }
    a
}

fn transpose(a: MatRef<'_, f64>) -> Mat<f64> {
    let mut t = Mat::<f64>::zeros(a.ncols(), a.nrows());
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            t[(j, i)] = a[(i, j)];
        }
    }
    t
}

fn matmul(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Mat<f64> {
    let mut c = Mat::<f64>::zeros(a.nrows(), b.ncols());
    gemm(1.0, a, b, 0.0, c.as_mut());
    c
}

fn frobenius(a: MatRef<'_, f64>) -> f64 {
    let mut sum = 0.0;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            let v = a[(i, j)];
            sum += v * v;
        }
    }
    sum.sqrt()
}

/// `||M^T M - I||_F` for a square-column matrix `m`.
fn orthogonality_error(m: MatRef<'_, f64>) -> f64 {
    let mt = transpose(m);
    let mut g = matmul(mt.as_ref(), m);
    for i in 0..g.nrows() {
        g[(i, i)] -= 1.0;
    }
    frobenius(g.as_ref())
}

/// `||X - Y||_F`.
fn diff_norm(x: MatRef<'_, f64>, y: MatRef<'_, f64>) -> f64 {
    let mut sum = 0.0;
    for j in 0..x.ncols() {
        for i in 0..x.nrows() {
            let d = x[(i, j)] - y[(i, j)];
            sum += d * d;
        }
    }
    sum.sqrt()
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.total_cmp(b));
    let mid = samples.len() / 2;
    match samples.len() {
        0 => f64::NAN,
        len if len % 2 == 1 => samples[mid],
        _ => 0.5 * (samples[mid - 1] + samples[mid]),
    }
}

/// Runs `f` `reps` times; returns the median seconds and the last result.
fn time_median<T, E, F>(reps: usize, mut f: F) -> BenchResult<(f64, T)>
where
    E: std::error::Error + 'static,
    F: FnMut() -> Result<T, E>,
{
    let mut times = Vec::with_capacity(reps);
    let mut last = None;
    for _ in 0..reps {
        let start = Instant::now();
        let out = f()?;
        times.push(start.elapsed().as_secs_f64());
        last = Some(out);
    }
    let value = last.ok_or("repetition count must be at least 1")?;
    Ok((median(&mut times), value))
}

struct EvdRow {
    secs: f64,
    resid: f64,
    orth: f64,
}

struct SvdRow {
    secs: f64,
    resid: f64,
    orth_u: f64,
    orth_v: f64,
}

/// Accuracy of an eigen-decomposition `(lambda, v)` of `a`.
fn evd_accuracy(a: &Mat<f64>, lambda: &[f64], v: MatRef<'_, f64>) -> (f64, f64) {
    let n = a.nrows();
    let av = matmul(a.as_ref(), v);
    let mut vl = Mat::<f64>::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            vl[(i, j)] = v[(i, j)] * lambda[j];
        }
    }
    let resid = diff_norm(av.as_ref(), vl.as_ref()) / frobenius(a.as_ref());
    (resid, orthogonality_error(v))
}

/// Accuracy of an SVD `(u, sigma, vt)` of `a`.
fn svd_accuracy(
    a: &Mat<f64>,
    u: MatRef<'_, f64>,
    sigma: &[f64],
    vt: MatRef<'_, f64>,
) -> (f64, f64, f64) {
    let n = a.nrows();
    let mut us = Mat::<f64>::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            us[(i, j)] = u[(i, j)] * sigma[j];
        }
    }
    let rec = matmul(us.as_ref(), vt);
    let resid = diff_norm(a.as_ref(), rec.as_ref()) / frobenius(a.as_ref());
    let v = transpose(vt);
    (
        resid,
        orthogonality_error(u),
        orthogonality_error(v.as_ref()),
    )
}

fn bench_evd(a: &Mat<f64>, reps: usize) -> BenchResult<(EvdRow, EvdRow)> {
    let (secs, evd) = time_median(reps, || SymmetricEvd::compute(a.as_ref()))?;
    let (resid, orth) = evd_accuracy(a, evd.eigenvalues(), evd.eigenvectors());
    let plain = EvdRow { secs, resid, orth };

    let (secs, evd) = time_median(reps, || SymmetricEvdDc::compute(a.as_ref()))?;
    let (resid, orth) = evd_accuracy(a, evd.eigenvalues(), evd.eigenvectors());
    let dc = EvdRow { secs, resid, orth };
    Ok((plain, dc))
}

fn bench_svd(r: &Mat<f64>, reps: usize) -> BenchResult<(SvdRow, SvdRow)> {
    let (secs, svd) = time_median(reps, || Svd::compute(r.as_ref()))?;
    let (resid, orth_u, orth_v) = svd_accuracy(r, svd.u(), svd.singular_values(), svd.vt());
    let jacobi = SvdRow {
        secs,
        resid,
        orth_u,
        orth_v,
    };

    let (secs, svd) = time_median(reps, || SvdDc::compute(r.as_ref()))?;
    let (resid, orth_u, orth_v) = svd_accuracy(r, svd.u(), svd.singular_values(), svd.vt());
    let dc = SvdRow {
        secs,
        resid,
        orth_u,
        orth_v,
    };
    Ok((jacobi, dc))
}

fn parse_args() -> BenchResult<(Vec<usize>, usize)> {
    let mut reps = match std::env::var("OXIBLAS_BENCH_REPS") {
        Ok(text) => text.trim().parse::<usize>()?,
        Err(_) => DEFAULT_REPS,
    };
    let mut sizes = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--reps" {
            let value = args.next().ok_or("--reps requires a value")?;
            reps = value.parse::<usize>()?;
        } else if let Some(value) = arg.strip_prefix("--reps=") {
            reps = value.parse::<usize>()?;
        } else {
            sizes.push(arg.parse::<usize>()?);
        }
    }
    if sizes.is_empty() {
        sizes.extend_from_slice(&DEFAULT_SIZES);
    }
    if reps == 0 {
        return Err("repetition count must be at least 1".into());
    }
    if sizes.contains(&0) {
        return Err("sizes must be positive".into());
    }
    Ok((sizes, reps))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (sizes, reps) = parse_args()?;
    println!("EVD / SVD benchmark (f64, square, median of {reps} run(s), seed {SEED:#x})");

    let mut evd_rows = Vec::new();
    let mut svd_rows = Vec::new();
    for &n in &sizes {
        let mut rng = Lcg(SEED ^ n as u64);
        let r = random_square(n, &mut rng);
        let a = symmetrize(&r);
        eprintln!("running n = {n} ...");
        evd_rows.push((n, bench_evd(&a, reps)?));
        svd_rows.push((n, bench_svd(&r, reps)?));
    }

    println!("\n=== Symmetric EVD ===");
    println!(
        "{:>6} {:>12} {:>12} {:>9} {:>10} {:>10} {:>10} {:>10}",
        "n", "Evd [s]", "EvdDc [s]", "Dc/Evd", "resid", "orth", "resid Dc", "orth Dc"
    );
    println!("{}", "-".repeat(87));
    for (n, (plain, dc)) in &evd_rows {
        println!(
            "{:>6} {:>12.4} {:>12.4} {:>9.2} {:>10.2e} {:>10.2e} {:>10.2e} {:>10.2e}",
            n,
            plain.secs,
            dc.secs,
            dc.secs / plain.secs,
            plain.resid,
            plain.orth,
            dc.resid,
            dc.orth
        );
    }

    println!("\n=== SVD ===");
    println!(
        "{:>6} {:>12} {:>12} {:>9} {:>10} {:>10} {:>10}  {:>10} {:>10} {:>10}",
        "n",
        "Svd [s]",
        "SvdDc [s]",
        "Dc/Svd",
        "resid",
        "orthU",
        "orthV",
        "resid Dc",
        "orthU Dc",
        "orthV Dc"
    );
    println!("{}", "-".repeat(113));
    for (n, (jacobi, dc)) in &svd_rows {
        println!(
            "{:>6} {:>12.4} {:>12.4} {:>9.2} {:>10.2e} {:>10.2e} {:>10.2e}  {:>10.2e} {:>10.2e} {:>10.2e}",
            n,
            jacobi.secs,
            dc.secs,
            dc.secs / jacobi.secs,
            jacobi.resid,
            jacobi.orth_u,
            jacobi.orth_v,
            dc.resid,
            dc.orth_u,
            dc.orth_v
        );
    }
    println!("\nDc/Evd and Dc/Svd are time ratios: below 1.0 means divide-and-conquer is faster.");
    Ok(())
}
