//! OxiBLAS Performance Regression CLI
//!
//! Usage:
//!   regress capture \[--output \<file\>\]            # Capture current performance as baseline
//!   regress check \[--baseline \<file\>\] \[--threshold \<pct\>\]  # Compare vs baseline
//!   regress report \[--baseline \<file\>\]           # Print human-readable performance report
//!   regress list                                 # List available benchmark names

use oxiblas_benchmarks::quick_perf;
use oxiblas_benchmarks::regression::{PerfBaseline, RegressionChecker};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_BASELINE: &str = "baseline.json";
const DEFAULT_THRESHOLD: f64 = 5.0;

// ---------------------------------------------------------------------------
// Argument helpers
// ---------------------------------------------------------------------------

/// Extract the value of a named flag from the argument slice.
///
/// For a flag like `--output baseline.json` this returns `Some("baseline.json")`.
fn flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].as_str())
}

/// Parse a float argument value, falling back to `default` on failure.
fn parse_f64(s: &str, default: f64) -> f64 {
    s.parse::<f64>().unwrap_or(default)
}

// ---------------------------------------------------------------------------
// Subcommand: capture
// ---------------------------------------------------------------------------

/// Run the full benchmark suite and save results as a JSON baseline.
fn cmd_capture(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let output_path = PathBuf::from(flag_value(args, "--output").unwrap_or(DEFAULT_BASELINE));

    println!("OxiBLAS Performance Capture");
    println!("===========================");
    println!("Running benchmark suite…");

    let baseline = quick_perf::benchmark_suite();

    println!("  version  : {}", baseline.version);
    println!("  platform : {}", baseline.platform);
    println!("  timestamp: {}", baseline.timestamp);
    println!();

    for m in &baseline.measurements {
        println!(
            "  {:40} {:8.3} GFLOP/s  ({:.3} ms mean)",
            m.name,
            m.throughput_gflops,
            m.mean_ns / 1_000_000.0,
        );
    }

    println!();
    RegressionChecker::save_baseline(&baseline, &output_path)?;
    println!("Baseline saved to: {}", output_path.display());

    Ok(())
}

// ---------------------------------------------------------------------------
// Subcommand: check
// ---------------------------------------------------------------------------

/// Load a baseline, run current benchmarks, and report PASS/FAIL per operation.
fn cmd_check(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let baseline_path = PathBuf::from(flag_value(args, "--baseline").unwrap_or(DEFAULT_BASELINE));
    let threshold_str = flag_value(args, "--threshold").unwrap_or("");
    let threshold = if threshold_str.is_empty() {
        DEFAULT_THRESHOLD
    } else {
        parse_f64(threshold_str, DEFAULT_THRESHOLD)
    };

    println!("OxiBLAS Performance Regression Check");
    println!("=====================================");
    println!("Baseline : {}", baseline_path.display());
    println!("Threshold: {:.1}%", threshold);
    println!();

    let baseline = load_baseline_or_error(&baseline_path)?;

    println!("Running current benchmarks…");
    let current = quick_perf::benchmark_suite();
    println!();

    let checker = RegressionChecker::new(threshold);
    let regressions = checker.check(&baseline, &current);

    // Build a lookup of current measurements for display.
    let mut all_pass = true;

    for base_m in &baseline.measurements {
        let cur_m = current.measurements.iter().find(|m| m.name == base_m.name);

        match cur_m {
            None => {
                println!("  [SKIP] {:40}  (not found in current run)", base_m.name);
            }
            Some(cur) => {
                let is_regression = regressions.iter().any(|r| r.name == base_m.name);
                let change_pct = (cur.throughput_gflops - base_m.throughput_gflops)
                    / base_m.throughput_gflops.max(f64::EPSILON)
                    * 100.0;

                if is_regression {
                    all_pass = false;
                    println!(
                        "  [FAIL] {:40}  baseline={:.3} current={:.3} GFLOP/s  ({:+.1}%)",
                        base_m.name, base_m.throughput_gflops, cur.throughput_gflops, change_pct,
                    );
                } else {
                    println!(
                        "  [PASS] {:40}  baseline={:.3} current={:.3} GFLOP/s  ({:+.1}%)",
                        base_m.name, base_m.throughput_gflops, cur.throughput_gflops, change_pct,
                    );
                }
            }
        }
    }

    println!();

    if all_pass {
        println!("Result: ALL PASS — no regressions detected.");
        Ok(())
    } else {
        let n = regressions.len();
        eprintln!(
            "Result: REGRESSION DETECTED — {} operation(s) degraded by more than {:.1}%.",
            n, threshold
        );
        // Return an error so the process exits with a non-zero status.
        Err(format!("{} regression(s) detected", n).into())
    }
}

// ---------------------------------------------------------------------------
// Subcommand: report
// ---------------------------------------------------------------------------

/// Load a baseline and print a formatted performance table.
fn cmd_report(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let baseline_path = PathBuf::from(flag_value(args, "--baseline").unwrap_or(DEFAULT_BASELINE));

    let baseline = load_baseline_or_error(&baseline_path)?;

    println!("OxiBLAS Performance Report");
    println!("==========================");
    println!("File     : {}", baseline_path.display());
    println!("Version  : {}", baseline.version);
    println!("Platform : {}", baseline.platform);
    println!("Captured : {}", baseline.timestamp);
    println!();

    // Header
    println!(
        "{:<42} {:>8}  {:>12}  {:>10}  {:>5}",
        "Benchmark", "GFLOP/s", "Mean (ms)", "StdDev (ms)", "Size"
    );
    println!("{}", "-".repeat(82));

    for m in &baseline.measurements {
        println!(
            "{:<42} {:>8.3}  {:>12.4}  {:>11.4}  {:>5}",
            m.name,
            m.throughput_gflops,
            m.mean_ns / 1_000_000.0,
            m.std_dev_ns / 1_000_000.0,
            m.matrix_size,
        );
    }

    println!("{}", "-".repeat(82));
    println!("{} measurements total.", baseline.measurements.len());

    Ok(())
}

// ---------------------------------------------------------------------------
// Subcommand: list
// ---------------------------------------------------------------------------

/// Print the names of all benchmarks available in the quick suite.
fn cmd_list() {
    println!("Available OxiBLAS benchmark operations:");
    println!();

    // The quick suite uses fixed sizes [64, 128] × three operations.
    let sizes: &[usize] = &[64, 128];
    let ops = [
        ("DGEMM", "gemm_f64", "f64"),
        ("SGEMM", "gemm_f32", "f32"),
        ("Cholesky", "cholesky_f64", "f64"),
    ];

    println!("{:<44}  {:<8}  FLOP formula", "Name", "Type");
    println!("{}", "-".repeat(72));

    for &n in sizes {
        for (label, prefix, dtype) in &ops {
            let name = format!("{prefix}_{n}x{n}");
            let formula = if *prefix == "cholesky_f64" {
                format!("n^3 / 3  (n={n})")
            } else {
                format!("2 * n^3  (n={n})")
            };
            println!("{:<44}  {:<8}  {}  [{}]", name, dtype, formula, label);
        }
    }

    println!();
    println!("Total: {} benchmarks", sizes.len() * ops.len());
}

// ---------------------------------------------------------------------------
// Usage / help
// ---------------------------------------------------------------------------

fn print_usage() {
    eprintln!(
        "OxiBLAS Performance Regression CLI

USAGE:
    regress <SUBCOMMAND> [OPTIONS]

SUBCOMMANDS:
    capture   Run benchmarks and save results as a JSON baseline
    check     Compare current performance against a saved baseline
    report    Print a formatted table from a saved baseline
    list      List available benchmark names

OPTIONS:
    --output <file>      Output path for 'capture'  [default: baseline.json]
    --baseline <file>    Input baseline for 'check' / 'report'  [default: baseline.json]
    --threshold <pct>    Regression threshold percentage for 'check'  [default: 5.0]

EXAMPLES:
    regress capture --output my_baseline.json
    regress check --baseline my_baseline.json --threshold 3.0
    regress report --baseline my_baseline.json
    regress list
"
    );
}

// ---------------------------------------------------------------------------
// Internal utilities
// ---------------------------------------------------------------------------

/// Load a baseline, providing a clear error message if the file is missing.
fn load_baseline_or_error(path: &Path) -> Result<PerfBaseline, Box<dyn std::error::Error>> {
    if !path.exists() {
        return Err(format!(
            "Baseline file not found: {}\n\
             Run 'regress capture --output {}' first.",
            path.display(),
            path.display()
        )
        .into());
    }
    let baseline = RegressionChecker::load_baseline(path)?;
    Ok(baseline)
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("capture") => cmd_capture(&args[2..])?,
        Some("check") => cmd_check(&args[2..])?,
        Some("report") => cmd_report(&args[2..])?,
        Some("list") => cmd_list(),
        Some("--help") | Some("-h") | Some("help") => print_usage(),
        Some(unknown) => {
            eprintln!("Unknown subcommand: '{unknown}'");
            eprintln!();
            print_usage();
            std::process::exit(1);
        }
        None => {
            print_usage();
            std::process::exit(1);
        }
    }

    Ok(())
}
