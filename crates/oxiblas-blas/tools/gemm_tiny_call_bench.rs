//! Per-call cost of small GEMM products through the public entry points.
//!
//! The benchmark program of `gemm_tiny_call_check.sh`, which copies this file
//! into two generated crates (one against a base revision of oxiblas, one
//! against the tree under test) and compares them. It uses only API that
//! exists in every release since 0.2.3: `level3::gemm::{gemm,
//! gemm_transposed}`, `level3::Trans`, `oxiblas_matrix::Mat` and
//! `oxiblas_core::scalar::Field`.
//!
//! Usage: `<binary> [SAMPLES] [SAMPLE_MS]` (defaults 200 and 0.5). For every
//! case the number of calls per sample is doubled from 64 until one sample
//! takes at least `SAMPLE_MS` milliseconds; then `SAMPLES` samples are timed,
//! each right after one sample of the clock reference ([`reference_sample`]):
//! a chain of dependent `f64` adds, the same code in every build, whose time
//! per add follows only the clock of the core the sample ran on. Output, one
//! line per case:
//! `CASE<TAB>name<TAB>min_ns=<min ns per call><TAB>med_ns=<median>`
//! `<TAB>ref_ns=<min ns per add of the reference samples>`.

use oxiblas_blas::level3::Trans;
use oxiblas_blas::level3::gemm::{gemm, gemm_transposed};
use oxiblas_matrix::Mat;
use std::hint::black_box;
use std::time::Instant;

/// The element types measured, with the two public entry points.
trait Elem: oxiblas_core::scalar::Field {
    /// Short type name for the case names.
    const NAME: &'static str;
    /// `x` converted to this type.
    fn value_of(x: f64) -> Self;
    /// A zero matrix.
    fn zeros(rows: usize, cols: usize) -> Mat<Self>;
    /// One call `C = alpha * op(A) * op(B) + beta * C`; `NoTrans`/`NoTrans`
    /// goes through `gemm`, everything else through `gemm_transposed`.
    fn call(
        ta: Trans,
        tb: Trans,
        alpha: Self,
        a: &Mat<Self>,
        b: &Mat<Self>,
        beta: Self,
        c: &mut Mat<Self>,
    );
}

macro_rules! elem {
    ($t:ty, $name:expr) => {
        impl Elem for $t {
            const NAME: &'static str = $name;
            fn value_of(x: f64) -> Self {
                x as $t
            }
            fn zeros(rows: usize, cols: usize) -> Mat<Self> {
                Mat::zeros(rows, cols)
            }
            fn call(
                ta: Trans,
                tb: Trans,
                alpha: Self,
                a: &Mat<Self>,
                b: &Mat<Self>,
                beta: Self,
                c: &mut Mat<Self>,
            ) {
                if ta == Trans::NoTrans && tb == Trans::NoTrans {
                    gemm(alpha, a.as_ref(), b.as_ref(), beta, c.as_mut());
                } else {
                    gemm_transposed(ta, tb, alpha, a.as_ref(), b.as_ref(), beta, c.as_mut());
                }
            }
        }
    };
}

elem!(f64, "f64");
elem!(f32, "f32");

/// A `rows × cols` matrix with small, deterministic values.
fn filled<T: Elem>(rows: usize, cols: usize, salt: usize) -> Mat<T> {
    let mut m = T::zeros(rows, cols);
    for j in 0..cols {
        for i in 0..rows {
            let x = (i * 7 + j * 13 + salt) % 23;
            m[(i, j)] = T::value_of(x as f64 / 11.0 - 1.0);
        }
    }
    m
}

fn trans_name(t: Trans) -> &'static str {
    if t == Trans::NoTrans { "n" } else { "t" }
}

/// Adds in one sample of the clock reference (about 30 µs at 4 GHz, some 700
/// ticks of the 24 MHz clock of `Instant` on macOS).
const REFERENCE_ADDS: u64 = 40_000;

/// `n` dependent `f64` adds: each add needs the result of the one before, and
/// floating-point adds are not reassociated, so the chain runs at one add per
/// add latency whatever the code placement, and its time follows only the
/// clock of the core.
#[inline(never)]
fn reference_chain(n: u64) -> f64 {
    let step = black_box(1.0e-9_f64);
    let mut x = black_box(0.0_f64);
    for _ in 0..n {
        x += step;
    }
    x
}

/// One sample of the clock reference: the time per add, in ns, of
/// [`REFERENCE_ADDS`] adds of [`reference_chain`].
///
/// Under load the program runs at a lower clock for seconds at a time, or on
/// a slower core, which makes every case timed meanwhile slower. Every sample
/// of a case follows one sample of this reference, so the fastest reference
/// sample and the fastest case sample of a case come from the same stretch of
/// time; the script scales each case to the fastest reference seen in any
/// run of either side.
fn reference_sample() -> f64 {
    let start = Instant::now();
    black_box(reference_chain(black_box(REFERENCE_ADDS)));
    start.elapsed().as_secs_f64() * 1e9 / REFERENCE_ADDS as f64
}

/// Times `C = op(A) * op(B)` for `op(A)` of `m × k` and prints one line.
fn bench<T: Elem>(
    label: &str,
    ta: Trans,
    tb: Trans,
    (m, k, n): (usize, usize, usize),
    samples: usize,
    sample_ns: f64,
) {
    let a: Mat<T> = if ta == Trans::NoTrans {
        filled(m, k, 1)
    } else {
        filled(k, m, 1)
    };
    let b: Mat<T> = if tb == Trans::NoTrans {
        filled(k, n, 5)
    } else {
        filled(n, k, 5)
    };
    let mut c: Mat<T> = T::zeros(m, n);
    let one = T::value_of(1.0);
    let zero = T::value_of(0.0);
    let run = |reps: u64, c: &mut Mat<T>| {
        let start = Instant::now();
        for _ in 0..reps {
            T::call(ta, tb, black_box(one), &a, &b, black_box(zero), c);
        }
        let elapsed = start.elapsed().as_secs_f64() * 1e9;
        black_box(&*c);
        elapsed
    };
    let mut reps: u64 = 64;
    while run(reps, &mut c) < sample_ns && reps < (1 << 40) {
        reps *= 2;
    }
    let mut reference = f64::INFINITY;
    let mut all: Vec<f64> = Vec::with_capacity(samples);
    for _ in 0..samples {
        reference = reference.min(reference_sample());
        all.push(run(reps, &mut c) / reps as f64);
    }
    all.sort_by(f64::total_cmp);
    let name = format!("{}_{}{}_{}", T::NAME, trans_name(ta), trans_name(tb), label);
    println!(
        "CASE\t{name}\tmin_ns={:.3}\tmed_ns={:.3}\tref_ns={:.5}",
        all[0],
        all[all.len() / 2],
        reference
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let samples = args
        .get(1)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(200)
        .max(1);
    let sample_ms = args
        .get(2)
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.5);
    let sample_ns = sample_ms * 1e6;
    let (nn, tn, nt, tt) = (
        (Trans::NoTrans, Trans::NoTrans),
        (Trans::Trans, Trans::NoTrans),
        (Trans::NoTrans, Trans::Trans),
        (Trans::Trans, Trans::Trans),
    );
    // Every cube from 1³ to 16³, all four operand combinations (the script
    // decides which of them it judges).
    for s in 1usize..=16 {
        let label = s.to_string();
        for (ta, tb) in [nn, tn, nt, tt] {
            bench::<f64>(&label, ta, tb, (s, s, s), samples, sample_ns);
            bench::<f32>(&label, ta, tb, (s, s, s), samples, sample_ns);
        }
    }
    // Information: sizes on the blocked route (one thread).
    for s in [48usize, 64, 128] {
        let label = format!("blk{s}");
        bench::<f64>(&label, nn.0, nn.1, (s, s, s), samples, sample_ns);
        bench::<f32>(&label, nn.0, nn.1, (s, s, s), samples, sample_ns);
    }
}
