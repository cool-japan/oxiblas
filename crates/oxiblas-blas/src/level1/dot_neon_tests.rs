//! Accuracy and routing tests for the NEON dot kernels.
//!
//! # Error bound
//!
//! Each of the 16 (f64) / 32 (f32) lanes accumulates at most `B / LANES = 64`
//! fused products per block, the block partials are added into the outer
//! accumulators (`ceil(n / B)` additions per lane), the tail adds at most one
//! FMA per lane, and the final reduction is a depth-3 pairwise tree, one lane
//! sum and (f64: one, f32: up to three) scalar FMAs. Every computed value is
//! therefore an ordinary floating-point sum with at most
//!
//! `k(n) = 64 + ceil(n / B) + 8`
//!
//! roundings on any path, and the standard bound (Higham, *Accuracy and
//! Stability of Numerical Algorithms*, §3.1) gives
//!
//! `|dot − x·y| ≤ γ_k · Σ|x_i·y_i|`,  `γ_k = k·u / (1 − k·u)`.
//!
//! The tests check this bound against a reference that is exact to within
//! `u·|x·y|` (Ogita–Rump–Oishi `Dot2`, or exact `i128` arithmetic), allowing
//! that extra `u·|x·y|` for the reference itself.

use super::*;

/// Deterministic xorshift generator.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in `[-1, 1)`.
    fn sym(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let z = s - a;
    (s, (a - (s - z)) + (b - z))
}

/// Ogita–Rump–Oishi `Dot2`: as accurate as a dot product evaluated in twice
/// the working precision and rounded once.
fn dot2(x: &[f64], y: &[f64]) -> f64 {
    let (mut p, mut s) = (0.0f64, 0.0f64);
    for (&a, &b) in x.iter().zip(y) {
        let h = a * b;
        let r = a.mul_add(b, -h);
        let (q, e) = two_sum(p, h);
        p = q;
        s += e + r;
    }
    p + s
}

fn abs_dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| (a * b).abs()).sum()
}

fn gamma(k: usize, u: f64) -> f64 {
    let ku = k as f64 * u;
    ku / (1.0 - ku)
}

fn bound_terms(n: usize, block: usize) -> usize {
    64 + n.div_ceil(block) + 8
}

const LENGTHS: [usize; 24] = [
    0, 1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 100, 1023, 1024, 1025, 2047, 2048, 2049, 4111,
    65_537, 100_000, 1_000_003,
];

#[test]
fn test_dot_neon_f64_error_bound_random() {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let u = f64::EPSILON / 2.0;
    for &n in &LENGTHS {
        let x: Vec<f64> = (0..n).map(|_| rng.sym()).collect();
        let y: Vec<f64> = (0..n).map(|_| rng.sym()).collect();
        let got = dot_f64(&x, &y);
        let exact = dot2(&x, &y);
        let tol = gamma(bound_terms(n, F64_BLOCK), u) * abs_dot(&x, &y) + u * exact.abs();
        assert!(
            (got - exact).abs() <= tol,
            "n={n}: got {got:e}, reference {exact:e}, |err| {:e} > {tol:e}",
            (got - exact).abs()
        );
    }
}

#[test]
fn test_dot_neon_f32_error_bound_random() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let u = f64::from(f32::EPSILON) / 2.0;
    for &n in &LENGTHS {
        let x: Vec<f32> = (0..n).map(|_| rng.sym() as f32).collect();
        let y: Vec<f32> = (0..n).map(|_| rng.sym() as f32).collect();
        let got = f64::from(dot_f32(&x, &y));
        // f32 products are exact in f64; Dot2 in f64 is exact far beyond f32.
        let xd: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        let yd: Vec<f64> = y.iter().map(|&v| f64::from(v)).collect();
        let exact = dot2(&xd, &yd);
        let tol = gamma(bound_terms(n, F32_BLOCK), u) * abs_dot(&xd, &yd) + u * exact.abs();
        assert!(
            (got - exact).abs() <= tol,
            "n={n}: got {got:e}, reference {exact:e}, |err| {:e} > {tol:e}",
            (got - exact).abs()
        );
    }
}

/// Integer-valued data scaled by powers of two: every product is exact, the
/// exact sum is computed in `i128`, and the kernel must meet the same bound.
#[test]
fn test_dot_neon_f64_error_bound_against_exact_integers() {
    let mut rng = Rng(0x0DDB_A11C_AFE5_EED5);
    let u = f64::EPSILON / 2.0;
    let scale = 2.0f64.powi(-30);
    for &n in &LENGTHS {
        let ai: Vec<i64> = (0..n)
            .map(|_| (rng.next_u64() >> 38) as i64 - (1 << 25))
            .collect();
        let bi: Vec<i64> = (0..n)
            .map(|_| (rng.next_u64() >> 38) as i64 - (1 << 25))
            .collect();
        let x: Vec<f64> = ai.iter().map(|&a| a as f64 * scale).collect();
        let y: Vec<f64> = bi.iter().map(|&b| b as f64 * scale).collect();
        let exact_int: i128 = ai
            .iter()
            .zip(&bi)
            .map(|(&a, &b)| i128::from(a) * i128::from(b))
            .sum();
        // One rounding into f64, then an exact power-of-two scaling.
        let exact = exact_int as f64 * scale * scale;
        let got = dot_f64(&x, &y);
        let tol = gamma(bound_terms(n, F64_BLOCK), u) * abs_dot(&x, &y) + u * exact.abs();
        assert!(
            (got - exact).abs() <= tol,
            "n={n}: got {got:e}, exact {exact:e}, |err| {:e} > {tol:e}",
            (got - exact).abs()
        );
    }
}

/// Small exact cases through every tail branch, including NaN/Inf propagation.
#[test]
fn test_dot_neon_tails_exact_and_special_values() {
    for n in 0..=80usize {
        let x: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
        let y: Vec<f64> = (0..n).map(|i| (i % 5) as f64 + 0.5).collect();
        let expected: f64 = x.iter().zip(&y).map(|(a, b)| a * b).sum();
        assert_eq!(dot_f64(&x, &y), expected, "f64 n={n}");
        let xs: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let ys: Vec<f32> = y.iter().map(|&v| v as f32).collect();
        assert_eq!(dot_f32(&xs, &ys), expected as f32, "f32 n={n}");
        if n > 0 {
            for p in [0, n / 2, n - 1] {
                let mut xn = x.clone();
                xn[p] = f64::NAN;
                assert!(dot_f64(&xn, &y).is_nan(), "f64 NaN at {p}/{n}");
                let mut xsn = xs.clone();
                xsn[p] = f32::INFINITY;
                let r = dot_f32(&xsn, &ys);
                assert!(r.is_infinite() && r > 0.0, "f32 Inf at {p}/{n}: {r}");
            }
        }
    }
}

/// `dot::<f64>` / `dot::<f32>` and the typed `dot_f64` / `dot_f32` entry
/// points are this kernel, bit for bit, on aarch64.
#[test]
fn test_dot_entry_points_reach_neon_kernel() {
    use crate::level1;
    let mut rng = Rng(7);
    for n in [1usize, 5, 16, 31, 33, 1000, 5000, 70_001] {
        let x: Vec<f64> = (0..n).map(|_| rng.sym()).collect();
        let y: Vec<f64> = (0..n).map(|_| rng.sym()).collect();
        let xs: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let ys: Vec<f32> = y.iter().map(|&v| v as f32).collect();
        let k = dot_f64(&x, &y);
        let k32 = dot_f32(&xs, &ys);
        assert_eq!(
            level1::dot_f64(&x, &y).to_bits(),
            k.to_bits(),
            "dot_f64 n={n}"
        );
        assert_eq!(
            level1::dot_f32(&xs, &ys).to_bits(),
            k32.to_bits(),
            "dot_f32 n={n}"
        );
        #[cfg(not(feature = "force-scalar"))]
        {
            assert_eq!(
                level1::dot(&x, &y).to_bits(),
                k.to_bits(),
                "dot::<f64> n={n}"
            );
            assert_eq!(
                level1::dot(&xs, &ys).to_bits(),
                k32.to_bits(),
                "dot::<f32> n={n}"
            );
        }
    }
}
