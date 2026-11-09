//! Tests of the secular-equation solver of the divide-and-conquer merge on
//! hard cases (roots next to poles with tiny weights, tightly clustered
//! poles, tiny `ρ`, huge dynamic range, the last root with a heavy
//! neighbouring pole), against a double-double reference.

use super::*;

/// Double-double number `hi + lo` (`|lo| <= ulp(hi) / 2`), enough for a
/// reference root about 32 digits deep.
#[derive(Clone, Copy, Debug)]
struct Dd {
    hi: f64,
    lo: f64,
}

/// `a + b` exactly, as an unnormalised pair.
fn two_sum(a: f64, b: f64) -> Dd {
    let s = a + b;
    let bb = s - a;
    Dd {
        hi: s,
        lo: (a - (s - bb)) + (b - bb),
    }
}

/// `a + b` exactly for `|a| >= |b|`.
fn quick_two_sum(a: f64, b: f64) -> Dd {
    let s = a + b;
    Dd {
        hi: s,
        lo: b - (s - a),
    }
}

impl Dd {
    fn from(x: f64) -> Self {
        Self { hi: x, lo: 0.0 }
    }

    fn add(self, o: Self) -> Self {
        let s = two_sum(self.hi, o.hi);
        let t = two_sum(self.lo, o.lo);
        let u = quick_two_sum(s.hi, s.lo + t.hi);
        quick_two_sum(u.hi, u.lo + t.lo)
    }

    fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }

    fn sub(self, o: Self) -> Self {
        self.add(o.neg())
    }

    fn mul_f64(self, b: f64) -> Self {
        let p = self.hi * b;
        let e = self.hi.mul_add(b, -p);
        quick_two_sum(p, e + self.lo * b)
    }

    fn div(self, o: Self) -> Self {
        let q1 = self.hi / o.hi;
        let r = self.sub(o.mul_f64(q1));
        let q2 = r.hi / o.hi;
        let r = r.sub(o.mul_f64(q2));
        let q3 = r.hi / o.hi;
        quick_two_sum(q1, q2).add(Self::from(q3))
    }

    fn half(self) -> Self {
        Self {
            hi: self.hi * 0.5,
            lo: self.lo * 0.5,
        }
    }

    fn to_f64(self) -> f64 {
        self.hi + self.lo
    }
}

/// `1 + Σ ζ_j / ((d_j - d_o) - η)` in double-double.
fn secular_dd(d: &[f64], zeta: &[f64], o: usize, eta: Dd) -> Dd {
    let mut f = Dd::from(1.0);
    for (&dj, &zj) in d.iter().zip(zeta) {
        let delta = two_sum(dj, -d[o]).sub(eta);
        f = f.add(Dd::from(zj).div(delta));
    }
    f
}

/// Reference `d_j - λ_i` for every `j` (double-double bisection on
/// `η = λ_i - d_o` from the nearer pole `d_o`).
fn reference_deltas(d: &[f64], zeta: &[f64], i: usize) -> Vec<f64> {
    let k = d.len();
    let sum: f64 = zeta.iter().sum();
    let (o, mut lo, mut hi) = if i + 1 < k {
        let gap = two_sum(d[i + 1], -d[i]);
        if secular_dd(d, zeta, i, gap.half()).hi >= 0.0 {
            (i, Dd::from(0.0), gap.half())
        } else {
            (i + 1, gap.half().neg(), Dd::from(0.0))
        }
    } else {
        (k - 1, Dd::from(0.0), Dd::from(sum * 2.0))
    };
    for _ in 0..400 {
        let mid = lo.add(hi).half();
        let width = hi.sub(lo).to_f64();
        if width <= 1e-33 * mid.to_f64().abs() || width == 0.0 {
            break;
        }
        if secular_dd(d, zeta, o, mid).hi >= 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let eta = lo.add(hi).half();
    d.iter()
        .map(|&dj| two_sum(dj, -d[o]).sub(eta).to_f64())
        .collect()
}

/// Solves every root of the secular equation `(d, zeta)` and checks the
/// differences `d_j - λ_i` against the reference to `tol` relative
/// accuracy (the eigenvectors are built from them).
fn check_all_roots(d: &[f64], zeta: &[f64], tol: f64, label: &str) {
    let k = d.len();
    assert!(
        d.windows(2).all(|w| w[0] < w[1]),
        "{label}: poles must ascend"
    );
    let sum: f64 = zeta.iter().sum();
    let mut delta = vec![0.0f64; k];
    for i in 0..k {
        let lambda = secular_root(d, zeta, sum, i, &mut delta)
            .unwrap_or_else(|e| panic!("{label}: root {i}: {e:?}"));
        let want = reference_deltas(d, zeta, i);
        for j in 0..k {
            let err = (delta[j] - want[j]).abs();
            assert!(
                err <= tol * want[j].abs(),
                "{label}: root {i}: d[{j}] - λ = {:e}, reference {:e} (rel. err {:e})",
                delta[j],
                want[j],
                err / want[j].abs()
            );
        }
        // λ itself is the rounded `d_o + η`.
        let lo = if i == 0 { f64::NEG_INFINITY } else { d[i] };
        assert!(lambda >= lo, "{label}: root {i} = {lambda} below its pole");
    }
}

/// Deterministic split-mix generator in `[0, 1)`.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

const EPS: f64 = f64::EPSILON;

/// Roots within `1e-25`..`1e-40` of their poles (tiny weights between
/// ordinary ones): from the middle of the gap a starting point would need
/// about a hundred bisection steps to get there.
#[test]
fn secular_roots_next_to_poles_with_tiny_weights() {
    let d = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let zeta = [1e-40, 0.5, 1e-30, 0.3, 1e-25, 0.2];
    check_all_roots(&d, &zeta, 64.0 * EPS * d.len() as f64, "tiny weights");
    let zeta = [0.5, 1e-38, 0.25, 1e-33, 0.25, 1e-36];
    check_all_roots(&d, &zeta, 64.0 * EPS * d.len() as f64, "tiny weights 2");
}

/// Poles 2^-46 apart (a cluster of twenty) next to well separated ones.
#[test]
fn secular_roots_tightly_clustered_poles() {
    let mut d: Vec<f64> = (0..20).map(|j| 0.5 + j as f64 * 2f64.powi(-46)).collect();
    d.extend([0.75, 1.0, 1.5]);
    let k = d.len();
    let zeta: Vec<f64> = (0..k)
        .map(|j| 0.5 / k as f64 * (1.0 + j as f64 / k as f64))
        .collect();
    check_all_roots(&d, &zeta, 64.0 * EPS * k as f64, "cluster");
}

/// A tiny `ρ`: every weight about 1e-28, so every root sits right next to
/// its pole.
#[test]
fn secular_roots_tiny_rho() {
    let k = 40;
    let d: Vec<f64> = (0..k).map(|j| -1.0 + 2.0 * j as f64 / k as f64).collect();
    let zeta: Vec<f64> = (0..k).map(|j| 1e-28 * (1.0 + (j % 7) as f64)).collect();
    check_all_roots(&d, &zeta, 64.0 * EPS * k as f64, "tiny rho");
}

/// Poles spread over 18 orders of magnitude and weights over 30.
#[test]
fn secular_roots_huge_dynamic_range() {
    let k: i32 = 31;
    let d: Vec<f64> = (0..k).map(|j| 2f64.powi(j * 2 - 60)).collect();
    let zeta: Vec<f64> = (0..k).map(|j| 10f64.powi(-((j * 7) % 31))).collect();
    check_all_roots(&d, &zeta, 64.0 * EPS * f64::from(k), "dynamic range");
}

/// The last root next to a heavy, close neighbouring pole (the one-pole
/// model has no root to the right there: `c0 <= 0`).
#[test]
fn secular_last_root_with_heavy_neighbour() {
    let d = [-1.0, -1e-3, 0.0];
    for &(z1, z2) in &[(3.0, 1e-12), (3.0, 1e-30), (0.9, 1e-3), (10.0, 1e-20)] {
        let zeta = [0.5, z1, z2];
        check_all_roots(
            &d,
            &zeta,
            64.0 * EPS * 3.0,
            &format!("last root z1={z1} z2={z2}"),
        );
    }
}

/// Random problems with log-uniform weights between 1e-30 and 1 and poles
/// with gaps between 1e-13 and 1.
#[test]
fn secular_roots_random_hard_problems() {
    let mut rng = Rng(0xD1A6_0001);
    for case in 0..40 {
        let k = 2 + (case * 7) % 37;
        let mut d = Vec::with_capacity(k);
        let mut x = -1.0f64;
        for _ in 0..k {
            d.push(x);
            // At least 14 ulps apart for the |x| < 40 reached here.
            x += 10f64.powf(-13.0 * rng.next());
        }
        let zeta: Vec<f64> = (0..k).map(|_| 10f64.powf(-30.0 * rng.next())).collect();
        check_all_roots(
            &d,
            &zeta,
            64.0 * EPS * k as f64,
            &format!("random case {case}"),
        );
    }
}

/// Two poles: the closed form of the quadratic, both roots.
#[test]
fn secular_roots_two_poles() {
    for &(gap, z0, z1) in &[
        (1.0, 0.5, 0.5),
        (1e-12, 0.3, 0.7),
        (2.0, 1e-35, 1.0),
        (0.5, 1.0, 1e-35),
    ] {
        let d = [0.25, 0.25 + gap];
        check_all_roots(
            &d,
            &[z0, z1],
            64.0 * EPS,
            &format!("two poles {gap} {z0} {z1}"),
        );
    }
}

/// Sets the iteration limit of [`secular_root`] on this thread and restores
/// the default when dropped (also when the test fails).
struct IterLimit;

impl IterLimit {
    fn set(limit: usize) -> Self {
        SECULAR_ITER_LIMIT.with(|c| c.set(limit));
        Self
    }
}

impl Drop for IterLimit {
    fn drop(&mut self) {
        SECULAR_ITER_LIMIT.with(|c| c.set(MAX_SECULAR_ITER));
    }
}

/// An iteration that does not converge within the limit is an error, not a
/// silently accepted iterate.
#[test]
fn secular_root_reports_non_convergence() {
    let d = [0.0, 1.0, 2.0];
    let zeta = [0.5, 0.3, 0.2];
    let mut delta = [0.0; 3];
    {
        let _limit = IterLimit::set(0);
        for i in 0..3 {
            let r = secular_root(&d, &zeta, 1.0, i, &mut delta);
            assert_eq!(r, Err(DcError::SecularEquationFailed), "root {i}");
        }
    }
    // The default limit solves the same equation.
    for i in 0..3 {
        assert!(
            secular_root(&d, &zeta, 1.0, i, &mut delta).is_ok(),
            "root {i}"
        );
    }
    // A NaN weight is reported, not accepted as a converged root.
    let nan_zeta = [0.5, f64::NAN, 0.2];
    for i in 0..3 {
        let r = secular_root(&d, &nan_zeta, f64::NAN, i, &mut delta);
        assert_eq!(r, Err(DcError::SecularEquationFailed), "NaN, root {i}");
    }
}

/// The non-convergence error reaches the public driver.
#[test]
fn secular_failure_propagates_to_symmetric_evd_dc() {
    use crate::evd::{SymmetricEvdDc, SymmetricEvdDcError};
    let n = 64;
    let mut rng = Rng(0xD1A6_0002);
    let mut a = Mat::zeros(n, n);
    for j in 0..n {
        for i in 0..=j {
            let v = rng.next() - 0.5;
            a[(i, j)] = v;
            a[(j, i)] = v;
        }
    }
    let failed = {
        let _limit = IterLimit::set(0);
        SymmetricEvdDc::<f64>::compute(a.as_ref()).err()
    };
    assert_eq!(failed, Some(SymmetricEvdDcError::SecularEquationFailed));
    assert!(SymmetricEvdDc::<f64>::compute(a.as_ref()).is_ok());
}
