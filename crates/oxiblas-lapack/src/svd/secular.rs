//! Secular equation of the bidiagonal divide-and-conquer SVD (LAPACK
//! `dlasd4`, `dlasd5` and `dlaed6`).
//!
//! A merge of [`crate::svd::bidiag_dc`] needs the singular values of an
//! "arrow" matrix, i.e. the square roots of the eigenvalues of
//! `diag(d)² + ρ·z·zᵀ` with `0 ≤ d₀ < d₁ < … < d_{n−1}`, `‖z‖ = 1` and `ρ > 0`.
//! The `i`-th root lies in `(dᵢ, dᵢ₊₁)` (the last one in
//! `(d_{n−1}, √(d²_{n−1} + ρ)]`) and solves
//!
//! ```text
//! f(σ) = 1/ρ + Σⱼ zⱼ² / ((dⱼ − σ)(dⱼ + σ)) = 0.
//! ```
//!
//! [`lasd4`] follows LAPACK 3.12 `dlasd4`: the iteration runs in the shifted
//! variable `τ = σ − d_origin` relative to the nearer pole, and keeps
//! `δⱼ = dⱼ − σ` and `ωⱼ = dⱼ + σ` updated incrementally, so every
//! `dⱼ² − σ² = δⱼ·ωⱼ` is available to high relative accuracy without
//! cancellation. Steps come from the rational two-pole model of Li's "middle
//! way" (switching to the fixed-weight model when convergence stalls) or, for
//! interior roots on the far side of the origin pole, from the three-pole
//! model solved by `dlaed6`, safeguarded by a bracket. Two roots use the
//! closed form of `dlasd5`.
//!
//! Differences from the reference: the initial guess of the largest root
//! uses the upper bound `√(d²_{n−1} + ρ) − d_{n−1}` where LAPACK 3.12 assigns
//! `τ = ρ` (an unconverted value of the squared variable; its step guard
//! recovers either way), the largest-root iteration additionally tracks a
//! bracket, and an iteration that runs out of steps (LAPACK returns
//! `INFO = 1`) falls back to bisection on its bracket before giving up.

use oxiblas_core::scalar::{Field, Real};

use super::dc_kernels::{fabs, lapack_eps, real_const};

/// Iteration limit of the rational root finder (`MAXIT` of `dlasd4`).
const MAXIT: usize = 400;
/// Iteration limit of the three-pole solver (`MAXIT` of `dlaed6`).
const LAED6_MAXIT: usize = 40;
/// Bisection steps of the fall-back (enough to resolve the full exponent
/// range of `f64` twice).
const MAX_BISECT: usize = 4300;

#[inline]
fn sqrt<R: Real>(x: R) -> R {
    <R as Real>::sqrt(x)
}

/// Root `i` (`0 ≤ i < n`) of the secular equation for singular values.
///
/// `d` must be strictly increasing with `d[0] ≥ 0`, `‖z‖ = 1` and `rho > 0`.
/// On success returns `σᵢ` and writes `delta[j] = d[j] − σᵢ` and
/// `work[j] = d[j] + σᵢ`, each to high relative accuracy. Returns `None` if
/// the iteration (including the bisection fall-back) produced no finite root.
pub(crate) fn lasd4<R: Field + Real>(
    i: usize,
    d: &[R],
    z: &[R],
    rho: R,
    delta: &mut [R],
    work: &mut [R],
) -> Option<R> {
    let n = d.len();
    let sigma = match n {
        0 => return None,
        1 => {
            delta[0] = R::one();
            work[0] = R::one();
            sqrt(d[0] * d[0] + rho * z[0] * z[0])
        }
        2 => lasd5(i, d, z, rho, delta, work),
        _ if i == n - 1 => lasd4_last(d, z, rho, delta, work)?,
        _ => lasd4_interior(i, d, z, rho, delta, work)?,
    };
    if sigma.is_finite() { Some(sigma) } else { None }
}

/// Closed-form root `i ∈ {0, 1}` of the 2×2 problem (LAPACK `dlasd5`).
fn lasd5<R: Field + Real>(
    i: usize,
    d: &[R],
    z: &[R],
    rho: R,
    delta: &mut [R],
    work: &mut [R],
) -> R {
    let zero = R::zero();
    let one = R::one();
    let two = real_const::<R>(2.0);
    let three = real_const::<R>(3.0);
    let four = real_const::<R>(4.0);
    let del = d[1] - d[0];
    let delsq = del * (d[1] + d[0]);
    let upper = |b: R, c: R| -> R {
        // σ² − d₁² for the root near d₁ (b = −delsq + ρ‖z‖², c = ρ·z₁²·delsq).
        if b > zero {
            -two * c / (b + sqrt(b * b + four * c))
        } else {
            (b - sqrt(b * b + four * c)) / two
        }
    };
    if i == 0 {
        let w = one
            + four
                * rho
                * (z[1] * z[1] / (d[0] + three * d[1]) - z[0] * z[0] / (three * d[0] + d[1]))
                / del;
        if w > zero {
            let b = delsq + rho * (z[0] * z[0] + z[1] * z[1]);
            let c = rho * z[0] * z[0] * delsq;
            // τ = σ² − d₀², then σ − d₀.
            let mut tau = two * c / (b + sqrt(fabs(b * b - four * c)));
            tau = tau / (d[0] + sqrt(d[0] * d[0] + tau));
            delta[0] = -tau;
            delta[1] = del - tau;
            work[0] = two * d[0] + tau;
            work[1] = (d[0] + tau) + d[1];
            d[0] + tau
        } else {
            let b = -delsq + rho * (z[0] * z[0] + z[1] * z[1]);
            let c = rho * z[1] * z[1] * delsq;
            let mut tau = upper(b, c);
            tau = tau / (d[1] + sqrt(fabs(d[1] * d[1] + tau)));
            delta[0] = -(del + tau);
            delta[1] = -tau;
            work[0] = d[0] + tau + d[1];
            work[1] = two * d[1] + tau;
            d[1] + tau
        }
    } else {
        let b = -delsq + rho * (z[0] * z[0] + z[1] * z[1]);
        let c = rho * z[1] * z[1] * delsq;
        let mut tau = if b > zero {
            (b + sqrt(b * b + four * c)) / two
        } else {
            two * c / (-b + sqrt(b * b + four * c))
        };
        tau = tau / (d[1] + sqrt(d[1] * d[1] + tau));
        delta[0] = -(del + tau);
        delta[1] = -tau;
        work[0] = d[0] + tau + d[1];
        work[1] = two * d[1] + tau;
        d[1] + tau
    }
}

/// Partial sums of the secular function around the origin pole `ii`.
struct SplitEval<R> {
    psi: R,
    dpsi: R,
    phi: R,
    dphi: R,
    /// Accumulated part of the rounding-error bound.
    erretm: R,
}

/// `ψ = Σ_{j<ii}`, `φ = Σ_{j>ii}` of `zⱼ²/(ωⱼδⱼ)` and their derivatives.
fn eval_split<R: Field + Real>(ii: usize, z: &[R], delta: &[R], work: &[R]) -> SplitEval<R> {
    let n = z.len();
    let mut psi = R::zero();
    let mut dpsi = R::zero();
    let mut erretm = R::zero();
    for j in 0..ii {
        let temp = z[j] / (work[j] * delta[j]);
        psi = psi + z[j] * temp;
        dpsi = dpsi + temp * temp;
        erretm = erretm + psi;
    }
    erretm = fabs(erretm);
    let mut phi = R::zero();
    let mut dphi = R::zero();
    for j in (ii + 1..n).rev() {
        let temp = z[j] / (work[j] * delta[j]);
        phi = phi + z[j] * temp;
        dphi = dphi + temp * temp;
        erretm = erretm + phi;
    }
    SplitEval {
        psi,
        dpsi,
        phi,
        dphi,
        erretm,
    }
}

/// `τ += η`, `σ += η`, `ωⱼ += η`, `δⱼ −= η`.
#[inline]
fn shift<R: Field + Real>(tau: &mut R, sigma: &mut R, delta: &mut [R], work: &mut [R], eta: R) {
    *tau = *tau + eta;
    *sigma = *sigma + eta;
    for (dj, wj) in delta.iter_mut().zip(work.iter_mut()) {
        *wj = *wj + eta;
        *dj = *dj - eta;
    }
}

/// Bisection fall-back on the bracket `[lo, hi]` of `τ = σ − d_org`:
/// recomputes `δ` and `ω` from `d` at every midpoint. Returns `σ`.
fn bisect<R: Field + Real>(
    d: &[R],
    z: &[R],
    rho: R,
    org: usize,
    mut lo: R,
    mut hi: R,
    delta: &mut [R],
    work: &mut [R],
) -> Option<R> {
    let eps = lapack_eps::<R>();
    let rhoinv = R::one() / rho;
    let two = real_const::<R>(2.0);
    let dorg = d[org];
    let mut tau = (lo + hi) / two;
    for _ in 0..MAX_BISECT {
        if !(lo.is_finite() && hi.is_finite()) {
            return None;
        }
        tau = (lo + hi) / two;
        for j in 0..d.len() {
            delta[j] = (d[j] - dorg) - tau;
            work[j] = d[j] + dorg + tau;
        }
        let mut w = rhoinv;
        let mut erretm = rhoinv;
        for j in 0..d.len() {
            let t = z[j] * z[j] / (work[j] * delta[j]);
            w = w + t;
            erretm = erretm + fabs(t);
        }
        if !(fabs(w) > eps * erretm) || !(tau > lo && tau < hi) {
            break;
        }
        if w <= R::zero() {
            lo = tau;
        } else {
            hi = tau;
        }
    }
    let sigma = dorg + tau;
    if sigma.is_finite() { Some(sigma) } else { None }
}

/// Largest root (`dlasd4`, `I = N` branch).
fn lasd4_last<R: Field + Real>(
    d: &[R],
    z: &[R],
    rho: R,
    delta: &mut [R],
    work: &mut [R],
) -> Option<R> {
    let n = d.len();
    let zero = R::zero();
    let two = real_const::<R>(2.0);
    let four = real_const::<R>(4.0);
    let eight = real_const::<R>(8.0);
    let eps = lapack_eps::<R>();
    let rhoinv = R::one() / rho;
    let last = n - 1;
    let prev = n - 2;
    let dn = d[last];

    // Initial guess from the midpoint σ² = d²_{n−1} + ρ/2.
    let temp = rho / two;
    let temp1 = temp / (dn + sqrt(dn * dn + temp));
    for j in 0..n {
        work[j] = d[j] + dn + temp1;
        delta[j] = (d[j] - dn) - temp1;
    }
    let mut psi = zero;
    for j in 0..prev {
        psi = psi + z[j] * z[j] / (delta[j] * work[j]);
    }
    let c = rhoinv + psi;
    let w = c
        + z[prev] * z[prev] / (delta[prev] * work[prev])
        + z[last] * z[last] / (delta[last] * work[last]);
    // Bracket of τ = σ − d_{n−1}.
    let upper = rho / (dn + sqrt(dn * dn + rho));
    let (mut lo, mut hi) = if w <= zero {
        (temp1, upper)
    } else {
        (zero, temp1)
    };
    let quadratic = |c: R| {
        let delsq = (dn - d[prev]) * (dn + d[prev]);
        let a = -c * delsq + z[prev] * z[prev] + z[last] * z[last];
        let b = z[last] * z[last] * delsq;
        let tau2 = if a < zero {
            two * b / (sqrt(a * a + four * b * c) - a)
        } else {
            (a + sqrt(a * a + four * b * c)) / (two * c)
        };
        tau2 / (dn + sqrt(dn * dn + tau2))
    };
    let mut tau = if w <= zero {
        let t1 = sqrt(dn * dn + rho);
        let bound = z[prev] * z[prev] / ((d[prev] + t1) * (dn - d[prev] + rho / (dn + t1)))
            + z[last] * z[last] / rho;
        if c <= bound { upper } else { quadratic(c) }
    } else {
        quadratic(c)
    };
    if !(tau.is_finite() && tau > zero) {
        tau = (lo + hi) / two;
    }
    let mut sigma = dn + tau;
    for j in 0..n {
        delta[j] = (d[j] - dn) - tau;
        work[j] = d[j] + dn + tau;
    }

    let evaluate = |delta: &[R], work: &[R]| {
        let ev = eval_split(last, z, delta, work);
        let temp = z[last] / (delta[last] * work[last]);
        let phi = z[last] * temp;
        let dphi = temp * temp;
        let erretm = eight * (-phi - ev.psi) + ev.erretm - phi + rhoinv;
        (rhoinv + phi + ev.psi, ev.dpsi, dphi, erretm)
    };
    let (mut w, mut dpsi, mut dphi, mut erretm) = evaluate(delta, work);

    for niter in 1..=MAXIT {
        if fabs(w) <= eps * erretm {
            return Some(sigma);
        }
        if w <= zero {
            lo = lo.max(tau);
        } else {
            hi = hi.min(tau);
        }
        let dtnsq1 = work[prev] * delta[prev];
        let dtnsq = work[last] * delta[last];
        let mut c = w - dtnsq1 * dpsi - dtnsq * dphi;
        let a = (dtnsq + dtnsq1) * w - dtnsq * dtnsq1 * (dpsi + dphi);
        let b = dtnsq * dtnsq1 * w;
        if niter == 1 && c < zero {
            c = fabs(c);
        }
        let mut eta = if c == zero {
            if niter == 1 {
                rho - sigma * sigma
            } else {
                -w / (dpsi + dphi)
            }
        } else if a >= zero {
            (a + sqrt(fabs(a * a - four * b * c))) / (two * c)
        } else {
            two * b / (a - sqrt(fabs(a * a - four * b * c)))
        };
        // η must have the opposite sign of w; fall back to a Newton step.
        if w * eta > zero || !eta.is_finite() {
            eta = -w / (dpsi + dphi);
        }
        let temp = eta - dtnsq;
        if niter == 1 {
            if temp > rho {
                eta = rho + dtnsq;
            }
        } else if temp <= zero {
            eta = eta / two;
        }
        eta = eta / (sigma + sqrt(eta + sigma * sigma));
        if !(tau + eta > lo && tau + eta < hi) || !eta.is_finite() {
            // Leaving the bracket: bisect instead.
            eta = (lo + hi) / two - tau;
        }
        shift(&mut tau, &mut sigma, delta, work, eta);
        (w, dpsi, dphi, erretm) = evaluate(delta, work);
    }
    if fabs(w) <= eps * erretm {
        return Some(sigma);
    }
    bisect(d, z, rho, last, lo, hi, delta, work)
}

/// Two-pole step of `dlasd4` for an interior root (the `SWTCH3 = .FALSE.`
/// formulas; `swtch` selects the fixed-weight variant).
#[allow(clippy::too_many_arguments)]
fn two_pole_eta<R: Field + Real>(
    orgati: bool,
    swtch: bool,
    ii_term: R,
    w: R,
    dw: R,
    dpsi: R,
    dphi: R,
    delsq: R,
    zi: R,
    zip1: R,
    dtipsq: R,
    dtisq: R,
) -> R {
    let zero = R::zero();
    let two = real_const::<R>(2.0);
    let four = real_const::<R>(4.0);
    let (mut dpsi, mut dphi) = (dpsi, dphi);
    let c = if !swtch {
        if orgati {
            w - dtipsq * dw + delsq * (zi / dtisq) * (zi / dtisq)
        } else {
            w - dtisq * dw - delsq * (zip1 / dtipsq) * (zip1 / dtipsq)
        }
    } else {
        if orgati {
            dpsi = dpsi + ii_term * ii_term;
        } else {
            dphi = dphi + ii_term * ii_term;
        }
        w - dtisq * dpsi - dtipsq * dphi
    };
    let mut a = (dtipsq + dtisq) * w - dtipsq * dtisq * dw;
    let b = dtipsq * dtisq * w;
    if c == zero {
        if a == zero {
            a = if !swtch {
                if orgati {
                    zi * zi + dtipsq * dtipsq * (dpsi + dphi)
                } else {
                    zip1 * zip1 + dtisq * dtisq * (dpsi + dphi)
                }
            } else {
                dtisq * dtisq * dpsi + dtipsq * dtipsq * dphi
            };
        }
        b / a
    } else if a <= zero {
        (a - sqrt(fabs(a * a - four * b * c))) / (two * c)
    } else {
        two * b / (a + sqrt(fabs(a * a - four * b * c)))
    }
}

/// State shared by the steps of an interior root.
struct Interior<'a, R> {
    d: &'a [R],
    z: &'a [R],
    i: usize,
    ii: usize,
    orgati: bool,
    rhoinv: R,
    delsq: R,
}

impl<R: Field + Real> Interior<'_, R> {
    /// Three-pole step of `dlasd4` (through `dlaed6`); `None` if `dlaed6`
    /// did not converge.
    #[allow(clippy::too_many_arguments)]
    fn three_pole_eta(
        &self,
        niter: usize,
        swtch: bool,
        w: R,
        ev: &SplitEval<R>,
        delta: &[R],
        work: &[R],
    ) -> Option<R> {
        let (d, z, ii) = (self.d, self.z, self.ii);
        let iim1 = ii - 1;
        let iip1 = ii + 1;
        let dtiim = work[iim1] * delta[iim1];
        let dtiip = work[iip1] * delta[iip1];
        let temp = self.rhoinv + ev.psi + ev.phi;
        let (c, zz0, zz2);
        if swtch {
            c = temp - dtiim * ev.dpsi - dtiip * ev.dphi;
            zz0 = dtiim * dtiim * ev.dpsi;
            zz2 = dtiip * dtiip * ev.dphi;
        } else if self.orgati {
            let temp1 = (z[iim1] / dtiim) * (z[iim1] / dtiim);
            let temp2 = (d[iim1] - d[iip1]) * (d[iim1] + d[iip1]) * temp1;
            c = temp - dtiip * (ev.dpsi + ev.dphi) - temp2;
            zz0 = z[iim1] * z[iim1];
            zz2 = if ev.dpsi < temp1 {
                dtiip * dtiip * ev.dphi
            } else {
                dtiip * dtiip * ((ev.dpsi - temp1) + ev.dphi)
            };
        } else {
            let temp1 = (z[iip1] / dtiip) * (z[iip1] / dtiip);
            let temp2 = (d[iip1] - d[iim1]) * (d[iim1] + d[iip1]) * temp1;
            c = temp - dtiim * (ev.dpsi + ev.dphi) - temp2;
            zz0 = if ev.dphi < temp1 {
                dtiim * dtiim * ev.dpsi
            } else {
                dtiim * dtiim * (ev.dpsi + (ev.dphi - temp1))
            };
            zz2 = z[iip1] * z[iip1];
        }
        let zz1 = z[ii] * z[ii];
        let dd = [dtiim, delta[ii] * work[ii], dtiip];
        laed6(niter, self.orgati, c, dd, [zz0, zz1, zz2], w)
    }

    /// Two-pole step with the current values.
    #[allow(clippy::too_many_arguments)]
    fn two_pole(&self, swtch: bool, w: R, dw: R, ev: &SplitEval<R>, delta: &[R], work: &[R]) -> R {
        let (z, i, ii) = (self.z, self.i, self.ii);
        let ip1 = i + 1;
        let dtipsq = work[ip1] * delta[ip1];
        let dtisq = work[i] * delta[i];
        let ii_term = z[ii] / (work[ii] * delta[ii]);
        two_pole_eta(
            self.orgati,
            swtch,
            ii_term,
            w,
            dw,
            ev.dpsi,
            ev.dphi,
            self.delsq,
            z[i],
            z[ip1],
            dtipsq,
            dtisq,
        )
    }
}

/// Interior root `i < n − 1` (`dlasd4`, `I < N` branch).
fn lasd4_interior<R: Field + Real>(
    i: usize,
    d: &[R],
    z: &[R],
    rho: R,
    delta: &mut [R],
    work: &mut [R],
) -> Option<R> {
    let n = d.len();
    let zero = R::zero();
    let two = real_const::<R>(2.0);
    let three = real_const::<R>(3.0);
    let four = real_const::<R>(4.0);
    let eight = real_const::<R>(8.0);
    let ten = real_const::<R>(10.0);
    let eps = lapack_eps::<R>();
    let rhoinv = R::one() / rho;
    let ip1 = i + 1;

    // Initial guess from the midpoint σ² = (dᵢ² + dᵢ₊₁²)/2.
    let delsq = (d[ip1] - d[i]) * (d[ip1] + d[i]);
    let delsq2 = delsq / two;
    let sq2 = sqrt((d[i] * d[i] + d[ip1] * d[ip1]) / two);
    let temp = delsq2 / (d[i] + sq2);
    for j in 0..n {
        work[j] = d[j] + d[i] + temp;
        delta[j] = (d[j] - d[i]) - temp;
    }
    let mut psi = zero;
    for j in 0..i {
        psi = psi + z[j] * z[j] / (work[j] * delta[j]);
    }
    let mut phi = zero;
    for j in (i + 2..n).rev() {
        phi = phi + z[j] * z[j] / (work[j] * delta[j]);
    }
    let c = rhoinv + psi + phi;
    let w = c + z[i] * z[i] / (work[i] * delta[i]) + z[ip1] * z[ip1] / (work[ip1] * delta[ip1]);

    let mut geomavg = false;
    let orgati;
    let ii;
    let mut sglb;
    let mut sgub;
    let mut tau;
    if w > zero {
        // dᵢ² < σ² < (dᵢ² + dᵢ₊₁²)/2: origin dᵢ.
        orgati = true;
        ii = i;
        sglb = zero;
        sgub = delsq2 / (d[i] + sq2);
        let a = c * delsq + z[i] * z[i] + z[ip1] * z[ip1];
        let b = z[i] * z[i] * delsq;
        let tau2 = if a > zero {
            two * b / (a + sqrt(fabs(a * a - four * b * c)))
        } else {
            (a - sqrt(fabs(a * a - four * b * c))) / (two * c)
        };
        tau = tau2 / (d[i] + sqrt(d[i] * d[i] + tau2));
        let small = sqrt(eps);
        if d[i] <= small * d[ip1] && fabs(z[i]) <= small && d[i] > zero {
            tau = (ten * d[i]).min(sgub);
            geomavg = true;
        }
    } else {
        // (dᵢ² + dᵢ₊₁²)/2 ≤ σ² < dᵢ₊₁²: origin dᵢ₊₁.
        orgati = false;
        ii = ip1;
        sglb = -delsq2 / (d[ii] + sq2);
        sgub = zero;
        let a = c * delsq - z[i] * z[i] - z[ip1] * z[ip1];
        let b = z[ip1] * z[ip1] * delsq;
        let tau2 = if a < zero {
            two * b / (a - sqrt(fabs(a * a + four * b * c)))
        } else {
            -(a + sqrt(fabs(a * a + four * b * c))) / (two * c)
        };
        tau = tau2 / (d[ip1] + sqrt(fabs(d[ip1] * d[ip1] + tau2)));
    }
    if !(tau.is_finite() && tau >= sglb && tau <= sgub) {
        tau = (sglb + sgub) / two;
    }
    let dii = d[ii];
    let mut sigma = dii + tau;
    for j in 0..n {
        work[j] = d[j] + dii + tau;
        delta[j] = (d[j] - dii) - tau;
    }
    let st = Interior {
        d,
        z,
        i,
        ii,
        orgati,
        rhoinv,
        delsq,
    };

    let mut ev = eval_split(ii, z, delta, work);
    let w_rest = rhoinv + ev.phi + ev.psi;
    let mut swtch3 = if orgati { w_rest < zero } else { w_rest > zero };
    if ii == 0 || ii == n - 1 {
        swtch3 = false;
    }
    let complete = |ev: &SplitEval<R>, delta: &[R], work: &[R]| {
        let temp = z[ii] / (work[ii] * delta[ii]);
        let dw = ev.dpsi + ev.dphi + temp * temp;
        let temp = z[ii] * temp;
        let w = rhoinv + ev.phi + ev.psi + temp;
        let erretm = eight * (ev.phi - ev.psi) + ev.erretm + two * rhoinv + three * fabs(temp);
        (w, dw, erretm)
    };
    let (mut w, mut dw, mut erretm) = complete(&ev, delta, work);
    let mut swtch = false;

    for niter in 2..=MAXIT + 1 {
        if fabs(w) <= eps * erretm {
            return Some(sigma);
        }
        if w <= zero {
            sglb = sglb.max(tau);
        } else {
            sgub = sgub.min(tau);
        }
        let mut eta = if swtch3 {
            match st.three_pole_eta(niter, swtch, w, &ev, delta, work) {
                Some(eta) => eta,
                None => {
                    swtch3 = false;
                    st.two_pole(swtch, w, dw, &ev, delta, work)
                }
            }
        } else {
            st.two_pole(swtch, w, dw, &ev, delta, work)
        };
        // η must have the opposite sign of w; fall back to a Newton step.
        if w * eta >= zero || !eta.is_finite() {
            eta = -w / dw;
        }
        eta = eta / (sigma + sqrt(sigma * sigma + eta));
        let next = tau + eta;
        if !(next <= sgub && next >= sglb) {
            eta = if w < zero {
                (sgub - tau) / two
            } else {
                (sglb - tau) / two
            };
            if geomavg {
                if w < zero {
                    if tau > zero {
                        eta = sqrt(sgub * tau) - tau;
                    }
                } else if sglb > zero {
                    eta = sqrt(sglb * tau) - tau;
                }
            }
        }
        let prew = w;
        shift(&mut tau, &mut sigma, delta, work, eta);
        ev = eval_split(ii, z, delta, work);
        (w, dw, erretm) = complete(&ev, delta, work);
        if niter == 2 {
            // First step taken: choose the model for the next ones.
            swtch = if orgati {
                -w > fabs(prew) / ten
            } else {
                w > fabs(prew) / ten
            };
        } else if w * prew > zero && fabs(w) > fabs(prew) / ten {
            swtch = !swtch;
        }
    }
    if fabs(w) <= eps * erretm {
        return Some(sigma);
    }
    bisect(d, z, rho, ii, sglb, sgub, delta, work)
}

/// The root closest to the origin of
/// `f(x) = ρ + z₀/(d₀ − x) + z₁/(d₁ − x) + z₂/(d₂ − x)` (LAPACK `dlaed6`),
/// between `d₁` and `d₂` if `orgati`, else between `d₀` and `d₁`. `finit` is
/// `f(0)`. Returns `None` if the cubically convergent iteration failed.
fn laed6<R: Field + Real>(
    kniter: usize,
    orgati: bool,
    rho: R,
    d: [R; 3],
    z: [R; 3],
    finit: R,
) -> Option<R> {
    let zero = R::zero();
    let one = R::one();
    let two = real_const::<R>(2.0);
    let four = real_const::<R>(4.0);
    let eight = real_const::<R>(8.0);
    let (mut lbd, mut ubd) = if orgati { (d[1], d[2]) } else { (d[0], d[1]) };
    if finit < zero {
        lbd = zero;
    } else {
        ubd = zero;
    }
    let quad = |a: R, b: R, c: R| -> Option<R> {
        let t = fabs(a).max(fabs(b)).max(fabs(c));
        if !(t > zero && t.is_finite()) {
            return None;
        }
        let (a, b, c) = (a / t, b / t, c / t);
        Some(if c == zero {
            b / a
        } else if a <= zero {
            (a - sqrt(fabs(a * a - four * b * c))) / (two * c)
        } else {
            two * b / (a + sqrt(fabs(a * a - four * b * c)))
        })
    };

    let mut tau = zero;
    if kniter == 2 {
        let (a, b, c) = if orgati {
            let temp = (d[2] - d[1]) / two;
            let c = rho + z[0] / ((d[0] - d[1]) - temp);
            (
                c * (d[1] + d[2]) + z[1] + z[2],
                c * d[1] * d[2] + z[1] * d[2] + z[2] * d[1],
                c,
            )
        } else {
            let temp = (d[0] - d[1]) / two;
            let c = rho + z[2] / ((d[2] - d[1]) - temp);
            (
                c * (d[0] + d[1]) + z[0] + z[1],
                c * d[0] * d[1] + z[0] * d[1] + z[1] * d[0],
                c,
            )
        };
        tau = quad(a, b, c)?;
        if tau < lbd || tau > ubd {
            tau = (lbd + ubd) / two;
        }
        if d[0] == tau || d[1] == tau || d[2] == tau {
            tau = zero;
        } else {
            let temp = finit
                + tau * z[0] / (d[0] * (d[0] - tau))
                + tau * z[1] / (d[1] * (d[1] - tau))
                + tau * z[2] / (d[2] * (d[2] - tau));
            if temp <= zero {
                lbd = tau;
            } else {
                ubd = tau;
            }
            if fabs(finit) <= fabs(temp) {
                tau = zero;
            }
        }
    }

    // Scale up (by a power of two) when 1/(dᵢ − τ)³ could overflow.
    let eps = lapack_eps::<R>();
    let e1 = <R as Real>::trunc(
        num_traits::Float::log2(<R as oxiblas_core::scalar::Scalar>::min_positive())
            / real_const::<R>(3.0),
    );
    let small1 = <R as Real>::powf(two, e1);
    let sminv1 = one / small1;
    let small2 = small1 * small1;
    let sminv2 = sminv1 * sminv1;
    let gap = if orgati {
        fabs(d[1] - tau).min(fabs(d[2] - tau))
    } else {
        fabs(d[0] - tau).min(fabs(d[1] - tau))
    };
    let mut dscale = d;
    let mut zscale = z;
    let mut sclinv = one;
    let scale = gap <= small1;
    if scale {
        let sclfac = if gap <= small2 {
            sclinv = small2;
            sminv2
        } else {
            sclinv = small1;
            sminv1
        };
        for k in 0..3 {
            dscale[k] = d[k] * sclfac;
            zscale[k] = z[k] * sclfac;
        }
        tau = tau * sclfac;
        lbd = lbd * sclfac;
        ubd = ubd * sclfac;
    }
    let unscale = |t: R| if scale { t * sclinv } else { t };

    let mut fc = zero;
    let mut df = zero;
    let mut ddf = zero;
    for k in 0..3 {
        let temp = one / (dscale[k] - tau);
        let temp1 = zscale[k] * temp;
        let temp2 = temp1 * temp;
        fc = fc + temp1 / dscale[k];
        df = df + temp2;
        ddf = ddf + temp2 * temp;
    }
    let mut f = finit + tau * fc;
    if fabs(f) <= zero {
        return Some(unscale(tau));
    }
    if f <= zero {
        lbd = tau;
    } else {
        ubd = tau;
    }

    // Gragg–Thornton–Warner cubically convergent iteration.
    for _ in 2..=LAED6_MAXIT {
        let (t1, t2) = if orgati {
            (dscale[1] - tau, dscale[2] - tau)
        } else {
            (dscale[0] - tau, dscale[1] - tau)
        };
        let a = (t1 + t2) * f - t1 * t2 * df;
        let b = t1 * t2 * f;
        let c = f - (t1 + t2) * df + t1 * t2 * ddf;
        let mut eta = quad(a, b, c)?;
        if f * eta >= zero {
            eta = -f / df;
        }
        tau = tau + eta;
        if tau < lbd || tau > ubd {
            tau = (lbd + ubd) / two;
        }
        fc = zero;
        let mut erretm = zero;
        df = zero;
        ddf = zero;
        for k in 0..3 {
            if dscale[k] - tau == zero {
                return Some(unscale(tau));
            }
            let temp = one / (dscale[k] - tau);
            let temp1 = zscale[k] * temp;
            let temp2 = temp1 * temp;
            let temp4 = temp1 / dscale[k];
            fc = fc + temp4;
            erretm = erretm + fabs(temp4);
            df = df + temp2;
            ddf = ddf + temp2 * temp;
        }
        f = finit + tau * fc;
        erretm = eight * (fabs(finit) + fabs(tau) * erretm) + fabs(tau) * df;
        if fabs(f) <= four * eps * erretm || (ubd - lbd) <= four * eps * fabs(tau) {
            return Some(unscale(tau));
        }
        if f <= zero {
            lbd = tau;
        } else {
            ubd = tau;
        }
    }
    None
}

#[cfg(test)]
#[path = "secular_tests.rs"]
mod tests;
