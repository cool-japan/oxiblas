//! Implicit QR iteration on a symmetric tridiagonal matrix with batched,
//! cache-blocked accumulation of the plane rotations.
//!
//! The iteration itself (Wilkinson shift, deflation at the bottom of the
//! active block) runs on `d`/`e` only and records each sweep's rotations.
//! Recorded sweeps are applied to the eigenvector block in row blocks of
//! [`ROW_BLOCK`] rows: within a row block the sweeps advance as a wavefront
//! (a later sweep's rotation on columns `(c, c+1)` runs once every earlier
//! sweep has passed column `c + 1`), so the few dozen columns in flight stay
//! in L1 while up to [`MAX_SWEEPS`] sweeps are applied. Only commuting
//! rotations (disjoint column pairs) are reordered, so the result is the
//! same as applying every rotation as soon as it is generated.

use oxiblas_core::scalar::{Field, Real, Scalar};

use super::sym_kernels::pow2_below;

/// Rows per block when applying recorded rotations.
const ROW_BLOCK: usize = 128;
/// Rotations a sweep may advance per wavefront step.
const WAVE: usize = 16;
/// Sweeps recorded before they are applied.
const MAX_SWEEPS: usize = 32;

/// When an off-diagonal entry of the tridiagonal counts as negligible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Deflation {
    /// `|e| <= 100·eps·(|d_{l-1}| + |d_l|)`. Needs fewer sweeps than
    /// [`Deflation::Strict`], so less rounding accumulates in the
    /// eigenvectors of large problems (the QR driver; the historical test).
    Relaxed,
    /// LAPACK `DSTEQR`: `e² <= eps_u²·|d_{l-1}·d_l| + safmin`, relative to
    /// the geometric mean of the neighbouring diagonal entries (the small
    /// divide-and-conquer leaves, where it halves the residual).
    Strict,
}

/// The tridiagonal QR iteration did not converge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NotConverged;

/// Column-major block that accumulates the rotations: rows `0..nrows` of
/// columns `0..ncols`, element `(i, j)` at `data[off + i + j * ld]`.
pub(crate) struct RotTarget<'a, T> {
    /// Backing storage.
    pub(crate) data: &'a mut [T],
    /// Offset of element `(0, 0)`.
    pub(crate) off: usize,
    /// Leading dimension.
    pub(crate) ld: usize,
    /// Number of rows to update.
    pub(crate) nrows: usize,
    /// The block is the identity on entry (columns `0..nrows`), so the
    /// rotations only need to touch the rows each column has filled in.
    pub(crate) identity: bool,
}

/// The rotations of one recorded sweep.
#[derive(Debug, Clone, Copy)]
struct Sweep {
    /// Index of the sweep's first rotation in the log.
    start: usize,
    /// Column of the first rotation (acting on columns `first, first + 1`).
    first: usize,
    /// Number of rotations (consecutive columns).
    len: usize,
}

/// Recorded rotations, reused across calls.
#[derive(Debug, Default)]
pub(crate) struct RotationLog<T> {
    c: Vec<T>,
    s: Vec<T>,
    /// Rows `lo[i]..hi[i]` that rotation `i` can change (outside them both
    /// of its columns are zero).
    lo: Vec<usize>,
    hi: Vec<usize>,
    sweeps: Vec<Sweep>,
    progress: Vec<usize>,
    /// Nonzero row range of every column of the target (support tracking).
    support: Vec<(usize, usize)>,
}

impl<T: Field + Real> RotationLog<T> {
    /// An empty log.
    pub(crate) fn new() -> Self {
        Self {
            c: Vec::new(),
            s: Vec::new(),
            lo: Vec::new(),
            hi: Vec::new(),
            sweeps: Vec::new(),
            progress: Vec::new(),
            support: Vec::new(),
        }
    }

    /// Starts support tracking for an `nrows × ncols` target that is the
    /// identity (`identity`) or dense on entry.
    fn init_support(&mut self, ncols: usize, nrows: usize, identity: bool) {
        self.support.clear();
        if identity {
            self.support
                .extend((0..ncols).map(|j| (j.min(nrows), (j + 1).min(nrows))));
        } else {
            self.support.resize(ncols, (0, nrows));
        }
    }

    /// Records rotation `(c, s)` on columns `(k, k + 1)`.
    #[inline]
    fn push(&mut self, k: usize, c: T, s: T) {
        let (t0, b0) = self.support[k];
        let (t1, b1) = self.support[k + 1];
        let range = (t0.min(t1), b0.max(b1));
        self.support[k] = range;
        self.support[k + 1] = range;
        self.c.push(c);
        self.s.push(s);
        self.lo.push(range.0);
        self.hi.push(range.1);
    }

    /// Applies every recorded rotation to `z` and clears the log.
    fn flush(&mut self, z: &mut Option<RotTarget<'_, T>>) {
        if let Some(target) = z.as_mut() {
            if !self.sweeps.is_empty() && target.nrows > 0 {
                let rots = Rotations {
                    c: &self.c,
                    s: &self.s,
                    lo: &self.lo,
                    hi: &self.hi,
                };
                apply_sweeps(target, &rots, &self.sweeps, &mut self.progress);
            }
        }
        self.c.clear();
        self.s.clear();
        self.lo.clear();
        self.hi.clear();
        self.sweeps.clear();
    }
}

/// Recorded rotations: coefficients and the rows each one can change.
struct Rotations<'a, T> {
    c: &'a [T],
    s: &'a [T],
    lo: &'a [usize],
    hi: &'a [usize],
}

/// Applies the recorded sweeps to all rows of `z`, one row block at a time,
/// advancing the sweeps as a wavefront.
fn apply_sweeps<T: Field + Real>(
    z: &mut RotTarget<'_, T>,
    rots: &Rotations<'_, T>,
    sweeps: &[Sweep],
    progress: &mut Vec<usize>,
) {
    let ld = z.ld;
    let off = z.off;
    let mut r0 = 0;
    while r0 < z.nrows {
        let r1 = (r0 + ROW_BLOCK).min(z.nrows);
        progress.clear();
        progress.resize(sweeps.len(), 0);
        loop {
            // `bound`: smallest next column over the unfinished earlier
            // sweeps; a rotation on (c, c+1) may run when c + 1 < bound.
            let mut bound = usize::MAX;
            let mut any = false;
            for (sw, done) in sweeps.iter().zip(progress.iter_mut()) {
                if *done == sw.len {
                    continue;
                }
                any = true;
                let allowed = bound.saturating_sub(sw.first + 1);
                let target = (*done + WAVE).min(sw.len).min(allowed);
                let mut idx = *done;
                while idx < target {
                    let col = sw.first + idx;
                    let ri = sw.start + idx;
                    let (c, s) = (rots.c[ri], rots.s[ri]);
                    let base = off + col * ld;
                    if idx + 1 < target {
                        // Two consecutive rotations on columns col..col+3 in
                        // one pass (the middle column stays in registers).
                        let (c2, s2) = (rots.c[ri + 1], rots.s[ri + 1]);
                        let a = r0.max(rots.lo[ri].min(rots.lo[ri + 1]));
                        let b = r1.min(rots.hi[ri].max(rots.hi[ri + 1]));
                        if a < b {
                            let (lo, hi) = z.data.split_at_mut(base + ld);
                            let (mid, last) = hi.split_at_mut(ld);
                            let x = &mut lo[base + a..base + b];
                            let y = &mut mid[a..b];
                            let w = &mut last[a..b];
                            rotate_pair(x, y, w, (c, s), (c2, s2));
                        }
                        idx += 2;
                        continue;
                    }
                    let a = r0.max(rots.lo[ri]);
                    let b = r1.min(rots.hi[ri]);
                    if a < b {
                        let (lo, hi) = z.data.split_at_mut(base + ld);
                        let x = &mut lo[base + a..base + b];
                        let y = &mut hi[a..b];
                        rotate(x, y, c, s);
                    }
                    idx += 1;
                }
                if target > *done {
                    *done = target;
                }
                if *done < sw.len {
                    bound = bound.min(sw.first + *done);
                }
            }
            if !any {
                break;
            }
        }
        r0 = r1;
    }
}

/// `(x, y) <- (c·x - s·y, s·x + c·y)` elementwise (fused multiply-adds).
#[inline]
fn rotate<T: Field + Real>(x: &mut [T], y: &mut [T], c: T, s: T) {
    for (xi, yi) in x.iter_mut().zip(y.iter_mut()) {
        let t1 = *xi;
        let t2 = *yi;
        *xi = Real::mul_add(c, t1, -(s * t2));
        *yi = Real::mul_add(s, t1, c * t2);
    }
}

/// Rotation `(c1, s1)` on `(x, y)` followed by `(c2, s2)` on `(y, w)`, with
/// the same arithmetic as two [`rotate`] calls.
#[inline]
fn rotate_pair<T: Field + Real>(x: &mut [T], y: &mut [T], w: &mut [T], g1: (T, T), g2: (T, T)) {
    let (c1, s1) = g1;
    let (c2, s2) = g2;
    for ((xi, yi), wi) in x.iter_mut().zip(y.iter_mut()).zip(w.iter_mut()) {
        let a = *xi;
        let b = *yi;
        let c = *wi;
        let b1 = Real::mul_add(s1, a, c1 * b);
        *xi = Real::mul_add(c1, a, -(s1 * b));
        *yi = Real::mul_add(c2, b1, -(s2 * c));
        *wi = Real::mul_add(s2, b1, c2 * c);
    }
}

/// Computes Givens rotation coefficients `(c, s)` with `c·a - s·b = r` and
/// `s·a + c·b = 0`.
#[inline]
fn givens_rotation<T: Field + Real>(a: T, b: T) -> (T, T) {
    if b == T::zero() {
        (T::one(), T::zero())
    } else if Scalar::abs(b) > Scalar::abs(a) {
        let t = -a / b;
        let s = T::one() / Real::sqrt(T::one() + t * t);
        (s * t, s)
    } else {
        let t = -b / a;
        let c = T::one() / Real::sqrt(T::one() + t * t);
        (c, c * t)
    }
}

/// Eigenvalues (in `d`, unsorted) of the symmetric tridiagonal matrix
/// `tridiag(e, d, e)` by the implicit QR algorithm with Wilkinson shifts.
///
/// Every rotation `G` (acting on columns `(k, k+1)`) is accumulated as
/// `Z := Z·G` into `z` when given, so a `z` holding `Q` on entry holds
/// `Q·Zₜ` on return, where `Zₜ` are the eigenvectors of the tridiagonal.
///
/// Off-diagonal entries are dropped according to `deflation`. Fails after
/// `max_iter · n` iterations without convergence (also the outcome for
/// non-finite input).
pub(crate) fn steqr<T: Field + Real>(
    d: &mut [T],
    e: &mut [T],
    mut z: Option<RotTarget<'_, T>>,
    max_iter: usize,
    deflation: Deflation,
    log: &mut RotationLog<T>,
) -> Result<(), NotConverged> {
    let n = d.len();
    if n <= 1 {
        return Ok(());
    }
    if d.iter().chain(e.iter()).any(|v| !v.is_finite()) {
        return Err(NotConverged);
    }

    let two = T::one() + T::one();
    let limit = max_iter.saturating_mul(n);
    let eps_u = <T as Scalar>::epsilon() / two;
    let eps2 = eps_u * eps_u;
    let safmin = <T as Scalar>::min_positive();
    let relaxed_eps = <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or(T::one());

    // Work at unit max-norm (a power-of-two scaling, exact both ways and
    // invisible to the rotations) so e² and |d·d| cannot overflow.
    let anorm = d
        .iter()
        .chain(e.iter())
        .fold(T::zero(), |acc, &v| acc.max(Scalar::abs(v)));
    if anorm == T::zero() {
        return Ok(());
    }
    let scale = pow2_below(anorm);
    if scale != T::one() {
        for v in d.iter_mut().chain(e.iter_mut()) {
            *v /= scale;
        }
    }
    log.flush(&mut None);
    match z.as_ref() {
        Some(t) => log.init_support(n, t.nrows, t.identity),
        None => log.init_support(n, 0, false),
    }

    let mut m = n - 1;
    let mut iter = 0;
    while m > 0 && iter < limit {
        iter += 1;

        // Largest unreduced block [l, m].
        let mut l = m;
        while l > 0 {
            let ev = e[l - 1];
            let (dl, dr) = (Scalar::abs(d[l - 1]), Scalar::abs(d[l]));
            let negligible = match deflation {
                Deflation::Relaxed => Scalar::abs(ev) <= relaxed_eps * (dl + dr),
                Deflation::Strict => ev * ev <= eps2 * dl * dr + safmin,
            };
            if negligible {
                e[l - 1] = T::zero();
                break;
            }
            l -= 1;
        }
        if l == m {
            m -= 1;
            continue;
        }

        // Wilkinson shift.
        let dd = (d[m - 1] - d[m]) / two;
        let ee = e[m - 1];
        let mu = d[m] - ee * ee / (dd + Real::signum(dd) * Real::hypot(dd, ee));

        // Implicit QR sweep over [l, m].
        let mut x = d[l] - mu;
        let mut zz = e[l];
        log.sweeps.push(Sweep {
            start: log.c.len(),
            first: l,
            len: m - l,
        });
        for k in l..m {
            let (c, s) = givens_rotation(x, zz);
            if k > l {
                // The regenerated off-diagonal is r = c·x - s·z (hypot would
                // lose its sign and corrupt the eigenvectors).
                e[k - 1] = c * x - s * zz;
            }
            let d1 = d[k];
            let d2 = d[k + 1];
            let ek = e[k];
            d[k] = c * c * d1 + s * s * d2 - (c + c) * s * ek;
            d[k + 1] = s * s * d1 + c * c * d2 + (c + c) * s * ek;
            e[k] = c * s * (d1 - d2) + (c * c - s * s) * ek;
            if k < m - 1 {
                x = e[k];
                zz = -s * e[k + 1];
                e[k + 1] = c * e[k + 1];
            }
            log.push(k, c, s);
        }
        if log.sweeps.len() >= MAX_SWEEPS {
            log.flush(&mut z);
        }
    }
    log.flush(&mut z);

    if scale != T::one() {
        for v in d.iter_mut() {
            *v *= scale;
        }
    }
    if iter >= limit && m > 0 {
        return Err(NotConverged);
    }
    Ok(())
}
