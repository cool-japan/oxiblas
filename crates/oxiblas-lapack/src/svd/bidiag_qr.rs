//! Bidiagonal SVD by implicit QR iteration (LAPACK `dbdsqr` / `dlasdq`).
//!
//! [`lasdq`] computes the SVD of an `n × (n + sqre)` upper bidiagonal matrix
//! (`sqre ∈ {0, 1}`) and accumulates the transformations into the columns of
//! caller-provided matrices `U` (`·×n`) and `V` (`·×(n + sqre)`, by columns):
//! on exit `B_in = U·[diag(σ) 0]·Vᵀ` relative to the matrices on entry, with
//! `σ` ascending. It is the leaf solver of the divide-and-conquer
//! ([`crate::svd::bidiag_dc`]) and the whole solver for small matrices.
//!
//! The iteration is a port of LAPACK 3.12 `dbdsqr` in its relative-accuracy
//! mode:
//!
//! - convergence/deflation tests are relative (`|e_i| ≤ tol·μ_i` with the
//!   `μ` recurrence of Demmel–Kahan, `tol = tolmul·ε`), so every singular
//!   value of the bidiagonal is computed to high relative accuracy;
//! - each sweep chases the bulge from the larger end of the block towards the
//!   smaller one and uses the Demmel–Kahan zero shift whenever a shift could
//!   destroy relative accuracy (`n·tol·σ_min/σ_max ≤ max(ε, tol/100)`),
//!   otherwise the `dlas2` shift of the trailing (leading) 2×2 block;
//! - 2×2 blocks are diagonalised directly by `dlasv2`;
//! - the iteration budget is `6·n²` inner steps; exceeding it returns
//!   [`BidiagDcError::NotConverged`].
//!
//! In addition (Golub–Van Loan, Algorithm 8.6.2), an exactly zero diagonal
//! entry — or one below `6·n²·safmin`, i.e. at the underflow level — is set
//! to zero and removed explicitly: for `d_k = 0` inside a block the
//! super-diagonal of row `k` is chased to the right with left rotations
//! (accumulated into `U`); for the last diagonal of a block the column is
//! chased upwards with right rotations (accumulated into `V`). Both split the
//! block. Larger diagonal entries are never zeroed (a normwise threshold such
//! as `ε·‖B‖` would destroy the relative accuracy of tiny singular values);
//! the zero-shift sweep deflates them instead.
//!
//! `sqre = 1` is reduced to the square case as in `dlasdq`: right rotations
//! turn the `n × (n + 1)` upper bidiagonal into a lower bidiagonal plus a zero
//! column (accumulated into `V`), and left rotations make it upper bidiagonal
//! again (accumulated into `U`).

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::Mat;

use super::bidiag_dc::BidiagDcError;
use super::dc_kernels::{fabs, lapack_eps, real_const, rot_cols, sign_of, swap_cols};

/// `MAXITR` of `dbdsqr`: the iteration budget is `MAXITR·n²` inner steps.
const MAXITR: usize = 6;

/// Plane rotation (LAPACK 3.10+ `dlartg`).
///
/// Returns `(c, s, r)` with `[c s; −s c]·[f; g] = [r; 0]`, `c ≥ 0` and
/// `sign(r) = sign(f)` (`r = |g|` when `f = 0`). Over- and underflow are
/// avoided by scaling outside `[√safmin, √(safmax/2)]`.
pub(crate) fn lartg<R: Field + Real>(f: R, g: R) -> (R, R, R) {
    let zero = R::zero();
    let one = R::one();
    let safmin = <R as Scalar>::min_positive();
    let safmax = one / safmin;
    let rtmin = <R as Real>::sqrt(safmin);
    let rtmax = <R as Real>::sqrt(safmax * real_const::<R>(0.5));
    let f1 = fabs(f);
    let g1 = fabs(g);
    if g == zero {
        (one, zero, f)
    } else if f == zero {
        (zero, sign_of(one, g), g1)
    } else if f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax {
        let d = <R as Real>::sqrt(f * f + g * g);
        let c = f1 / d;
        let r = sign_of(d, f);
        (c, g / r, r)
    } else {
        let u = safmax.min(safmin.max(f1).max(g1));
        let fs = f / u;
        let gs = g / u;
        let d = <R as Real>::sqrt(fs * fs + gs * gs);
        let c = fabs(fs) / d;
        let r = sign_of(d, f);
        (c, gs / r, r * u)
    }
}

/// Singular values `(σ_min, σ_max)` of the upper triangular 2×2
/// `[[f, g], [0, h]]` (LAPACK `dlas2`).
pub(crate) fn las2<R: Field + Real>(f: R, g: R, h: R) -> (R, R) {
    let zero = R::zero();
    let one = R::one();
    let two = real_const::<R>(2.0);
    let fa = fabs(f);
    let ga = fabs(g);
    let ha = fabs(h);
    let fhmn = fa.min(ha);
    let fhmx = fa.max(ha);
    if fhmn == zero {
        let ssmax = if fhmx == zero {
            ga
        } else {
            let mx = fhmx.max(ga);
            let mn = fhmx.min(ga);
            mx * <R as Real>::sqrt(one + (mn / mx) * (mn / mx))
        };
        (zero, ssmax)
    } else if ga < fhmx {
        let as_ = one + fhmn / fhmx;
        let at = (fhmx - fhmn) / fhmx;
        let au = (ga / fhmx) * (ga / fhmx);
        let c = two / (<R as Real>::sqrt(as_ * as_ + au) + <R as Real>::sqrt(at * at + au));
        (fhmn * c, fhmx / c)
    } else {
        let au = fhmx / ga;
        if au == zero {
            // Avoid harmful underflow if the exponent range is asymmetric.
            ((fhmn * fhmx) / ga, ga)
        } else {
            let as_ = one + fhmn / fhmx;
            let at = (fhmx - fhmn) / fhmx;
            let c = one
                / (<R as Real>::sqrt(one + (as_ * au) * (as_ * au))
                    + <R as Real>::sqrt(one + (at * au) * (at * au)));
            let ssmin = (fhmn * c) * au;
            (ssmin + ssmin, ga / (c + c))
        }
    }
}

/// Result of [`lasv2`]:
/// `[csl snl; −snl csl]·[[f, g], [0, h]]·[csr −snr; snr csr] = diag(ssmax, ssmin)`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Lasv2<R> {
    /// Smaller singular value (signed).
    pub ssmin: R,
    /// Larger singular value (signed).
    pub ssmax: R,
    /// Sine of the right rotation.
    pub snr: R,
    /// Cosine of the right rotation.
    pub csr: R,
    /// Sine of the left rotation.
    pub snl: R,
    /// Cosine of the left rotation.
    pub csl: R,
}

/// SVD of the upper triangular 2×2 `[[f, g], [0, h]]` (LAPACK `dlasv2`).
pub(crate) fn lasv2<R: Field + Real>(f: R, g: R, h: R) -> Lasv2<R> {
    let zero = R::zero();
    let one = R::one();
    let two = real_const::<R>(2.0);
    let four = real_const::<R>(4.0);
    let half = real_const::<R>(0.5);

    let mut ft = f;
    let mut fa = fabs(ft);
    let mut ht = h;
    let mut ha = fabs(h);
    // pmax: 1 if f is the largest entry, 2 if g, 3 if h.
    let mut pmax = 1u8;
    let swap = ha > fa;
    if swap {
        pmax = 3;
        core::mem::swap(&mut ft, &mut ht);
        core::mem::swap(&mut fa, &mut ha);
    }
    let gt = g;
    let ga = fabs(gt);
    let (ssmin, ssmax, clt, crt, slt, srt);
    if ga == zero {
        // Diagonal matrix.
        ssmin = ha;
        ssmax = fa;
        clt = one;
        crt = one;
        slt = zero;
        srt = zero;
    } else {
        let mut gasmal = true;
        let mut out = (zero, zero, one, one, zero, zero);
        if ga > fa {
            pmax = 2;
            if fa / ga < lapack_eps::<R>() {
                // Very large g.
                gasmal = false;
                let smin = if ha > one {
                    fa / (ga / ha)
                } else {
                    (fa / ga) * ha
                };
                out = (smin, ga, one, ft / gt, ht / gt, one);
            }
        }
        if gasmal {
            let dd = fa - ha;
            let mut l = if dd == fa { one } else { dd / fa };
            let m = gt / ft;
            let mut t = two - l;
            let mm = m * m;
            let tt = t * t;
            let s = <R as Real>::sqrt(tt + mm);
            let r = if l == zero {
                fabs(m)
            } else {
                <R as Real>::sqrt(l * l + mm)
            };
            let a = half * (s + r);
            let smin = ha / a;
            let smax = fa * a;
            if mm == zero {
                t = if l == zero {
                    sign_of(two, ft) * sign_of(one, gt)
                } else {
                    gt / sign_of(dd, ft) + m / t
                };
            } else {
                t = (m / (s + t) + m / (r + l)) * (one + a);
            }
            l = <R as Real>::sqrt(t * t + four);
            let crt_ = two / l;
            let srt_ = t / l;
            let clt_ = (crt_ + srt_ * m) / a;
            let slt_ = (ht / ft) * srt_ / a;
            out = (smin, smax, clt_, crt_, slt_, srt_);
        }
        ssmin = out.0;
        ssmax = out.1;
        clt = out.2;
        crt = out.3;
        slt = out.4;
        srt = out.5;
    }
    let (csl, snl, csr, snr) = if swap {
        (srt, crt, slt, clt)
    } else {
        (clt, slt, crt, srt)
    };
    let tsign = match pmax {
        1 => sign_of(one, csr) * sign_of(one, csl) * sign_of(one, f),
        2 => sign_of(one, snr) * sign_of(one, csl) * sign_of(one, g),
        _ => sign_of(one, snr) * sign_of(one, snl) * sign_of(one, h),
    };
    Lasv2 {
        ssmax: sign_of(ssmax, tsign),
        ssmin: sign_of(ssmin, tsign * sign_of(one, f) * sign_of(one, h)),
        snr,
        csr,
        snl,
        csl,
    }
}

/// SVD of the `n × (n + sqre)` upper bidiagonal with diagonal `d` (`n`) and
/// super-diagonal `e` (`n − 1 + sqre`) by implicit QR (LAPACK `dlasdq`).
///
/// `u` (`p × n`, any `p`) and `v` (`q × (n + sqre)`) are multiplied from the
/// right by the left and right rotations: if `B = U₀·B₀·V₀ᵀ` on entry, then
/// on exit `B = U·[diag(d) 0]·Vᵀ`. On success `d` holds the singular values in
/// ascending order (the columns of `u` and `v` are permuted accordingly) and
/// `e` is overwritten.
pub(crate) fn lasdq<R: Field + Real>(
    sqre: usize,
    d: &mut [R],
    e: &mut [R],
    u: &mut Mat<R>,
    v: &mut Mat<R>,
) -> Result<(), BidiagDcError> {
    let n = d.len();
    if n == 0 {
        return Ok(());
    }
    let nu = u.nrows();
    let nv = v.nrows();
    if sqre == 1 {
        // Upper n×(n+1) → lower n×n plus a zero column (rotations on the right).
        for i in 0..n - 1 {
            let (cs, sn, r) = lartg(d[i], e[i]);
            d[i] = r;
            e[i] = sn * d[i + 1];
            d[i + 1] = cs * d[i + 1];
            rot_cols(v, i, i + 1, 0, nv, cs, sn);
        }
        let (cs, sn, r) = lartg(d[n - 1], e[n - 1]);
        d[n - 1] = r;
        e[n - 1] = R::zero();
        rot_cols(v, n - 1, n, 0, nv, cs, sn);
        // Lower → upper (rotations on the left).
        for i in 0..n - 1 {
            let (cs, sn, r) = lartg(d[i], e[i]);
            d[i] = r;
            e[i] = sn * d[i + 1];
            d[i + 1] = cs * d[i + 1];
            rot_cols(u, i, i + 1, 0, nu, cs, sn);
        }
    }
    let ne = n - 1;
    bdsqr(d, &mut e[..ne], u, v)?;

    // Ascending order, one transposition per position (as `dlasdq`).
    for i in 0..n {
        let mut isub = i;
        let mut smin = d[i];
        for (j, &dj) in d.iter().enumerate().skip(i + 1) {
            if dj < smin {
                isub = j;
                smin = dj;
            }
        }
        if isub != i {
            d[isub] = d[i];
            d[i] = smin;
            swap_cols(u, isub, i, 0, nu);
            swap_cols(v, isub, i, 0, nv);
        }
    }
    Ok(())
}

/// Square `n × n` upper bidiagonal SVD by implicit QR (LAPACK `dbdsqr`,
/// relative-accuracy mode), with explicit removal of zero diagonal entries.
///
/// On success `d` holds the (unsorted) non-negative singular values and the
/// rotations have been accumulated into the columns of `u` and `v`.
fn bdsqr<R: Field + Real>(
    d: &mut [R],
    e: &mut [R],
    u: &mut Mat<R>,
    v: &mut Mat<R>,
) -> Result<(), BidiagDcError> {
    let n = d.len();
    let nv = v.nrows();
    if d.iter().chain(e.iter()).any(|x| !x.is_finite()) {
        return Err(BidiagDcError::NotConverged);
    }
    if n > 1 {
        qr_iterate(d, e, u, v)?;
    }
    // All singular values converged: make them non-negative (the sign goes
    // into the matching right singular vector, as in `dbdsqr`).
    let ld = v.row_stride();
    let vd = v.raw_data_mut();
    for (i, di) in d.iter_mut().enumerate() {
        if *di < R::zero() {
            *di = -*di;
            for x in &mut vd[i * ld..i * ld + nv] {
                *x = -*x;
            }
        }
    }
    Ok(())
}

/// The QR iteration of [`bdsqr`] for `n ≥ 2`.
fn qr_iterate<R: Field + Real>(
    d: &mut [R],
    e: &mut [R],
    u: &mut Mat<R>,
    v: &mut Mat<R>,
) -> Result<(), BidiagDcError> {
    let n = d.len();
    let nu = u.nrows();
    let nv = v.nrows();
    let zero = R::zero();
    let eps = lapack_eps::<R>();
    let unfl = <R as Scalar>::min_positive();
    let tolmul = real_const::<R>(10.0)
        .max(real_const::<R>(100.0).min(<R as Real>::powf(eps, real_const::<R>(-0.125))));
    let tol = tolmul * eps;
    let nr = real_const::<R>(n as f64);
    let maxitr = real_const::<R>(MAXITR as f64);

    // Lower bound on the smallest singular value (relative accuracy mode).
    let mut sminoa = fabs(d[0]);
    if sminoa != zero {
        let mut mu = sminoa;
        for i in 1..n {
            mu = fabs(d[i]) * (mu / (mu + fabs(e[i - 1])));
            sminoa = sminoa.min(mu);
            if sminoa == zero {
                break;
            }
        }
    }
    sminoa = sminoa / <R as Real>::sqrt(nr);
    let underflow = maxitr * (nr * (nr * unfl));
    let thresh = (tol * sminoa).max(underflow);

    let maxitdivn = MAXITR * n;
    let mut iterdivn = 0usize;
    let mut iter: isize = -1;
    let mut oldll: isize = -2;
    let mut oldm: isize = -2;
    let mut idir = 0u8;
    // `m` is the last index of the unconverged part.
    let mut m = n - 1;

    'outer: loop {
        if m == 0 {
            break;
        }
        if iter >= n as isize {
            iter -= n as isize;
            iterdivn += 1;
            if iterdivn >= maxitdivn {
                return Err(BidiagDcError::NotConverged);
            }
        }

        // Find the diagonal block [ll, m] to work on.
        let mut smax = fabs(d[m]);
        let mut split = None;
        for l in (0..m).rev() {
            let abss = fabs(d[l]);
            let abse = fabs(e[l]);
            if abse <= thresh {
                split = Some(l);
                break;
            }
            smax = smax.max(abss).max(abse);
        }
        let ll = match split {
            Some(l) => {
                e[l] = zero;
                if l == m - 1 {
                    // Bottom singular value converged.
                    m -= 1;
                    continue;
                }
                l + 1
            }
            None => 0,
        };

        // e[ll..m] are non-zero.
        if ll == m - 1 {
            // 2×2 block.
            let r = lasv2(d[m - 1], e[m - 1], d[m]);
            d[m - 1] = r.ssmax;
            e[m - 1] = zero;
            d[m] = r.ssmin;
            rot_cols(v, m - 1, m, 0, nv, r.csr, r.snr);
            rot_cols(u, m - 1, m, 0, nu, r.csl, r.snl);
            if m < 2 {
                break;
            }
            m -= 2;
            continue;
        }

        // Exactly zero (or underflowed) diagonal entries are removed explicitly.
        if let Some(k) = (ll..=m).find(|&k| fabs(d[k]) <= underflow) {
            d[k] = zero;
            if k < m {
                chase_row(d, e, u, k, m);
            } else {
                chase_column(d, e, v, ll, m);
            }
            iter += 1;
            continue;
        }

        // A new block: chase from the larger end towards the smaller one.
        if ll as isize > oldm || (m as isize) < oldll {
            idir = if fabs(d[ll]) >= fabs(d[m]) { 1 } else { 2 };
        }

        // Convergence tests.
        let mut smin;
        if idir == 1 {
            if fabs(e[m - 1]) <= tol * fabs(d[m]) {
                e[m - 1] = zero;
                continue;
            }
            let mut mu = fabs(d[ll]);
            smin = mu;
            for l in ll..m {
                if fabs(e[l]) <= tol * mu {
                    e[l] = zero;
                    continue 'outer;
                }
                mu = fabs(d[l + 1]) * (mu / (mu + fabs(e[l])));
                smin = smin.min(mu);
            }
        } else {
            if fabs(e[ll]) <= tol * fabs(d[ll]) {
                e[ll] = zero;
                continue;
            }
            let mut mu = fabs(d[m]);
            smin = mu;
            for l in (ll..m).rev() {
                if fabs(e[l]) <= tol * mu {
                    e[l] = zero;
                    continue 'outer;
                }
                mu = fabs(d[l]) * (mu / (mu + fabs(e[l])));
                smin = smin.min(mu);
            }
        }
        oldll = ll as isize;
        oldm = m as isize;

        // Shift: zero when shifting could ruin relative accuracy.
        let mut shift = zero;
        if nr * tol * (smin / smax) > eps.max(real_const::<R>(0.01) * tol) {
            let (sll, s) = if idir == 1 {
                (fabs(d[ll]), las2(d[m - 1], e[m - 1], d[m]).0)
            } else {
                (fabs(d[m]), las2(d[ll], e[ll], d[ll + 1]).0)
            };
            shift = s;
            if sll > zero && (shift / sll) * (shift / sll) < eps {
                shift = zero;
            }
        }
        iter += (m - ll) as isize;

        if shift == zero {
            if idir == 1 {
                zero_shift_down(d, e, u, v, ll, m);
                if fabs(e[m - 1]) <= thresh {
                    e[m - 1] = zero;
                }
            } else {
                zero_shift_up(d, e, u, v, ll, m);
                if fabs(e[ll]) <= thresh {
                    e[ll] = zero;
                }
            }
        } else if idir == 1 {
            shifted_down(d, e, u, v, ll, m, shift);
            if fabs(e[m - 1]) <= thresh {
                e[m - 1] = zero;
            }
        } else {
            shifted_up(d, e, u, v, ll, m, shift);
            if fabs(e[ll]) <= thresh {
                e[ll] = zero;
            }
        }
    }
    Ok(())
}

/// Removes row `k` (`d[k] = 0`, `k < m`) of the block ending at `m`: the
/// super-diagonal `e[k]` is chased to the right by left rotations of rows
/// `(j, k)`, `j = k+1..=m`, accumulated into `u`.
fn chase_row<R: Field + Real>(d: &mut [R], e: &mut [R], u: &mut Mat<R>, k: usize, m: usize) {
    let nu = u.nrows();
    let mut f = e[k];
    e[k] = R::zero();
    for j in k + 1..=m {
        if f == R::zero() {
            break;
        }
        let (c, s, r) = lartg(d[j], f);
        d[j] = r;
        rot_cols(u, j, k, 0, nu, c, s);
        if j < m {
            f = -s * e[j];
            e[j] = c * e[j];
        }
    }
}

/// Removes column `m` (`d[m] = 0`) of the block `[ll, m]`: the super-diagonal
/// `e[m − 1]` is chased upwards by right rotations of columns `(j, m)`,
/// `j = m−1..=ll`, accumulated into `v`.
fn chase_column<R: Field + Real>(d: &mut [R], e: &mut [R], v: &mut Mat<R>, ll: usize, m: usize) {
    let nv = v.nrows();
    let mut f = e[m - 1];
    e[m - 1] = R::zero();
    for j in (ll..m).rev() {
        if f == R::zero() {
            break;
        }
        let (c, s, r) = lartg(d[j], f);
        d[j] = r;
        rot_cols(v, j, m, 0, nv, c, s);
        if j > ll {
            f = -s * e[j - 1];
            e[j - 1] = c * e[j - 1];
        }
    }
}

/// Demmel–Kahan zero-shift sweep, top to bottom, on the block `[ll, m]`.
fn zero_shift_down<R: Field + Real>(
    d: &mut [R],
    e: &mut [R],
    u: &mut Mat<R>,
    v: &mut Mat<R>,
    ll: usize,
    m: usize,
) {
    let (nu, nv) = (u.nrows(), v.nrows());
    let mut cs = R::one();
    let mut oldcs = R::one();
    let mut oldsn = R::zero();
    for i in ll..m {
        let (c, sn, r) = lartg(d[i] * cs, e[i]);
        cs = c;
        if i > ll {
            e[i - 1] = oldsn * r;
        }
        let (oc, os, di) = lartg(oldcs * r, d[i + 1] * sn);
        oldcs = oc;
        oldsn = os;
        d[i] = di;
        rot_cols(v, i, i + 1, 0, nv, cs, sn);
        rot_cols(u, i, i + 1, 0, nu, oldcs, oldsn);
    }
    let h = d[m] * cs;
    d[m] = h * oldcs;
    e[m - 1] = h * oldsn;
}

/// Demmel–Kahan zero-shift sweep, bottom to top, on the block `[ll, m]`.
fn zero_shift_up<R: Field + Real>(
    d: &mut [R],
    e: &mut [R],
    u: &mut Mat<R>,
    v: &mut Mat<R>,
    ll: usize,
    m: usize,
) {
    let (nu, nv) = (u.nrows(), v.nrows());
    let mut cs = R::one();
    let mut oldcs = R::one();
    let mut oldsn = R::zero();
    for i in (ll + 1..=m).rev() {
        let (c, sn, r) = lartg(d[i] * cs, e[i - 1]);
        cs = c;
        if i < m {
            e[i] = oldsn * r;
        }
        let (oc, os, di) = lartg(oldcs * r, d[i - 1] * sn);
        oldcs = oc;
        oldsn = os;
        d[i] = di;
        rot_cols(v, i - 1, i, 0, nv, oldcs, -oldsn);
        rot_cols(u, i - 1, i, 0, nu, cs, -sn);
    }
    let h = d[ll] * cs;
    d[ll] = h * oldcs;
    e[ll] = h * oldsn;
}

/// Shifted implicit QR sweep, top to bottom, on the block `[ll, m]`.
#[allow(clippy::too_many_arguments)]
fn shifted_down<R: Field + Real>(
    d: &mut [R],
    e: &mut [R],
    u: &mut Mat<R>,
    v: &mut Mat<R>,
    ll: usize,
    m: usize,
    shift: R,
) {
    let (nu, nv) = (u.nrows(), v.nrows());
    let mut f = (fabs(d[ll]) - shift) * (sign_of(R::one(), d[ll]) + shift / d[ll]);
    let mut g = e[ll];
    for i in ll..m {
        let (cosr, sinr, r) = lartg(f, g);
        if i > ll {
            e[i - 1] = r;
        }
        f = cosr * d[i] + sinr * e[i];
        e[i] = cosr * e[i] - sinr * d[i];
        g = sinr * d[i + 1];
        d[i + 1] = cosr * d[i + 1];
        let (cosl, sinl, r) = lartg(f, g);
        d[i] = r;
        f = cosl * e[i] + sinl * d[i + 1];
        d[i + 1] = cosl * d[i + 1] - sinl * e[i];
        if i < m - 1 {
            g = sinl * e[i + 1];
            e[i + 1] = cosl * e[i + 1];
        }
        rot_cols(v, i, i + 1, 0, nv, cosr, sinr);
        rot_cols(u, i, i + 1, 0, nu, cosl, sinl);
    }
    e[m - 1] = f;
}

/// Shifted implicit QR sweep, bottom to top, on the block `[ll, m]`.
#[allow(clippy::too_many_arguments)]
fn shifted_up<R: Field + Real>(
    d: &mut [R],
    e: &mut [R],
    u: &mut Mat<R>,
    v: &mut Mat<R>,
    ll: usize,
    m: usize,
    shift: R,
) {
    let (nu, nv) = (u.nrows(), v.nrows());
    let mut f = (fabs(d[m]) - shift) * (sign_of(R::one(), d[m]) + shift / d[m]);
    let mut g = e[m - 1];
    for i in (ll + 1..=m).rev() {
        let (cosr, sinr, r) = lartg(f, g);
        if i < m {
            e[i] = r;
        }
        f = cosr * d[i] + sinr * e[i - 1];
        e[i - 1] = cosr * e[i - 1] - sinr * d[i];
        g = sinr * d[i - 1];
        d[i - 1] = cosr * d[i - 1];
        let (cosl, sinl, r) = lartg(f, g);
        d[i] = r;
        f = cosl * e[i - 1] + sinl * d[i - 1];
        d[i - 1] = cosl * d[i - 1] - sinl * e[i - 1];
        if i > ll + 1 {
            g = sinl * e[i - 2];
            e[i - 2] = cosl * e[i - 2];
        }
        rot_cols(v, i - 1, i, 0, nv, cosl, -sinl);
        rot_cols(u, i - 1, i, 0, nu, cosr, -sinr);
    }
    e[ll] = f;
}

#[cfg(test)]
#[path = "bidiag_qr_tests.rs"]
mod tests;
