//! Real Schur iteration kernels.
//!
//! - [`lanv2`]: standardization of a real 2×2 block (LAPACK `DLANV2`).
//! - [`hessenberg_qr`]: Francis double-shift QR iteration on an upper
//!   Hessenberg matrix, accumulating the Schur vectors (LAPACK `DLAHQR` with
//!   `WANTT = WANTZ = true`).
//! - [`scaling_exponent`] / [`scale_by_pow2`]: exact power-of-two scaling used
//!   to bring a matrix into a safe range before the iteration.
//!
//! The output of [`hessenberg_qr`] is in *standardized* real Schur form:
//! every entry below the first subdiagonal is exactly zero, a subdiagonal
//! entry is non-zero exactly when it belongs to a 2×2 diagonal block, and
//! every 2×2 block holds a complex-conjugate eigenvalue pair with equal
//! diagonal entries and off-diagonal entries of opposite sign.

use num_traits::Float;
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::Mat;

use super::SchurError;

/// Converts a small `f64` constant to `T`.
#[inline]
pub(crate) fn cst<T: Real>(x: f64) -> T {
    T::from_f64(x).unwrap_or_else(T::zero)
}

/// Fortran `SIGN(a, b)`: `|a|` carrying the sign of `b` (`b = ±0` gives `+|a|`).
#[inline]
fn sign<T: Real>(a: T, b: T) -> T {
    let a = Scalar::abs(a);
    if b < T::zero() { -a } else { a }
}

/// A standardized 2×2 block and the rotation that produced it.
///
/// `[aa bb; cc dd] = [cs −sn; sn cs] · [a b; c d] · [cs sn; −sn cs]`, where
/// `[aa bb; cc dd]` is the block passed to [`lanv2`] and `[a b; c d]` the
/// standardized result. Either `c = 0` (real eigenvalues `a` and `d`) or
/// `a = d` and `b·c < 0` (eigenvalues `a ± i·sqrt(|b|)·sqrt(|c|)`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Standardized2x2<T> {
    pub(crate) a: T,
    pub(crate) b: T,
    pub(crate) c: T,
    pub(crate) d: T,
    pub(crate) cs: T,
    pub(crate) sn: T,
}

/// Computes the Schur factorization of a real 2×2 nonsymmetric matrix in
/// standardized form (LAPACK `DLANV2`).
///
/// On return either `c = 0` (two real eigenvalues `a` and `d`, the block is
/// upper triangular) or `a = d` and `b·c < 0` (a complex-conjugate pair
/// `a ± i·sqrt(|b|)·sqrt(|c|)`).
pub(crate) fn lanv2<T: Field + Real>(a: T, b: T, c: T, d: T) -> Standardized2x2<T> {
    let zero = T::zero();
    let one = T::one();
    let half = cst::<T>(0.5);
    let multpl = cst::<T>(4.0);
    let eps = <T as Scalar>::epsilon();
    let safmin = <T as Scalar>::min_positive();
    // SAFMN2 = 2^trunc(log2(safmin / eps) / 2), SAFMX2 = 1 / SAFMN2.
    let half_exponent = Float::trunc(Float::log2(safmin / eps) / cst::<T>(2.0));
    let safmn2 = pow2::<T>(num_traits::ToPrimitive::to_i32(&half_exponent).unwrap_or(0));
    let safmx2 = one / safmn2;

    let (mut a, mut b, mut c, mut d) = (a, b, c, d);
    let (mut cs, mut sn);

    if c == zero {
        cs = one;
        sn = zero;
    } else if b == zero {
        // Swap rows and columns.
        cs = zero;
        sn = one;
        core::mem::swap(&mut a, &mut d);
        b = -c;
        c = zero;
    } else if (a - d) == zero && Real::signum(b) != Real::signum(c) {
        cs = one;
        sn = zero;
    } else {
        let temp = a - d;
        let mut p = half * temp;
        let bcmax = Float::max(Scalar::abs(b), Scalar::abs(c));
        let bcmis = Float::min(Scalar::abs(b), Scalar::abs(c)) * Real::signum(b) * Real::signum(c);
        let mut scale = Float::max(Scalar::abs(p), bcmax);
        let mut z = (p / scale) * p + (bcmax / scale) * bcmis;

        if z >= multpl * eps {
            // Real eigenvalues: compute a and d.
            z = p + sign(Real::sqrt(scale) * Real::sqrt(z), p);
            a = d + z;
            d = d - (bcmax / z) * bcmis;
            // Compute b and the rotation matrix.
            let tau = Real::hypot(c, z);
            cs = z / tau;
            sn = c / tau;
            b = b - c;
            c = zero;
        } else {
            // Complex eigenvalues, or real (almost) equal eigenvalues: make the
            // diagonal entries equal.
            let mut count = 0;
            let mut sigma = b + c;
            let mut temp = temp;
            loop {
                count += 1;
                scale = Float::max(Scalar::abs(temp), Scalar::abs(sigma));
                if scale >= safmx2 {
                    sigma = sigma * safmn2;
                    temp = temp * safmn2;
                    if count <= 20 {
                        continue;
                    }
                }
                if scale <= safmn2 {
                    sigma = sigma * safmx2;
                    temp = temp * safmx2;
                    if count <= 20 {
                        continue;
                    }
                }
                break;
            }
            p = half * temp;
            let tau = Real::hypot(sigma, temp);
            cs = Real::sqrt(half * (one + Scalar::abs(sigma) / tau));
            sn = -(p / (tau * cs)) * sign(one, sigma);

            // [aa bb; cc dd] = [a b; c d] · [cs −sn; sn cs]
            let aa = a * cs + b * sn;
            let bb = -a * sn + b * cs;
            let cc = c * cs + d * sn;
            let dd = -c * sn + d * cs;

            // [a b; c d] = [cs sn; −sn cs] · [aa bb; cc dd]
            a = aa * cs + cc * sn;
            b = bb * cs + dd * sn;
            c = -aa * sn + cc * cs;
            d = -bb * sn + dd * cs;

            let temp = half * (a + d);
            a = temp;
            d = temp;

            if c != zero {
                if b != zero {
                    if Real::signum(b) == Real::signum(c) {
                        // Real eigenvalues: reduce to upper triangular form.
                        let sab = Real::sqrt(Scalar::abs(b));
                        let sac = Real::sqrt(Scalar::abs(c));
                        p = sign(sab * sac, c);
                        let tau = one / Real::sqrt(Scalar::abs(b + c));
                        a = temp + p;
                        d = temp - p;
                        b = b - c;
                        c = zero;
                        let cs1 = sab * tau;
                        let sn1 = sac * tau;
                        let rotated_cs = cs * cs1 - sn * sn1;
                        sn = cs * sn1 + sn * cs1;
                        cs = rotated_cs;
                    }
                } else {
                    b = -c;
                    c = zero;
                    let swapped = cs;
                    cs = -sn;
                    sn = swapped;
                }
            }
        }
    }

    Standardized2x2 { a, b, c, d, cs, sn }
}

/// LAPACK's small-number threshold `safmin·(n/ulp)` for an order-`n`
/// problem, capped at `ulp²`.
///
/// The cap never binds for `f32`/`f64` (where the LAPACK value is below
/// `1e-30·n`), but for a format with a narrow exponent range such as `f16`
/// the LAPACK value (`≈ 0.06·n`) would exceed the rounding level of the O(1)
/// entries of a scaled matrix and treat significant entries as negligible.
pub(crate) fn small_number<T: Real>(n: usize) -> T {
    let ulp = <T as Scalar>::epsilon();
    let lapack = <T as Scalar>::min_positive() * (cst::<T>(n as f64) / ulp);
    Float::min(lapack, ulp * ulp)
}

/// Overflow threshold paired with [`small_number`]: LAPACK's
/// `(1 − ulp)/smlnum`, capped at `ulp·max_value` so that it stays well inside
/// the representable range (the cap only binds for narrow formats).
pub(crate) fn big_number<T: Real>(smlnum: T) -> T {
    let ulp = <T as Scalar>::epsilon();
    Float::min((T::one() - ulp) / smlnum, ulp * <T as Scalar>::max_value())
}

/// `2^k`, computed exactly for every `k` whose result is representable.
#[inline]
pub(crate) fn pow2<T: Real>(k: i32) -> T {
    Float::powi(cst::<T>(2.0), k)
}

/// Returns `k` such that `2^k · anrm ∈ [1, 2)`, or `0` when `anrm` is zero or
/// not finite (no scaling is applied then).
pub(crate) fn scaling_exponent<T: Real>(anrm: T) -> i32 {
    if !(anrm > T::zero()) || !Float::is_finite(anrm) {
        return 0;
    }
    let e = Float::floor(Float::log2(anrm));
    let mut k = -num_traits::ToPrimitive::to_i32(&e).unwrap_or(0);
    // `log2` is rounded (log2(f64::MAX) evaluates to exactly 1024), so the
    // estimate can be one off near a binade boundary: correct it exactly.
    let two = cst::<T>(2.0);
    for _ in 0..2 {
        let scaled = scale_by_pow2(anrm, k);
        if scaled >= two {
            k -= 1;
        } else if scaled < T::one() {
            k += 1;
        } else {
            break;
        }
    }
    k
}

/// Multiplies `x` by `2^k` exactly (barring underflow), splitting the factor in
/// two so that each partial factor is representable even when `2^k` is not.
#[inline]
pub(crate) fn scale_by_pow2<T: Real>(x: T, k: i32) -> T {
    if k == 0 {
        return x;
    }
    let k1 = k / 2;
    let k2 = k - k1;
    x * pow2::<T>(k1) * pow2::<T>(k2)
}

/// Applies the plane rotation `[x; y] ← [cs sn; −sn cs]·[x; y]` (BLAS `DROT`)
/// to rows `r1`, `r2` of `m` in the columns `cols`.
#[inline]
fn rot_rows<T: Field + Real>(
    m: &mut Mat<T>,
    r1: usize,
    r2: usize,
    cols: core::ops::Range<usize>,
    cs: T,
    sn: T,
) {
    for j in cols {
        let x = m[(r1, j)];
        let y = m[(r2, j)];
        m[(r1, j)] = cs * x + sn * y;
        m[(r2, j)] = cs * y - sn * x;
    }
}

/// Applies the plane rotation `[x, y] ← [cs·x + sn·y, cs·y − sn·x]` (BLAS
/// `DROT`) to columns `c1`, `c2` of `m` in the rows `rows`.
#[inline]
fn rot_cols<T: Field + Real>(
    m: &mut Mat<T>,
    c1: usize,
    c2: usize,
    rows: core::ops::Range<usize>,
    cs: T,
    sn: T,
) {
    for i in rows {
        let x = m[(i, c1)];
        let y = m[(i, c2)];
        m[(i, c1)] = cs * x + sn * y;
        m[(i, c2)] = cs * y - sn * x;
    }
}

/// Standardizes the 2×2 diagonal block of `t` at rows/columns `(k, k + 1)` with
/// [`lanv2`] and applies the rotation to the rest of `t` and to `q`.
pub(crate) fn standardize_block<T: Field + Real>(t: &mut Mat<T>, q: &mut Mat<T>, k: usize) {
    let n = t.nrows();
    let s = lanv2(t[(k, k)], t[(k, k + 1)], t[(k + 1, k)], t[(k + 1, k + 1)]);
    t[(k, k)] = s.a;
    t[(k, k + 1)] = s.b;
    t[(k + 1, k)] = s.c;
    t[(k + 1, k + 1)] = s.d;
    if s.cs == T::one() && s.sn == T::zero() {
        return;
    }
    rot_rows(t, k, k + 1, (k + 2)..n, s.cs, s.sn);
    rot_cols(t, k, k + 1, 0..k, s.cs, s.sn);
    let nq = q.nrows();
    rot_cols(q, k, k + 1, 0..nq, s.cs, s.sn);
}

/// Euclidean norm of a short vector without destructive overflow/underflow.
#[inline]
fn nrm2<T: Real>(x: &[T]) -> T {
    x.iter().fold(T::zero(), |acc, &xi| Real::hypot(acc, xi))
}

/// Generates an elementary reflector `H = I − tau·(1, v)(1, v)ᵀ` such that
/// `H·(alpha, x) = (beta, 0)` (LAPACK `DLARFG`). On return `x` holds `v`; the
/// function returns `(beta, tau)`.
fn larfg<T: Field + Real>(alpha: T, x: &mut [T]) -> (T, T) {
    if x.is_empty() {
        return (alpha, T::zero());
    }
    let mut xnorm = nrm2(x);
    if xnorm == T::zero() {
        return (alpha, T::zero());
    }
    let mut alpha = alpha;
    let mut beta = -sign(Real::hypot(alpha, xnorm), alpha);
    let safmin = <T as Scalar>::min_positive() / <T as Scalar>::epsilon();
    let mut knt = 0;
    if Scalar::abs(beta) < safmin {
        // xnorm and beta may be inaccurate; scale x and recompute them.
        let rsafmn = T::one() / safmin;
        loop {
            knt += 1;
            for xi in x.iter_mut() {
                *xi = *xi * rsafmn;
            }
            beta = beta * rsafmn;
            alpha = alpha * rsafmn;
            if !(Scalar::abs(beta) < safmin && knt < 20) {
                break;
            }
        }
        xnorm = nrm2(x);
        beta = -sign(Real::hypot(alpha, xnorm), alpha);
    }
    let tau = (beta - alpha) / beta;
    let scal = T::one() / (alpha - beta);
    for xi in x.iter_mut() {
        *xi = *xi * scal;
    }
    for _ in 0..knt {
        beta = beta * safmin;
    }
    (beta, tau)
}

/// Deflation test for the subdiagonal entry `t[k, k−1]` (`k ≥ 1`): the
/// conservative criterion of Ahues & Kressner used by LAPACK `DLAHQR`.
fn negligible_subdiagonal<T: Field + Real>(t: &Mat<T>, k: usize, smlnum: T, ulp: T) -> bool {
    let n = t.nrows();
    let h_sub = Scalar::abs(t[(k, k - 1)]);
    if h_sub <= smlnum {
        return true;
    }
    let mut tst = Scalar::abs(t[(k - 1, k - 1)]) + Scalar::abs(t[(k, k)]);
    if tst == T::zero() {
        if k >= 2 {
            tst = tst + Scalar::abs(t[(k - 1, k - 2)]);
        }
        if k + 1 < n {
            tst = tst + Scalar::abs(t[(k + 1, k)]);
        }
    }
    if h_sub <= ulp * tst {
        let h_sup = Scalar::abs(t[(k - 1, k)]);
        let ab = Float::max(h_sub, h_sup);
        let ba = Float::min(h_sub, h_sup);
        let diff = Scalar::abs(t[(k - 1, k - 1)] - t[(k, k)]);
        let h_kk = Scalar::abs(t[(k, k)]);
        let aa = Float::max(h_kk, diff);
        let bb = Float::min(h_kk, diff);
        let s = aa + ab;
        if ba * (ab / s) <= Float::max(smlnum, ulp * (bb * (aa / s))) {
            return true;
        }
    }
    false
}

/// Number of sweeps without a deflation after which an exceptional shift is used.
const KEXSH: usize = 10;

/// Reduces the upper Hessenberg matrix `t` to standardized real Schur form by
/// the Francis double-shift QR iteration, accumulating every orthogonal
/// transformation into `q` (`q ← q·Z`, so `A = q·t·qᵀ` holds throughout when
/// it held on entry).
///
/// `max_sweeps` bounds the *total* number of QR sweeps over all deflations;
/// deflations themselves are free. When the budget runs out while an
/// unreduced block of order ≥ 3 remains, [`SchurError::NotConverged`] is
/// returned.
pub(crate) fn hessenberg_qr<T: Field + Real>(
    t: &mut Mat<T>,
    q: &mut Mat<T>,
    max_sweeps: usize,
) -> Result<(), SchurError> {
    let n = t.nrows();
    if n == 0 {
        return Ok(());
    }
    // Clear the trash below the first subdiagonal.
    for j in 0..n {
        for i in (j + 2)..n {
            t[(i, j)] = T::zero();
        }
    }

    let ulp = <T as Scalar>::epsilon();
    let smlnum = small_number::<T>(n);

    let mut sweeps = 0usize;
    let mut kdefl = 0usize;
    let mut i = n - 1;
    loop {
        // QR sweeps on the active block [l, i] until a 1×1 or 2×2 block splits
        // off at the bottom.
        let mut l = 0usize;
        loop {
            let mut k = i;
            while k > l {
                if negligible_subdiagonal(t, k, smlnum, ulp) {
                    break;
                }
                k -= 1;
            }
            l = k;
            if l > 0 {
                t[(l, l - 1)] = T::zero();
            }
            if l + 1 >= i {
                break;
            }
            if sweeps >= max_sweeps {
                return Err(SchurError::NotConverged);
            }
            sweeps += 1;
            kdefl += 1;
            francis_sweep(t, q, l, i, kdefl, ulp);
        }

        if l < i {
            // A 2×2 block split off: standardize it.
            standardize_block(t, q, i - 1);
        }
        kdefl = 0;
        if l == 0 {
            return Ok(());
        }
        i = l - 1;
    }
}

/// Computes the shifts for a double-shift sweep on the active block `[l, i]`
/// (`i ≥ l + 2`). Exceptional shifts are used every `KEXSH` sweeps without a
/// deflation. Returns `(rt1r, rt1i, rt2r, rt2i)`.
fn double_shifts<T: Field + Real>(t: &Mat<T>, l: usize, i: usize, kdefl: usize) -> (T, T, T, T) {
    let zero = T::zero();
    let dat1 = cst::<T>(0.75);
    let dat2 = cst::<T>(-0.4375);
    let (h11, h12, h21, h22) = if kdefl.is_multiple_of(2 * KEXSH) {
        let s = Scalar::abs(t[(i, i - 1)]) + Scalar::abs(t[(i - 1, i - 2)]);
        let h11 = dat1 * s + t[(i, i)];
        (h11, dat2 * s, s, h11)
    } else if kdefl.is_multiple_of(KEXSH) {
        let s = Scalar::abs(t[(l + 1, l)]) + Scalar::abs(t[(l + 2, l + 1)]);
        let h11 = dat1 * s + t[(l, l)];
        (h11, dat2 * s, s, h11)
    } else {
        (t[(i - 1, i - 1)], t[(i - 1, i)], t[(i, i - 1)], t[(i, i)])
    };
    let s = Scalar::abs(h11) + Scalar::abs(h12) + Scalar::abs(h21) + Scalar::abs(h22);
    if s == zero {
        return (zero, zero, zero, zero);
    }
    let (h11, h12, h21, h22) = (h11 / s, h12 / s, h21 / s, h22 / s);
    let tr = (h11 + h22) / cst::<T>(2.0);
    let det = (h11 - tr) * (h22 - tr) - h12 * h21;
    let rtdisc = Real::sqrt(Scalar::abs(det));
    if det >= zero {
        // Complex conjugate shifts.
        (tr * s, rtdisc * s, tr * s, -(rtdisc * s))
    } else {
        // Real shifts: use only the one closer to h22, twice.
        let rt1 = tr + rtdisc;
        let rt2 = tr - rtdisc;
        let r = if Scalar::abs(rt1 - h22) <= Scalar::abs(rt2 - h22) {
            rt1 * s
        } else {
            rt2 * s
        };
        (r, zero, r, zero)
    }
}

/// One Francis double-shift QR sweep on the active block `[l, i]` (`i ≥ l + 2`).
fn francis_sweep<T: Field + Real>(
    t: &mut Mat<T>,
    q: &mut Mat<T>,
    l: usize,
    i: usize,
    kdefl: usize,
    ulp: T,
) {
    let n = t.nrows();
    let nq = q.nrows();
    let (rt1r, rt1i, rt2r, rt2i) = double_shifts(t, l, i, kdefl);

    // Look for two consecutive small subdiagonal entries: start the sweep at
    // the lowest row `m` where doing so keeps `t[m, m−1]` negligible.
    let mut v = [T::zero(); 3];
    let mut m = i - 2;
    loop {
        let h21s = t[(m + 1, m)];
        let s = Scalar::abs(t[(m, m)] - rt2r) + Scalar::abs(rt2i) + Scalar::abs(h21s);
        let h21s = h21s / s;
        v[0] = h21s * t[(m, m + 1)] + (t[(m, m)] - rt1r) * ((t[(m, m)] - rt2r) / s)
            - rt1i * (rt2i / s);
        v[1] = h21s * (t[(m, m)] + t[(m + 1, m + 1)] - rt1r - rt2r);
        v[2] = h21s * t[(m + 2, m + 1)];
        let s = Scalar::abs(v[0]) + Scalar::abs(v[1]) + Scalar::abs(v[2]);
        v[0] = v[0] / s;
        v[1] = v[1] / s;
        v[2] = v[2] / s;
        if m == l {
            break;
        }
        let h00 = Scalar::abs(t[(m, m - 1)]) * (Scalar::abs(v[1]) + Scalar::abs(v[2]));
        let h01 = ulp
            * Scalar::abs(v[0])
            * (Scalar::abs(t[(m - 1, m - 1)])
                + Scalar::abs(t[(m, m)])
                + Scalar::abs(t[(m + 1, m + 1)]));
        if h00 <= h01 {
            break;
        }
        m -= 1;
    }

    // Double-shift QR step: chase the bulge from row m down to row i.
    for k in m..i {
        let nr = (i - k + 1).min(3);
        if k > m {
            v[0] = t[(k, k - 1)];
            v[1] = t[(k + 1, k - 1)];
            if nr == 3 {
                v[2] = t[(k + 2, k - 1)];
            }
        }
        let (beta, tau) = larfg(v[0], &mut v[1..nr]);
        v[0] = beta;
        if k > m {
            t[(k, k - 1)] = beta;
            t[(k + 1, k - 1)] = T::zero();
            if k + 1 < i {
                t[(k + 2, k - 1)] = T::zero();
            }
        } else if m > l {
            // Equivalent to t[k, k−1] = −t[k, k−1] but safe when v[1], v[2]
            // underflow.
            t[(k, k - 1)] = t[(k, k - 1)] * (T::one() - tau);
        }
        let v2 = v[1];
        let t1 = tau;
        let t2 = t1 * v2;
        if nr == 3 {
            let v3 = v[2];
            let t3 = t1 * v3;
            for j in k..n {
                let sum = t[(k, j)] + v2 * t[(k + 1, j)] + v3 * t[(k + 2, j)];
                t[(k, j)] = t[(k, j)] - sum * t1;
                t[(k + 1, j)] = t[(k + 1, j)] - sum * t2;
                t[(k + 2, j)] = t[(k + 2, j)] - sum * t3;
            }
            for r in 0..=(k + 3).min(i) {
                let sum = t[(r, k)] + v2 * t[(r, k + 1)] + v3 * t[(r, k + 2)];
                t[(r, k)] = t[(r, k)] - sum * t1;
                t[(r, k + 1)] = t[(r, k + 1)] - sum * t2;
                t[(r, k + 2)] = t[(r, k + 2)] - sum * t3;
            }
            for r in 0..nq {
                let sum = q[(r, k)] + v2 * q[(r, k + 1)] + v3 * q[(r, k + 2)];
                q[(r, k)] = q[(r, k)] - sum * t1;
                q[(r, k + 1)] = q[(r, k + 1)] - sum * t2;
                q[(r, k + 2)] = q[(r, k + 2)] - sum * t3;
            }
        } else if nr == 2 {
            for j in k..n {
                let sum = t[(k, j)] + v2 * t[(k + 1, j)];
                t[(k, j)] = t[(k, j)] - sum * t1;
                t[(k + 1, j)] = t[(k + 1, j)] - sum * t2;
            }
            for r in 0..=i {
                let sum = t[(r, k)] + v2 * t[(r, k + 1)];
                t[(r, k)] = t[(r, k)] - sum * t1;
                t[(r, k + 1)] = t[(r, k + 1)] - sum * t2;
            }
            for r in 0..nq {
                let sum = q[(r, k)] + v2 * q[(r, k + 1)];
                q[(r, k)] = q[(r, k)] - sum * t1;
                q[(r, k + 1)] = q[(r, k + 1)] - sum * t2;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// For `f32`/`f64` the capped thresholds are exactly LAPACK's
    /// `safmin·(n/ulp)` and `(1 − ulp)/smlnum`; the caps only bind for narrow
    /// formats such as `f16`.
    #[test]
    fn small_and_big_numbers_match_lapack_for_f32_and_f64() {
        for n in [1usize, 2, 10, 100, 10_000] {
            let lapack64 = f64::MIN_POSITIVE * (n as f64 / f64::EPSILON);
            assert_eq!(small_number::<f64>(n), lapack64);
            assert_eq!(big_number(lapack64), (1.0 - f64::EPSILON) / lapack64);

            let lapack32 = f32::MIN_POSITIVE * (n as f32 / f32::EPSILON);
            assert_eq!(small_number::<f32>(n), lapack32);
            assert_eq!(big_number(lapack32), (1.0 - f32::EPSILON) / lapack32);
        }
    }

    /// `scaling_exponent` brings the largest magnitude into `[1, 2)` and leaves
    /// zero and non-finite norms unscaled; the power-of-two scaling round-trips
    /// exactly.
    #[test]
    fn scaling_exponent_targets_unit_binade() {
        for anrm in [1e-300f64, 3e-20, 0.75, 1.0, 1.5, 7.0, 1e150, f64::MAX] {
            let k = scaling_exponent(anrm);
            let scaled = scale_by_pow2(anrm, k);
            assert!((1.0..2.0).contains(&scaled), "{anrm:e} → {scaled:e}");
            assert_eq!(scale_by_pow2(scaled, -k), anrm, "exact round trip");
        }
        assert_eq!(scaling_exponent(0.0f64), 0);
        assert_eq!(scaling_exponent(f64::INFINITY), 0);
        assert_eq!(scaling_exponent(f64::NAN), 0);
    }
}
