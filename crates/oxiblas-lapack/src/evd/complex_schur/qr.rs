//! Complex Schur iteration kernel: implicit single-shift QR on an upper
//! Hessenberg matrix with Wilkinson and exceptional shifts and the
//! Ahues–Kressner deflation criterion (the strategy of LAPACK `ZLAHQR`,
//! realised with complex Givens rotations), accumulating the Schur vectors.
//!
//! On success the matrix is upper triangular (every entry below the diagonal
//! is exactly zero) and holds the eigenvalues on its diagonal.

use num_traits::{Float, One, Zero};
use oxiblas_core::scalar::{ComplexScalar, Field, Real, Scalar};
use oxiblas_matrix::Mat;

use super::ComplexSchurError;
use crate::evd::schur::{ladiv, small_number};

/// Converts a small `f64` constant to the real type `R`.
#[inline]
fn rcst<R: Real>(x: f64) -> R {
    R::from_f64(x).unwrap_or_else(R::zero)
}

/// `|re| + |im|` (LAPACK `CABS1`).
#[inline]
pub(crate) fn cabs1<T: Scalar>(z: T) -> T::Real {
    Scalar::abs(z.real()) + Scalar::abs(z.imag())
}

/// Robust complex division `a / b` (Smith's algorithm on the components).
#[inline]
pub(crate) fn cdiv<T: Field + ComplexScalar>(a: T, b: T) -> T {
    let (re, im) = ladiv(a.real(), a.imag(), b.real(), b.imag());
    T::from_real_imag(re, im)
}

/// `z / s` for a real `s`, component-wise.
#[inline]
fn div_real<T: Field + ComplexScalar>(z: T, s: T::Real) -> T {
    T::from_real_imag(z.real() / s, z.imag() / s)
}

/// Complex plane rotation `G = [c s; −conj(s) c]` (`c` real, `G` unitary)
/// with `G·[f; g] = [r; 0]`. Returns `(c, s, r)`.
fn givens<T: Field + ComplexScalar>(f: T, g: T) -> (T::Real, T, T) {
    let zero = T::Real::zero();
    let one = T::Real::one();
    let g_abs = Scalar::abs(g);
    if g_abs == zero {
        return (one, T::zero(), f);
    }
    let f_abs = Scalar::abs(f);
    if f_abs == zero {
        return (
            zero,
            T::from_real_imag(g.real() / g_abs, -(g.imag() / g_abs)),
            T::from_real(g_abs),
        );
    }
    let norm = Real::hypot(f_abs, g_abs);
    let c = f_abs / norm;
    let phase = div_real(f, f_abs);
    let s = phase * T::from_real_imag(g.real() / norm, -(g.imag() / norm));
    let r = phase * T::from_real(norm);
    (c, s, r)
}

/// Deflation test for the subdiagonal entry `h[k, k−1]` (`k ≥ 1`): the
/// conservative criterion of Ahues & Kressner used by LAPACK `ZLAHQR`.
fn negligible_subdiagonal<T: Field + ComplexScalar>(
    h: &Mat<T>,
    k: usize,
    smlnum: T::Real,
    ulp: T::Real,
) -> bool {
    let n = h.nrows();
    let h_sub = cabs1(h[(k, k - 1)]);
    if h_sub <= smlnum {
        return true;
    }
    let mut tst = cabs1(h[(k - 1, k - 1)]) + cabs1(h[(k, k)]);
    if tst == T::Real::zero() {
        if k >= 2 {
            tst = tst + cabs1(h[(k - 1, k - 2)]);
        }
        if k + 1 < n {
            tst = tst + cabs1(h[(k + 1, k)]);
        }
    }
    if h_sub <= ulp * tst {
        let h_sup = cabs1(h[(k - 1, k)]);
        let ab = Float::max(h_sub, h_sup);
        let ba = Float::min(h_sub, h_sup);
        let diff = cabs1(h[(k - 1, k - 1)] - h[(k, k)]);
        let h_kk = cabs1(h[(k, k)]);
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

/// Reduces the upper Hessenberg matrix `h` to upper triangular (complex Schur)
/// form, accumulating every unitary transformation into `q` (`q ← q·Z`, so
/// `A = q·h·qᴴ` holds throughout when it held on entry).
///
/// `max_sweeps` bounds the *total* number of QR sweeps over all deflations;
/// when it runs out before `h` is triangular, [`ComplexSchurError::NotConverged`]
/// is returned.
pub(crate) fn complex_hessenberg_qr<T: Field + ComplexScalar>(
    h: &mut Mat<T>,
    q: &mut Mat<T>,
    max_sweeps: usize,
) -> Result<(), ComplexSchurError> {
    let n = h.nrows();
    if n == 0 {
        return Ok(());
    }
    // Clear the trash below the first subdiagonal.
    for j in 0..n {
        for i in (j + 2)..n {
            h[(i, j)] = T::zero();
        }
    }

    let ulp = <T::Real as Scalar>::epsilon();
    let smlnum = small_number::<T::Real>(n);

    let mut sweeps = 0usize;
    let mut kdefl = 0usize;
    let mut i = n - 1;
    loop {
        // Sweeps on the active block [l, i] until h[i, i−1] is negligible.
        let mut l = 0usize;
        loop {
            let mut k = i;
            while k > l {
                if negligible_subdiagonal(h, k, smlnum, ulp) {
                    break;
                }
                k -= 1;
            }
            l = k;
            if l > 0 {
                h[(l, l - 1)] = T::zero();
            }
            if l >= i {
                break;
            }
            if sweeps >= max_sweeps {
                return Err(ComplexSchurError::NotConverged);
            }
            sweeps += 1;
            kdefl += 1;
            single_shift_sweep(h, q, l, i, kdefl, ulp);
        }
        kdefl = 0;
        if i == 0 {
            return Ok(());
        }
        i -= 1;
    }
}

/// Shift for a sweep on the active block `[l, i]` (`i ≥ l + 1`): the
/// Wilkinson shift, or an exceptional shift every `KEXSH` sweeps without a
/// deflation.
fn shift<T: Field + ComplexScalar>(h: &Mat<T>, l: usize, i: usize, kdefl: usize) -> T {
    let zero = T::Real::zero();
    let dat1 = rcst::<T::Real>(0.75);
    if kdefl.is_multiple_of(2 * KEXSH) {
        return h[(i, i)] + T::from_real(dat1 * Scalar::abs(h[(i, i - 1)]));
    }
    if kdefl.is_multiple_of(KEXSH) {
        return h[(l, l)] + T::from_real(dat1 * Scalar::abs(h[(l + 1, l)]));
    }
    // Wilkinson's shift: the eigenvalue of the trailing 2×2 block closer to
    // h[i, i], computed without overflow (LAPACK ZLAHQR).
    let mut t = h[(i, i)];
    let u = h[(i - 1, i)].csqrt() * h[(i, i - 1)].csqrt();
    let s = cabs1(u);
    if s != zero {
        let x = (h[(i - 1, i - 1)] - t) * T::from_real(rcst::<T::Real>(0.5));
        let sx = cabs1(x);
        let s = Float::max(s, sx);
        let xs = div_real(x, s);
        let us = div_real(u, s);
        let mut y = (xs * xs + us * us).csqrt() * T::from_real(s);
        if sx > zero {
            let xn = div_real(x, sx);
            if xn.real() * y.real() + xn.imag() * y.imag() < zero {
                y = -y;
            }
        }
        t = t - u * cdiv(u, x + y);
    }
    t
}

/// One implicit single-shift QR sweep on the active block `[l, i]`.
fn single_shift_sweep<T: Field + ComplexScalar>(
    h: &mut Mat<T>,
    q: &mut Mat<T>,
    l: usize,
    i: usize,
    kdefl: usize,
    ulp: T::Real,
) {
    let n = h.nrows();
    let nq = q.nrows();
    let sigma = shift(h, l, i, kdefl);

    // Look for two consecutive small subdiagonal entries: start the sweep at
    // the lowest row `m` where doing so keeps `h[m, m−1]` negligible.
    let mut m = i - 1;
    while m > l {
        let h11 = h[(m, m)];
        let h22 = h[(m + 1, m + 1)];
        let h11s = cabs1(h11 - sigma);
        let h21 = cabs1(h[(m + 1, m)]);
        let s = h11s + h21;
        let h10 = cabs1(h[(m, m - 1)]);
        if h10 * (h21 / s) <= ulp * ((h11s / s) * (cabs1(h11) + cabs1(h22))) {
            break;
        }
        m -= 1;
    }

    // Chase the bulge from row m down to row i.
    for k in m..i {
        let (f, g) = if k == m {
            (h[(m, m)] - sigma, h[(m + 1, m)])
        } else {
            (h[(k, k - 1)], h[(k + 1, k - 1)])
        };
        let (c, s, r) = givens(f, g);
        let cc = T::from_real(c);
        let sc = s.conj();
        if k > m {
            h[(k, k - 1)] = r;
            h[(k + 1, k - 1)] = T::zero();
        } else if m > l {
            // The rotation scales h[m, m−1] by c; the entry it would create at
            // (m + 1, m − 1) is negligible by the choice of m.
            h[(m, m - 1)] = h[(m, m - 1)] * cc;
        }
        for j in k..n {
            let a = h[(k, j)];
            let b = h[(k + 1, j)];
            h[(k, j)] = cc * a + s * b;
            h[(k + 1, j)] = cc * b - sc * a;
        }
        for row in 0..=(k + 2).min(i) {
            let a = h[(row, k)];
            let b = h[(row, k + 1)];
            h[(row, k)] = cc * a + sc * b;
            h[(row, k + 1)] = cc * b - s * a;
        }
        for row in 0..nq {
            let a = q[(row, k)];
            let b = q[(row, k + 1)];
            q[(row, k)] = cc * a + sc * b;
            q[(row, k + 1)] = cc * b - s * a;
        }
    }
}
