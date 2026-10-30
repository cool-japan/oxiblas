//! 2×2 kernels of `DHGEQZ`: `DLAG2` (generalized eigenvalues of a 2×2
//! pencil with scaling against over/underflow) and `DLASV2` (SVD of a 2×2
//! upper triangular matrix).

use num_traits::Float;
use oxiblas_core::scalar::{Field, Real, Scalar};

use super::util::{Consts, sign};

/// Output of [`dlag2`]: the generalized eigenvalues are
/// `(wr1 ± i·wi) / scale1` (complex pair, `wi > 0`) or `wr1 / scale1` and
/// `wr2 / scale2` (real, `wi = 0`).
#[derive(Debug, Clone, Copy)]
pub(super) struct Lag2<T> {
    /// `SCALE1 > 0`.
    pub(super) scale1: T,
    /// `SCALE2 > 0` (equals `scale1` for a complex pair).
    pub(super) scale2: T,
    /// `WR1`: for real eigenvalues the one closest to the (2,2) element of
    /// `A·B⁻¹` (a Wilkinson-type shift); for a complex pair the real part.
    pub(super) wr1: T,
    /// `WR2` (equals `wr1` for a complex pair).
    pub(super) wr2: T,
    /// `WI ≥ 0`: imaginary part of the pair, zero for real eigenvalues.
    pub(super) wi: T,
}

/// `DLAG2`: eigenvalues of the 2×2 pencil `(A, B)` with `B` upper triangular
/// (`b[1][0]` is ignored), computed with scaling to avoid over/underflow.
///
/// `a` and `b` are indexed `[row][column]`; `safmin` is the safe minimum
/// passed by the caller (`DHGEQZ` uses `SAFMIN · SAFETY`).
pub(super) fn dlag2<T: Field + Real>(
    a: [[T; 2]; 2],
    b: [[T; 2]; 2],
    safmin: T,
    k: &Consts<T>,
) -> Lag2<T> {
    let one = T::one();
    let zero = T::zero();
    let half = k.half;
    let rtmin = Real::sqrt(safmin);
    let rtmax = one / rtmin;
    let safmax = one / safmin;

    // Scale A.
    let anorm = Float::max(
        Float::max(
            Scalar::abs(a[0][0]) + Scalar::abs(a[1][0]),
            Scalar::abs(a[0][1]) + Scalar::abs(a[1][1]),
        ),
        safmin,
    );
    let ascale = one / anorm;
    let a11 = ascale * a[0][0];
    let a21 = ascale * a[1][0];
    let a12 = ascale * a[0][1];
    let a22 = ascale * a[1][1];

    // Perturb B if necessary to insure non-singularity.
    let mut b11 = b[0][0];
    let mut b12 = b[0][1];
    let mut b22 = b[1][1];
    let bmin = rtmin
        * Float::max(
            Float::max(Scalar::abs(b11), Scalar::abs(b12)),
            Float::max(Scalar::abs(b22), rtmin),
        );
    if Scalar::abs(b11) < bmin {
        b11 = sign(bmin, b11);
    }
    if Scalar::abs(b22) < bmin {
        b22 = sign(bmin, b22);
    }

    // Scale B.
    let bnorm = Float::max(
        Float::max(Scalar::abs(b11), Scalar::abs(b12) + Scalar::abs(b22)),
        safmin,
    );
    let bsize = Float::max(Scalar::abs(b11), Scalar::abs(b22));
    let bscale = one / bsize;
    b11 = b11 * bscale;
    b12 = b12 * bscale;
    b22 = b22 * bscale;

    // Compute larger eigenvalue by the method of C. van Loan
    // (AS is A shifted by −SHIFT·B).
    let binv11 = one / b11;
    let binv22 = one / b22;
    let s1 = a11 * binv11;
    let s2 = a22 * binv22;
    let ss = a21 * (binv11 * binv22);
    let (as12, abi22, pp, shift) = if Scalar::abs(s1) <= Scalar::abs(s2) {
        let as12 = a12 - s1 * b12;
        let as22 = a22 - s1 * b22;
        let abi22 = as22 * binv22 - ss * b12;
        (as12, abi22, half * abi22, s1)
    } else {
        let as12 = a12 - s2 * b12;
        let as11 = a11 - s2 * b11;
        let abi22 = -(ss * b12);
        (as12, abi22, half * (as11 * binv11 + abi22), s2)
    };
    let qq = ss * as12;
    let (discr, r) = if Scalar::abs(pp * rtmin) >= one {
        let discr = (rtmin * pp) * (rtmin * pp) + qq * safmin;
        (discr, Real::sqrt(Scalar::abs(discr)) * rtmax)
    } else if pp * pp + Scalar::abs(qq) <= safmin {
        let discr = (rtmax * pp) * (rtmax * pp) + qq * safmax;
        (discr, Real::sqrt(Scalar::abs(discr)) * rtmin)
    } else {
        let discr = pp * pp + qq;
        (discr, Real::sqrt(Scalar::abs(discr)))
    };

    // Note: the test of R covers DISCR small, negative and flushed to zero
    // during the computation of R.
    let (mut wr1, mut wr2, mut wi) = if discr >= zero || r == zero {
        let sum = pp + sign(r, pp);
        let diff = pp - sign(r, pp);
        let wbig = shift + sum;
        // Compute smaller eigenvalue.
        let mut wsmall = shift + diff;
        if half * Scalar::abs(wbig) > Float::max(Scalar::abs(wsmall), safmin) {
            let wdet = (a11 * a22 - a12 * a21) * (binv11 * binv22);
            wsmall = wdet / wbig;
        }
        // Choose the (real) eigenvalue closest to the 2,2 element of A·B⁻¹
        // for WR1.
        if pp > abi22 {
            (Float::min(wbig, wsmall), Float::max(wbig, wsmall), zero)
        } else {
            (Float::max(wbig, wsmall), Float::min(wbig, wsmall), zero)
        }
    } else {
        // Complex eigenvalues.
        let wr = shift + pp;
        (wr, wr, r)
    };

    // Further scaling to avoid underflow and overflow in computing SCALE1
    // and overflow in computing w·B. WSCALE is bounded from above by C1/C2
    // and from below by C3/C4:
    //   C1: s·A must never overflow;   C2: w·B must never overflow;
    //   C3 (with C2): s·A − w·B must never overflow;
    //   C4: s should not underflow;    C5: max(s, |w|) should be at least 2.
    let c1 = bsize * (safmin * Float::max(one, ascale));
    let c2 = safmin * Float::max(one, bnorm);
    let c3 = bsize * safmin;
    let c4 = if ascale <= one && bsize <= one {
        Float::min(one, (ascale / safmin) * bsize)
    } else {
        one
    };
    let c5 = if ascale <= one || bsize <= one {
        Float::min(one, ascale * bsize)
    } else {
        one
    };
    // Returns (WSCALE if WSIZE ≠ 1, SCALE) for an eigenvalue of size |w|.
    let scale_for = |wabs: T| -> (Option<T>, T) {
        let wsize = Float::max(
            Float::max(safmin, c1),
            Float::max(
                k.fuzzy1 * (wabs * c2 + c3),
                Float::min(c4, half * Float::max(wabs, c5)),
            ),
        );
        if wsize != one {
            let wscale = one / wsize;
            let scale = if wsize > one {
                (Float::max(ascale, bsize) * wscale) * Float::min(ascale, bsize)
            } else {
                (Float::min(ascale, bsize) * wscale) * Float::max(ascale, bsize)
            };
            (Some(wscale), scale)
        } else {
            (None, ascale * bsize)
        }
    };

    // Scale first eigenvalue.
    let (wscale1, scale1) = scale_for(Scalar::abs(wr1) + Scalar::abs(wi));
    if let Some(wscale) = wscale1 {
        wr1 = wr1 * wscale;
        if wi != zero {
            wi = wi * wscale;
        }
    }
    let scale2 = if wi != zero {
        wr2 = wr1;
        scale1
    } else {
        // Scale second eigenvalue (if real).
        let (wscale2, scale2) = scale_for(Scalar::abs(wr2));
        if let Some(wscale) = wscale2 {
            wr2 = wr2 * wscale;
        }
        scale2
    };

    Lag2 {
        scale1,
        scale2,
        wr1,
        wr2,
        wi,
    }
}

/// Output of [`dlasv2`] for `[[f, g], [0, h]]`:
/// `[csl snl; −snl csl] · [f g; 0 h] · [csr −snr; snr csr] = diag(ssmax, ssmin)`.
#[derive(Debug, Clone, Copy)]
pub(super) struct Lasv2<T> {
    /// Smaller singular value (signed).
    pub(super) ssmin: T,
    /// Larger singular value (signed).
    pub(super) ssmax: T,
    /// Sine of the right rotation.
    pub(super) snr: T,
    /// Cosine of the right rotation.
    pub(super) csr: T,
    /// Sine of the left rotation.
    pub(super) snl: T,
    /// Cosine of the left rotation.
    pub(super) csl: T,
}

/// `DLASV2`: singular value decomposition of the 2×2 upper triangular
/// matrix `[[f, g], [0, h]]` (signed singular values, accurate to a few ulps).
pub(super) fn dlasv2<T: Field + Real>(f: T, g: T, h: T, k: &Consts<T>) -> Lasv2<T> {
    let one = T::one();
    let zero = T::zero();
    let two = k.two;

    let mut ft = f;
    let mut fa = Scalar::abs(ft);
    let mut ht = h;
    let mut ha = Scalar::abs(h);

    // PMAX points to the maximum absolute element of the matrix:
    // 1 = F, 2 = G, 3 = H.
    let mut pmax = 1u8;
    let swap = ha > fa;
    if swap {
        pmax = 3;
        core::mem::swap(&mut ft, &mut ht);
        core::mem::swap(&mut fa, &mut ha);
        // Now FA >= HA.
    }
    let gt = g;
    let ga = Scalar::abs(gt);
    if ga != zero && ga > fa {
        pmax = 2;
    }

    let (ssmin, ssmax, clt, crt, slt, srt) = if ga == zero {
        // Diagonal matrix.
        (ha, fa, one, one, zero, zero)
    } else if ga > fa && fa / ga < k.unit_roundoff {
        // Case of very large GA (DLAMCH('EPS') is the unit roundoff).
        let ssmin = if ha > one {
            fa / (ga / ha)
        } else {
            (fa / ga) * ha
        };
        (ssmin, ga, one, ft / gt, ht / gt, one)
    } else {
        // Normal case.
        let d = fa - ha;
        // Copes with infinite F or H; 0 <= L <= 1.
        let l = if d == fa { one } else { d / fa };
        // abs(M) <= 1/macheps.
        let m = gt / ft;
        // T >= 1.
        let t = two - l;
        let mm = m * m;
        let tt = t * t;
        // 1 <= S <= 1 + 1/macheps.
        let s = Real::sqrt(tt + mm);
        // 0 <= R <= 1 + 1/macheps.
        let r = if l == zero {
            Scalar::abs(m)
        } else {
            Real::sqrt(l * l + mm)
        };
        // 1 <= A <= 1 + abs(M).
        let a = k.half * (s + r);
        let ssmin = ha / a;
        let ssmax = fa * a;
        let t = if mm == zero {
            // M is very tiny.
            if l == zero {
                sign(two, ft) * sign(one, gt)
            } else {
                gt / sign(d, ft) + m / t
            }
        } else {
            (m / (s + t) + m / (r + l)) * (one + a)
        };
        let l = Real::sqrt(t * t + k.four);
        let crt = two / l;
        let srt = t / l;
        let clt = (crt + srt * m) / a;
        let slt = (ht / ft) * srt / a;
        (ssmin, ssmax, clt, crt, slt, srt)
    };

    let (csl, snl, csr, snr) = if swap {
        (srt, crt, slt, clt)
    } else {
        (clt, slt, crt, srt)
    };

    // Correct signs of SSMAX and SSMIN.
    let tsign = match pmax {
        1 => sign(one, csr) * sign(one, csl) * sign(one, f),
        2 => sign(one, snr) * sign(one, csl) * sign(one, g),
        _ => sign(one, snr) * sign(one, snl) * sign(one, h),
    };
    Lasv2 {
        ssmin: sign(ssmin, tsign * sign(one, f) * sign(one, h)),
        ssmax: sign(ssmax, tsign),
        snr,
        csr,
        snl,
        csl,
    }
}
