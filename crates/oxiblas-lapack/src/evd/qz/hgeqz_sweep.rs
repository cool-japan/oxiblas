//! Shift selection and QZ sweeps of `DHGEQZ` (label 110 onwards).
//!
//! * `select_shift`: exceptional shift every 10th step, otherwise the
//!   `DLAG2` eigenvalue of the trailing 2×2 block closest to `H(ilast,ilast)`.
//! * `single_shift_sweep`: implicit single-shift QZ sweep (loops 120–190).
//! * `standardize_2x2`: 2×2 block with complex eigenvalues — `DLASV2`
//!   diagonalization of the `T` block and the EISPACK `QZVAL` formulas for
//!   `(ALPHAR, ALPHAI, BETA)` (label 200, IFIRST + 1 = ILAST).
//! * `double_shift_sweep`: Francis implicit double-shift sweep for blocks
//!   of order at least 3 (loops 230–340, with the `DLAGBC` pivoted 2×2
//!   solve that zeroes a column of `T`).

use num_traits::Float;
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::Mat;

use super::hgeqz::QzTol;
use super::lag2::{dlag2, dlasv2};
use super::pipeline::QzWork;
use super::util::{Consts, lapy2, lapy3, larfg, lartg, rot_cols, rot_rows};

/// Shift chosen for one QZ step.
pub(super) enum Shift<T> {
    /// Real shift `wr / s1`: implicit single-shift sweep.
    Real {
        /// Scale `s` of the shift (`s1 > 0`).
        s1: T,
        /// Numerator `w` of the shift.
        wr: T,
    },
    /// Complex conjugate shifts (label 200): 2×2 standardization or a
    /// Francis double-shift sweep.
    Complex,
}

/// The 2×2 block `m(i..=i+1, i..=i+1)` as `[row][column]`.
fn block2<T: Field + Real>(m: &Mat<T>, i: usize) -> [[T; 2]; 2] {
    [
        [m[(i, i)], m[(i, i + 1)]],
        [m[(i + 1, i)], m[(i + 1, i + 1)]],
    ]
}

/// Computes the shift for a QZ step on a block ending at `ilast`
/// (`iiter` already incremented; `maxit_t` is `DBLE(MAXIT)`).
pub(super) fn select_shift<T: Field + Real>(
    w: &QzWork<T>,
    k: &Consts<T>,
    ilast: usize,
    iiter: usize,
    maxit_t: T,
    eshift: &mut T,
) -> Shift<T> {
    if iiter % 10 == 0 {
        // Exceptional shift. Chosen for no particularly good reason
        // (single shift only).
        let h_sub = w.h[(ilast, ilast - 1)];
        let t_prev = w.t[(ilast - 1, ilast - 1)];
        if (maxit_t * k.safmin) * Scalar::abs(h_sub) < Scalar::abs(t_prev) {
            *eshift = h_sub / t_prev;
        } else {
            *eshift = *eshift + T::one() / (k.safmin * maxit_t);
        }
        return Shift::Real {
            s1: T::one(),
            wr: *eshift,
        };
    }
    // Shifts based on the generalized eigenvalues of the bottom-right 2×2
    // block of A and B; the first eigenvalue returned by DLAG2 is the
    // Wilkinson shift (AEP p. 512). (The reference's subsequent
    // `TEMP = MAX(S1, ...)` is dead code and omitted.)
    let g = dlag2(
        block2(&w.h, ilast - 1),
        block2(&w.t, ilast - 1),
        k.safmin * k.safety,
        k,
    );
    if g.wi != T::zero() {
        return Shift::Complex;
    }
    let t_ll = w.t[(ilast, ilast)];
    let h_ll = w.h[(ilast, ilast)];
    if Scalar::abs((g.wr1 / g.scale1) * t_ll - h_ll) > Scalar::abs((g.wr2 / g.scale2) * t_ll - h_ll)
    {
        Shift::Real {
            s1: g.scale2,
            wr: g.wr2,
        }
    } else {
        Shift::Real {
            s1: g.scale1,
            wr: g.wr1,
        }
    }
}

/// Implicit single-shift QZ sweep with shift `wr / s1` on `ifirst..=ilast`
/// (shift overflow guard, loop 120 start search and loops 140–190).
pub(super) fn single_shift_sweep<T: Field + Real>(
    w: &mut QzWork<T>,
    tol: &QzTol<T>,
    k: &Consts<T>,
    ifirst: usize,
    ilast: usize,
    s1: T,
    wr: T,
) {
    let n = w.h.nrows();

    // Fiddle with shift to avoid overflow.
    let temp = Float::min(tol.ascale, T::one()) * (k.half * k.safmax);
    let mut scale = if s1 > temp { temp / s1 } else { T::one() };
    let temp = Float::min(tol.bscale, T::one()) * (k.half * k.safmax);
    if Scalar::abs(wr) > temp {
        scale = Float::min(scale, temp / Scalar::abs(wr));
    }
    let s1 = scale * s1;
    let wr = scale * wr;

    // Now check for two consecutive small subdiagonals (loop 120).
    let mut istart = ifirst;
    for j in ((ifirst + 1)..ilast).rev() {
        let mut temp = Scalar::abs(s1 * w.h[(j, j - 1)]);
        let mut temp2 = Scalar::abs(s1 * w.h[(j, j)] - wr * w.t[(j, j)]);
        let tempr = Float::max(temp, temp2);
        if tempr < T::one() && tempr != T::zero() {
            temp = temp / tempr;
            temp2 = temp2 / tempr;
        }
        if Scalar::abs((tol.ascale * w.h[(j + 1, j)]) * temp) <= (tol.ascale * tol.atol) * temp2 {
            istart = j;
            break;
        }
    }

    // Do an implicit single-shift QZ sweep. Initial Q.
    let temp = s1 * w.h[(istart, istart)] - wr * w.t[(istart, istart)];
    let temp2 = s1 * w.h[(istart + 1, istart)];
    let (c0, s0, _) = lartg(temp, temp2, k);

    // Sweep (loop 190).
    for j in istart..ilast {
        let (c, s) = if j > istart {
            let temp = w.h[(j, j - 1)];
            let (c, s, r) = lartg(temp, w.h[(j + 1, j - 1)], k);
            w.h[(j, j - 1)] = r;
            w.h[(j + 1, j - 1)] = T::zero();
            (c, s)
        } else {
            (c0, s0)
        };
        // Loop 140: rows j, j+1 of H and T over columns j..=ILASTM.
        rot_rows(&mut w.h, j, j + 1, j..n, c, s);
        rot_rows(&mut w.t, j, j + 1, j..n, c, s);
        // Loop 150: columns j, j+1 of Q.
        rot_cols(&mut w.q, j, j + 1, 0..n, c, s);

        let temp = w.t[(j + 1, j + 1)];
        let (c, s, r) = lartg(temp, w.t[(j + 1, j)], k);
        w.t[(j + 1, j + 1)] = r;
        w.t[(j + 1, j)] = T::zero();
        // Loop 160: columns j+1, j of H over rows IFRSTM..=min(j+2, ILAST).
        rot_cols(&mut w.h, j + 1, j, 0..((j + 2).min(ilast) + 1), c, s);
        // Loop 170: columns j+1, j of T over rows IFRSTM..=j.
        rot_cols(&mut w.t, j + 1, j, 0..(j + 1), c, s);
        // Loop 180: columns j+1, j of Z.
        rot_cols(&mut w.z, j + 1, j, 0..n, c, s);
    }
}

/// Label 200 with `IFIRST + 1 = ILAST`: the 2×2 block ending at `ilast` has
/// complex eigenvalues.
///
/// Step 1 rotates the block so that its `T` part is `diag(B11, B22)` with
/// `B11, B22 ≥ 0` (`DLASV2`); step 2 recomputes the shift and, if it is
/// still complex, evaluates `(ALPHAR, ALPHAI, BETA)` for both members with
/// the EISPACK `QZVAL` formulas (normalized so `beta > 0` and the first
/// member has `alpha_imag > 0`) and returns `true` (the pair deflates).
/// Returns `false` when standardization moved the shift onto the real
/// line; the caller then continues with real single shifts.
pub(super) fn standardize_2x2<T: Field + Real>(
    w: &mut QzWork<T>,
    k: &Consts<T>,
    ilast: usize,
) -> bool {
    let n = w.h.nrows();
    let one = T::one();
    let zero = T::zero();
    // IFIRST = ILAST − 1.
    let i = ilast - 1;

    // Step 1: standardize, i.e. rotate so that T = diag(B11, B22), B11 ≥ 0.
    let sv = dlasv2(w.t[(i, i)], w.t[(i, ilast)], w.t[(ilast, ilast)], k);
    let (mut b22, mut b11, mut sr, mut cr) = (sv.ssmin, sv.ssmax, sv.snr, sv.csr);
    let (sl, cl) = (sv.snl, sv.csl);
    if b11 < zero {
        cr = -cr;
        sr = -sr;
        b11 = -b11;
        b22 = -b22;
    }
    // DROT(ILASTM+1−IFIRST, H(ILAST−1,ILAST−1), LDH, H(ILAST,ILAST−1), LDH, CL, SL)
    rot_rows(&mut w.h, i, ilast, i..n, cl, sl);
    // DROT(ILAST+1−IFRSTM, H(IFRSTM,ILAST−1), 1, H(IFRSTM,ILAST), 1, CR, SR)
    rot_cols(&mut w.h, i, ilast, 0..(ilast + 1), cr, sr);
    // DROT(ILASTM−ILAST, T(ILAST−1,ILAST+1), LDT, T(ILAST,ILAST+1), LDT, CL, SL)
    rot_rows(&mut w.t, i, ilast, (ilast + 1)..n, cl, sl);
    // DROT(IFIRST−IFRSTM, T(IFRSTM,ILAST−1), 1, T(IFRSTM,ILAST), 1, CR, SR)
    rot_cols(&mut w.t, i, ilast, 0..i, cr, sr);
    // DROT(N, Q(1,ILAST−1), 1, Q(1,ILAST), 1, CL, SL)
    rot_cols(&mut w.q, i, ilast, 0..n, cl, sl);
    // DROT(N, Z(1,ILAST−1), 1, Z(1,ILAST), 1, CR, SR)
    rot_cols(&mut w.z, i, ilast, 0..n, cr, sr);

    w.t[(i, i)] = b11;
    w.t[(i, ilast)] = zero;
    w.t[(ilast, i)] = zero;
    w.t[(ilast, ilast)] = b22;

    // If B22 is negative, negate column ILAST (loops 210/220).
    if b22 < zero {
        for j in 0..=ilast {
            w.h[(j, ilast)] = -w.h[(j, ilast)];
            w.t[(j, ilast)] = -w.t[(j, ilast)];
        }
        for j in 0..n {
            w.z[(j, ilast)] = -w.z[(j, ilast)];
        }
        b22 = -b22;
    }

    // Step 2: compute ALPHAR, ALPHAI and BETA. Recompute the shift.
    let g = dlag2(block2(&w.h, i), block2(&w.t, i), k.safmin * k.safety, k);
    // If standardization has perturbed the shift onto the real line, do
    // another (real single-shift) QR step.
    if g.wi == zero {
        return false;
    }
    let (s1, wr, wi) = (g.scale1, g.wr1, g.wi);
    let s1inv = one / s1;

    // EISPACK QZVAL computation of alpha and beta.
    let a11 = w.h[(i, i)];
    let a21 = w.h[(ilast, i)];
    let a12 = w.h[(i, ilast)];
    let a22 = w.h[(ilast, ilast)];

    // Complex Givens rotation on the right (assume some element of
    // C = s·A − w·B exceeds the underflow threshold):
    //   (sA − wB) · [CZ  −conj(SZ); SZ  CZ].
    let c11r = s1 * a11 - wr * b11;
    let c11i = -(wi * b11);
    let c12 = s1 * a12;
    let c21 = s1 * a21;
    let c22r = s1 * a22 - wr * b22;
    let c22i = -(wi * b22);
    let abs = |x: T| Scalar::abs(x);
    let (cz, szr, szi) = if abs(c11r) + abs(c11i) + abs(c12) > abs(c21) + abs(c22r) + abs(c22i) {
        let t1 = lapy3(c12, c11r, c11i);
        (c12 / t1, -c11r / t1, -c11i / t1)
    } else {
        let cz = lapy2(c22r, c22i);
        if cz <= k.safmin {
            (zero, one, zero)
        } else {
            let tempr = c22r / cz;
            let tempi = c22i / cz;
            let t1 = lapy2(cz, c21);
            (cz / t1, -c21 * tempr / t1, c21 * tempi / t1)
        }
    };

    // Givens rotation on the left: [CQ  SQ; −conj(SQ)  CQ] applied to A or B.
    let an = abs(a11) + abs(a12) + abs(a21) + abs(a22);
    let bn = abs(b11) + abs(b22);
    let wabs = abs(wr) + abs(wi);
    let (cq, sqr, sqi) = if s1 * an > wabs * bn {
        (cz * b11, szr * b22, -(szi * b22))
    } else {
        let a1r = cz * a11 + szr * a12;
        let a1i = szi * a12;
        let a2r = cz * a21 + szr * a22;
        let a2i = szi * a22;
        let cq = lapy2(a1r, a1i);
        if cq <= k.safmin {
            (zero, one, zero)
        } else {
            let tempr = a1r / cq;
            let tempi = a1i / cq;
            (cq, tempr * a2r + tempi * a2i, tempi * a2r - tempr * a2i)
        }
    };
    let t1 = lapy3(cq, sqr, sqi);
    let (cq, sqr, sqi) = (cq / t1, sqr / t1, sqi / t1);

    // Compute the diagonal elements of Q·B·Z.
    let tempr = sqr * szr - sqi * szi;
    let tempi = sqr * szi + sqi * szr;
    let b1r = cq * cz * b11 + tempr * b22;
    let b1i = tempi * b22;
    let b1a = lapy2(b1r, b1i);
    let b2r = cq * cz * b22 + tempr * b11;
    let b2i = -(tempi * b11);
    let b2a = lapy2(b2r, b2i);

    // Normalize so beta > 0 and Im(alpha1) > 0.
    w.beta[i] = b1a;
    w.beta[ilast] = b2a;
    w.alphar[i] = (wr * b1a) * s1inv;
    w.alphai[i] = (wi * b1a) * s1inv;
    w.alphar[ilast] = (wr * b2a) * s1inv;
    w.alphai[ilast] = -((wi * b2a) * s1inv);
    true
}

/// `DLAGBC`-style pivoted 2×2 solve used to zero column `j` of `T`: returns
/// `(scale, u1, u2)` with `T(j+1..=j+2, j+1..=j+2) · [u1; u2] =
/// scale · T(j+1..=j+2, j)`, `0 ≤ scale ≤ 1` chosen to avoid overflow.
fn zero_column_vector<T: Field + Real>(t: &Mat<T>, j: usize, k: &Consts<T>) -> (T, T, T) {
    let one = T::one();
    let zero = T::zero();
    let abs = |x: T| Scalar::abs(x);

    // Swap rows to pivot.
    let temp = Float::max(abs(t[(j + 1, j + 1)]), abs(t[(j + 1, j + 2)]));
    let temp2 = Float::max(abs(t[(j + 2, j + 1)]), abs(t[(j + 2, j + 2)]));
    if Float::max(temp, temp2) < k.safmin {
        return (zero, one, zero);
    }
    let (mut w11, mut w21, mut w12, mut w22, mut u1, mut u2) = if temp >= temp2 {
        (
            t[(j + 1, j + 1)],
            t[(j + 2, j + 1)],
            t[(j + 1, j + 2)],
            t[(j + 2, j + 2)],
            t[(j + 1, j)],
            t[(j + 2, j)],
        )
    } else {
        (
            t[(j + 2, j + 1)],
            t[(j + 1, j + 1)],
            t[(j + 2, j + 2)],
            t[(j + 1, j + 2)],
            t[(j + 2, j)],
            t[(j + 1, j)],
        )
    };

    // Swap columns if necessary.
    let ilpivt = abs(w12) > abs(w11);
    if ilpivt {
        core::mem::swap(&mut w11, &mut w12);
        core::mem::swap(&mut w21, &mut w22);
    }

    // LU-factor (W21 becomes zero).
    let temp = w21 / w11;
    u2 = u2 - temp * u1;
    w22 = w22 - temp * w12;

    // Compute SCALE and solve (label 250 follows either branch).
    let mut scale = one;
    if abs(w22) < k.safmin {
        scale = zero;
        u2 = one;
        u1 = -w12 / w11;
    } else {
        if abs(w22) < abs(u2) {
            scale = abs(w22 / u2);
        }
        if abs(w11) < abs(u1) {
            scale = Float::min(scale, abs(w11 / u1));
        }
        u2 = (scale * u2) / w22;
        u1 = (scale * u1 - w12 * u2) / w11;
    }
    if ilpivt {
        core::mem::swap(&mut u1, &mut u2);
    }
    (scale, u1, u2)
}

/// Applies `I − tau·v·vᵀ` (`v = [1, v1, v2]`) from the left to rows
/// `j..=j+2` of `m` over the columns `cols` (loop 230 body).
fn reflect3_rows<T: Field + Real>(
    m: &mut Mat<T>,
    j: usize,
    v: [T; 3],
    tau: T,
    cols: core::ops::Range<usize>,
) {
    let t2 = tau * v[1];
    let t3 = tau * v[2];
    for jc in cols {
        let temp = m[(j, jc)] + v[1] * m[(j + 1, jc)] + v[2] * m[(j + 2, jc)];
        m[(j, jc)] = m[(j, jc)] - temp * tau;
        m[(j + 1, jc)] = m[(j + 1, jc)] - temp * t2;
        m[(j + 2, jc)] = m[(j + 2, jc)] - temp * t3;
    }
}

/// Applies `I − tau·v·vᵀ` (`v = [1, v1, v2]`) from the right to columns
/// `j..=j+2` of `m` over the rows `rows` (loops 240/260/270/280 bodies).
fn reflect3_cols<T: Field + Real>(
    m: &mut Mat<T>,
    j: usize,
    v: [T; 3],
    tau: T,
    rows: core::ops::Range<usize>,
) {
    let t2 = tau * v[1];
    let t3 = tau * v[2];
    for jr in rows {
        let temp = m[(jr, j)] + v[1] * m[(jr, j + 1)] + v[2] * m[(jr, j + 2)];
        m[(jr, j)] = m[(jr, j)] - temp * tau;
        m[(jr, j + 1)] = m[(jr, j + 1)] - temp * t2;
        m[(jr, j + 2)] = m[(jr, j + 2)] - temp * t3;
    }
}

/// Francis implicit double-shift QZ sweep on `ifirst..=ilast`
/// (`ilast − ifirst ≥ 2`), label 200 "usual case".
///
/// The first column of `(A·B⁻¹)² − c·A·B⁻¹ + d·I` is formed with the
/// EISPACK `QZIT` formula, then the bulge is chased with 3×3 Householder
/// reflectors from the left (rows of `H`, `T`, columns of `Q`) and from the
/// right (zeroing column `j` of `T` via `zero_column_vector`), finishing
/// with two Givens rotations.
pub(super) fn double_shift_sweep<T: Field + Real>(
    w: &mut QzWork<T>,
    tol: &QzTol<T>,
    k: &Consts<T>,
    ifirst: usize,
    ilast: usize,
) {
    let n = w.h.nrows();
    let one = T::one();
    let zero = T::zero();
    let (asc, bsc) = (tol.ascale, tol.bscale);
    let (h, t) = (&w.h, &w.t);

    // Eigenvalue equation is w² − c·w + d = 0, so compute the first column
    // of (A·B⁻¹)² − c·A·B⁻¹ + d using the formula in QZIT (from EISPACK).
    let ad11 = (asc * h[(ilast - 1, ilast - 1)]) / (bsc * t[(ilast - 1, ilast - 1)]);
    let ad21 = (asc * h[(ilast, ilast - 1)]) / (bsc * t[(ilast - 1, ilast - 1)]);
    let ad12 = (asc * h[(ilast - 1, ilast)]) / (bsc * t[(ilast, ilast)]);
    let ad22 = (asc * h[(ilast, ilast)]) / (bsc * t[(ilast, ilast)]);
    let u12 = t[(ilast - 1, ilast)] / t[(ilast, ilast)];
    let ad11l = (asc * h[(ifirst, ifirst)]) / (bsc * t[(ifirst, ifirst)]);
    let ad21l = (asc * h[(ifirst + 1, ifirst)]) / (bsc * t[(ifirst, ifirst)]);
    let ad12l = (asc * h[(ifirst, ifirst + 1)]) / (bsc * t[(ifirst + 1, ifirst + 1)]);
    let ad22l = (asc * h[(ifirst + 1, ifirst + 1)]) / (bsc * t[(ifirst + 1, ifirst + 1)]);
    let ad32l = (asc * h[(ifirst + 2, ifirst + 1)]) / (bsc * t[(ifirst + 1, ifirst + 1)]);
    let u12l = t[(ifirst, ifirst + 1)] / t[(ifirst + 1, ifirst + 1)];

    let mut alpha = (ad11 - ad11l) * (ad22 - ad11l) - ad12 * ad21
        + ad21 * u12 * ad11l
        + (ad12l - ad11l * u12l) * ad21l;
    let mut x = [
        ((ad22l - ad11l) - ad21l * u12l - (ad11 - ad11l) - (ad22 - ad11l) + ad21 * u12) * ad21l,
        ad32l * ad21l,
    ];
    let istart = ifirst;
    let mut tau = larfg(&mut alpha, &mut x, k);
    let mut v = [one, x[0], x[1]];

    // Sweep (loop 290).
    for j in istart..=(ilast - 2) {
        // All but last elements: use 3×3 Householder transforms.
        // Zero the (j−1)st column of A.
        if j > istart {
            let mut alpha = w.h[(j, j - 1)];
            let mut x = [w.h[(j + 1, j - 1)], w.h[(j + 2, j - 1)]];
            tau = larfg(&mut alpha, &mut x, k);
            w.h[(j, j - 1)] = alpha;
            v = [one, x[0], x[1]];
            w.h[(j + 1, j - 1)] = zero;
            w.h[(j + 2, j - 1)] = zero;
        }
        // Loop 230: rows j..=j+2 of H and T over columns j..=ILASTM.
        reflect3_rows(&mut w.h, j, v, tau, j..n);
        reflect3_rows(&mut w.t, j, v, tau, j..n);
        // Loop 240: columns j..=j+2 of Q.
        reflect3_cols(&mut w.q, j, v, tau, 0..n);

        // Zero the j-th column of B (see DLAGBC for details) and compute
        // the Householder vector.
        let (scale, u1, u2) = zero_column_vector(&w.t, j, k);
        let t1 = Real::sqrt(scale * scale + u1 * u1 + u2 * u2);
        tau = one + scale / t1;
        let vs = -one / (scale + t1);
        v = [one, vs * u1, vs * u2];

        // Apply the transformations from the right.
        // Loop 260: rows IFRSTM..=min(j+3, ILAST) of H.
        reflect3_cols(&mut w.h, j, v, tau, 0..((j + 3).min(ilast) + 1));
        // Loop 270: rows IFRSTM..=j+2 of T.
        reflect3_cols(&mut w.t, j, v, tau, 0..(j + 3));
        // Loop 280: columns j..=j+2 of Z.
        reflect3_cols(&mut w.z, j, v, tau, 0..n);
        w.t[(j + 1, j)] = zero;
        w.t[(j + 2, j)] = zero;
    }

    // Last elements: use Givens rotations. Rotations from the left.
    let j = ilast - 1;
    let temp = w.h[(j, j - 1)];
    let (c, s, r) = lartg(temp, w.h[(j + 1, j - 1)], k);
    w.h[(j, j - 1)] = r;
    w.h[(j + 1, j - 1)] = zero;
    // Loop 300: rows j, j+1 of H and T over columns j..=ILASTM.
    rot_rows(&mut w.h, j, j + 1, j..n, c, s);
    rot_rows(&mut w.t, j, j + 1, j..n, c, s);
    // Loop 310: columns j, j+1 of Q.
    rot_cols(&mut w.q, j, j + 1, 0..n, c, s);

    // Rotations from the right.
    let temp = w.t[(j + 1, j + 1)];
    let (c, s, r) = lartg(temp, w.t[(j + 1, j)], k);
    w.t[(j + 1, j + 1)] = r;
    w.t[(j + 1, j)] = zero;
    // Loop 320: columns j+1, j of H over rows IFRSTM..=ILAST.
    rot_cols(&mut w.h, j + 1, j, 0..(ilast + 1), c, s);
    // Loop 330: columns j+1, j of T over rows IFRSTM..=ILAST−1.
    rot_cols(&mut w.t, j + 1, j, 0..ilast, c, s);
    // Loop 340: columns j+1, j of Z.
    rot_cols(&mut w.z, j + 1, j, 0..n, c, s);
}
