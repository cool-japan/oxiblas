//! Shared numerical kernels of the QZ pipeline.
//!
//! Idiomatic translations of the LAPACK auxiliaries used by `DGGES` and
//! `DHGEQZ` (Netlib LAPACK 3.12, BSD-3): the `DLAMCH` machine constants,
//! `DLARTG` (LAPACK 3.10 safe-scaling semantics), `DLARFG`, `DLAPY2`,
//! `DLAPY3`, a scaled 2-norm (`DNRM2`), the Frobenius norm of an upper
//! Hessenberg block (`DLANHS('F')`) and the plane rotations `DROT` on rows or
//! columns over explicit index ranges.
//!
//! All constants are derived from the scalar traits (no float literals), so
//! the kernels are valid for `f16`/`QuadFloat` as well as `f32`/`f64`.

use core::ops::Range;

use num_traits::Float;
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::Mat;

/// Machine constants (LAPACK `DLAMCH`) and small exact constants.
#[derive(Debug, Clone, Copy)]
pub(super) struct Consts<T> {
    /// Relative machine precision `DLAMCH('P') = eps · base` (the `DHGEQZ`
    /// `ULP` and the `DGGES` `EPS`); equals `<T as Scalar>::epsilon()`.
    pub(super) eps: T,
    /// Unit roundoff `DLAMCH('E') = eps / 2` (used by `DLASV2` and `DLARFG`).
    pub(super) unit_roundoff: T,
    /// Safe minimum `DLAMCH('S')`: the smallest positive normal number.
    pub(super) safmin: T,
    /// `1 / safmin`.
    pub(super) safmax: T,
    /// `DGGES` `SMLNUM = sqrt(safmin) / eps`.
    pub(super) smlnum: T,
    /// `DGGES` `BIGNUM = 1 / smlnum`.
    pub(super) bignum: T,
    /// `DLARTG` `RTMIN = sqrt(safmin)`.
    pub(super) rtmin: T,
    /// `DLARTG` `RTMAX = sqrt(safmax / 2)`.
    pub(super) rtmax: T,
    /// `1/2`.
    pub(super) half: T,
    /// `2`.
    pub(super) two: T,
    /// `4`.
    pub(super) four: T,
    /// `DHGEQZ` `SAFETY = 100` (safety factor of the `DLAG2` safe minimum).
    pub(super) safety: T,
    /// `DLAG2` `FUZZY1 = 1 + 1e-5`.
    pub(super) fuzzy1: T,
}

impl<T: Field + Real> Consts<T> {
    /// Builds the constants for `T`.
    pub(super) fn new() -> Self {
        let one = T::one();
        let two = one + one;
        let half = one / two;
        let four = two + two;
        let ten = four + four + two;
        let eps = <T as Scalar>::epsilon();
        let safmin = <T as Scalar>::min_positive();
        let safmax = one / safmin;
        let rtmin = Real::sqrt(safmin);
        let smlnum = rtmin / eps;
        Self {
            eps,
            unit_roundoff: eps * half,
            safmin,
            safmax,
            smlnum,
            bignum: one / smlnum,
            rtmin,
            rtmax: Real::sqrt(safmax / two),
            half,
            two,
            four,
            safety: T::from_u32(100).unwrap_or(ten * ten),
            fuzzy1: one
                + T::from_f64(1.0e-5).unwrap_or_else(|| one / (ten * ten * ten * ten * ten)),
        }
    }
}

/// Fortran `SIGN(A, B)`: `|a|` carrying the sign bit of `b`.
///
/// Uses IEEE signed-zero semantics (`SIGN(x, -0.0) = -|x|`, as gfortran and
/// Fortran 95+ processors with signed zeros do).
#[inline]
pub(super) fn sign<T: Real>(a: T, b: T) -> T {
    let magnitude = Scalar::abs(a);
    if Float::is_sign_negative(b) {
        -magnitude
    } else {
        magnitude
    }
}

/// `DLARTG` (LAPACK 3.10, Anderson's safe scaling): plane rotation with
/// `[c s; −s c] · [f; g] = [r; 0]`.
///
/// Returns `(c, s, r)` with `c ≥ 0`, `c² + s² = 1` and
/// `r = sign(f) · hypot(f, g)` (`r = f` when `g = 0`; `r = |g|` when
/// `f = 0`).
pub(super) fn lartg<T: Field + Real>(f: T, g: T, k: &Consts<T>) -> (T, T, T) {
    let f1 = Scalar::abs(f);
    let g1 = Scalar::abs(g);
    if g == T::zero() {
        (T::one(), T::zero(), f)
    } else if f == T::zero() {
        (T::zero(), sign(T::one(), g), g1)
    } else if f1 > k.rtmin && f1 < k.rtmax && g1 > k.rtmin && g1 < k.rtmax {
        let d = Real::sqrt(f * f + g * g);
        let r = sign(d, f);
        (f1 / d, g / r, r)
    } else {
        let u = Float::min(k.safmax, Float::max(k.safmin, Float::max(f1, g1)));
        let fs = f / u;
        let gs = g / u;
        let d = Real::sqrt(fs * fs + gs * gs);
        let r = sign(d, f);
        (Scalar::abs(fs) / d, gs / r, r * u)
    }
}

/// `DLAPY2`: `sqrt(x² + y²)` without unnecessary overflow (NaN-propagating).
pub(super) fn lapy2<T: Field + Real>(x: T, y: T) -> T {
    // DLAPY2 returns the NaN argument (Y wins when both are NaN).
    if Float::is_nan(y) {
        return y;
    }
    if Float::is_nan(x) {
        return x;
    }
    let xabs = Scalar::abs(x);
    let yabs = Scalar::abs(y);
    let w = Float::max(xabs, yabs);
    let z = Float::min(xabs, yabs);
    if z == T::zero() || w > <T as Scalar>::max_value() {
        w
    } else {
        let q = z / w;
        w * Real::sqrt(T::one() + q * q)
    }
}

/// `DLAPY3`: `sqrt(x² + y² + z²)` without unnecessary overflow.
pub(super) fn lapy3<T: Field + Real>(x: T, y: T, z: T) -> T {
    let xabs = Scalar::abs(x);
    let yabs = Scalar::abs(y);
    let zabs = Scalar::abs(z);
    let w = Float::max(xabs, Float::max(yabs, zabs));
    if w == T::zero() || w > <T as Scalar>::max_value() {
        // W can be zero for max(0, NaN, 0); the sum keeps the NaN.
        xabs + yabs + zabs
    } else {
        let (p, q, r) = (xabs / w, yabs / w, zabs / w);
        w * Real::sqrt(p * p + q * q + r * r)
    }
}

/// Scaled sum of squares (`DLASSQ` update rule): accumulates
/// `scale² · sumsq = Σ xᵢ²` without overflow or harmful underflow.
#[derive(Debug, Clone, Copy)]
pub(super) struct SumSq<T> {
    scale: T,
    sumsq: T,
}

impl<T: Field + Real> SumSq<T> {
    /// Empty accumulator (`scale = 0`, `sumsq = 1`).
    pub(super) fn new() -> Self {
        Self {
            scale: T::zero(),
            sumsq: T::one(),
        }
    }

    /// Adds `x²`.
    pub(super) fn push(&mut self, x: T) {
        if x != T::zero() {
            let absx = Scalar::abs(x);
            if self.scale < absx {
                let q = self.scale / absx;
                self.sumsq = T::one() + self.sumsq * q * q;
                self.scale = absx;
            } else {
                let q = absx / self.scale;
                self.sumsq = self.sumsq + q * q;
            }
        }
    }

    /// `sqrt(Σ xᵢ²)`.
    pub(super) fn norm(&self) -> T {
        self.scale * Real::sqrt(self.sumsq)
    }
}

/// Scaled Euclidean norm of a vector (reference `DNRM2` recurrence).
pub(super) fn nrm2<T: Field + Real>(x: &[T]) -> T {
    let mut acc = SumSq::new();
    for &xi in x {
        acc.push(xi);
    }
    acc.norm()
}

/// `DLANHS('F')` of the upper Hessenberg block `m(lo..=hi, lo..=hi)`:
/// Frobenius norm over the entries on or above the first subdiagonal.
pub(super) fn lanhs_fro<T: Field + Real>(m: &Mat<T>, lo: usize, hi: usize) -> T {
    let mut acc = SumSq::new();
    for j in lo..=hi {
        for i in lo..=hi.min(j + 1) {
            acc.push(m[(i, j)]);
        }
    }
    acc.norm()
}

/// `DLARFG`: generates an elementary reflector `H = I − tau·u·uᵀ` with
/// `u = [1; v]` such that `H · [alpha; x] = [beta; 0]`.
///
/// On return `alpha` holds `beta` and `x` holds `v`; returns `tau`
/// (`tau = 0` and `H = I` when `x` is zero or empty).
pub(super) fn larfg<T: Field + Real>(alpha: &mut T, x: &mut [T], k: &Consts<T>) -> T {
    if x.is_empty() {
        return T::zero();
    }
    let mut xnorm = nrm2(x);
    if xnorm == T::zero() {
        return T::zero();
    }
    let mut beta = -sign(lapy2(*alpha, xnorm), *alpha);
    // DLARFG uses SAFMIN = DLAMCH('S') / DLAMCH('E').
    let safmin = k.safmin / k.unit_roundoff;
    let mut knt = 0usize;
    if Scalar::abs(beta) < safmin {
        // XNORM, BETA may be inaccurate; scale X and recompute them (label 10).
        let rsafmn = T::one() / safmin;
        loop {
            knt += 1;
            for xi in x.iter_mut() {
                *xi = *xi * rsafmn;
            }
            beta = beta * rsafmn;
            *alpha = *alpha * rsafmn;
            if !(Scalar::abs(beta) < safmin && knt < 20) {
                break;
            }
        }
        // New BETA is at most 1, at least SAFMIN.
        xnorm = nrm2(x);
        beta = -sign(lapy2(*alpha, xnorm), *alpha);
    }
    let tau = (beta - *alpha) / beta;
    let scal = T::one() / (*alpha - beta);
    for xi in x.iter_mut() {
        *xi = *xi * scal;
    }
    // If ALPHA is subnormal, it may lose relative accuracy (loop 20).
    for _ in 0..knt {
        beta = beta * safmin;
    }
    *alpha = beta;
    tau
}

/// `DROT` on two rows: for every column `j` in `cols`,
/// `(m[r1,j], m[r2,j]) ← (c·m[r1,j] + s·m[r2,j], c·m[r2,j] − s·m[r1,j])`.
pub(super) fn rot_rows<T: Field + Real>(
    m: &mut Mat<T>,
    r1: usize,
    r2: usize,
    cols: Range<usize>,
    c: T,
    s: T,
) {
    for j in cols {
        let x = m[(r1, j)];
        let y = m[(r2, j)];
        m[(r1, j)] = c * x + s * y;
        m[(r2, j)] = c * y - s * x;
    }
}

/// `DROT` on two columns: for every row `i` in `rows`,
/// `(m[i,c1], m[i,c2]) ← (c·m[i,c1] + s·m[i,c2], c·m[i,c2] − s·m[i,c1])`.
pub(super) fn rot_cols<T: Field + Real>(
    m: &mut Mat<T>,
    c1: usize,
    c2: usize,
    rows: Range<usize>,
    c: T,
    s: T,
) {
    for i in rows {
        let x = m[(i, c1)];
        let y = m[(i, c2)];
        m[(i, c1)] = c * x + s * y;
        m[(i, c2)] = c * y - s * x;
    }
}
