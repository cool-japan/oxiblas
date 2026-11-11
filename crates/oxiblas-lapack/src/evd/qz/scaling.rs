//! Norm scaling of the pencil (`DGGES` around `DLANGE('M')`/`DLASCL`).
//!
//! `DGGES` scales `A` (and independently `B`) when its largest entry lies
//! outside `[SMLNUM, BIGNUM]`, runs the QZ pipeline on the scaled pencil and
//! finally undoes the scaling on `S`, `T` and the eigenvalue triples. Before
//! unscaling it rescales complex-pair triples whose unscaled values would
//! over- or underflow.

use num_traits::Float;
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::Mat;

use super::util::Consts;

/// Which entries `DLASCL` touches (its `TYPE` argument).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ScaleKind {
    /// `TYPE = 'G'`: every entry.
    General,
    /// `TYPE = 'U'`: entries on or above the diagonal.
    UpperTriangular,
    /// `TYPE = 'H'`: entries on or above the first subdiagonal.
    UpperHessenberg,
}

/// `DLANGE('M')`: largest absolute entry (NaN-propagating like the reference).
pub(super) fn lange_max<T: Field + Real>(m: &Mat<T>) -> T {
    let mut value = T::zero();
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            let temp = Scalar::abs(m[(i, j)]);
            if value < temp || Float::is_nan(temp) {
                value = temp;
            }
        }
    }
    value
}

/// Multiplier sequence of `DLASCL` for `cto / cfrom` (label 10 loop): each
/// factor is applied to the data in turn, so the product is formed without
/// over- or underflow. An exact final factor of one is omitted (`DLASCL`
/// returns early when `MUL = 1`).
///
/// `cfrom` must be nonzero and not NaN (`DLASCL` `INFO = -4` otherwise).
fn lascl_factors<T: Field + Real>(cfrom: T, cto: T, k: &Consts<T>) -> Vec<T> {
    let smlnum = k.safmin;
    let bignum = T::one() / smlnum;
    let mut cfromc = cfrom;
    let mut ctoc = cto;
    let mut factors = Vec::new();
    loop {
        let cfrom1 = cfromc * smlnum;
        let (mul, done) = if cfrom1 == cfromc {
            // CFROMC is an inf. Multiply by a correctly signed zero for
            // finite CTOC, or a NaN if CTOC is infinite.
            (ctoc / cfromc, true)
        } else {
            let cto1 = ctoc / bignum;
            if cto1 == ctoc {
                // CTOC is either 0 or an inf; it is itself the factor.
                (ctoc, true)
            } else if Scalar::abs(cfrom1) > Scalar::abs(ctoc) && ctoc != T::zero() {
                cfromc = cfrom1;
                (smlnum, false)
            } else if Scalar::abs(cto1) > Scalar::abs(cfromc) {
                ctoc = cto1;
                (bignum, false)
            } else {
                let mul = ctoc / cfromc;
                if mul == T::one() {
                    return factors;
                }
                (mul, true)
            }
        };
        factors.push(mul);
        if done {
            return factors;
        }
    }
}

/// `DLASCL(kind, cfrom, cto)` on a square matrix: multiplies the entries
/// selected by `kind` by `cto / cfrom` without over- or underflow.
pub(super) fn lascl_mat<T: Field + Real>(
    kind: ScaleKind,
    cfrom: T,
    cto: T,
    m: &mut Mat<T>,
    k: &Consts<T>,
) {
    let (nrows, ncols) = (m.nrows(), m.ncols());
    for mul in lascl_factors(cfrom, cto, k) {
        for j in 0..ncols {
            let last = match kind {
                ScaleKind::General => nrows,
                ScaleKind::UpperTriangular => (j + 1).min(nrows),
                ScaleKind::UpperHessenberg => (j + 2).min(nrows),
            };
            for i in 0..last {
                m[(i, j)] = m[(i, j)] * mul;
            }
        }
    }
}

/// `DLASCL('G', cfrom, cto)` on a vector (the `N × 1` calls of `DGGES`).
pub(super) fn lascl_slice<T: Field + Real>(cfrom: T, cto: T, v: &mut [T], k: &Consts<T>) {
    for mul in lascl_factors(cfrom, cto, k) {
        for x in v.iter_mut() {
            *x = *x * mul;
        }
    }
}

/// Rescales the triple `i` by `factor` when that is meaningful.
///
/// Deliberate deviation from `DGGES`: the reference applies
/// `WORK(1) = |A(i,i)/ALPHAR(i)|` unconditionally, which is `inf`/NaN for
/// `ALPHAR(i) = 0` (a purely imaginary pair) and collapses the triple to
/// `(0, 0, 0)` when `A(i,i) = 0`. Such factors are skipped; the eigenvalue
/// ratio is invariant under the rescaling anyway.
fn rescale_triple<T: Field + Real>(
    i: usize,
    factor: T,
    alphar: &mut [T],
    alphai: &mut [T],
    beta: &mut [T],
) {
    if Float::is_finite(factor) && factor > T::zero() {
        beta[i] = beta[i] * factor;
        alphar[i] = alphar[i] * factor;
        alphai[i] = alphai[i] * factor;
    }
}

/// `DGGES` loop 20 (ILASCL): before undoing the scaling of `A`, rescale
/// complex-pair triples whose `ALPHAR`/`ALPHAI` would over- or underflow so
/// that they are on the order of `A(i,i)` / `A(i,i+1)` of the Schur form
/// `s`. Comparisons are signed, exactly as in the reference, except that an
/// exactly zero `ALPHAR(i)` skips the `ALPHAR` test (the reference evaluates
/// `SAFMIN / 0 = +inf` there and poisons the triple), leaving the decision to
/// the `ALPHAI` test.
pub(super) fn guard_unscale_a<T: Field + Real>(
    s: &Mat<T>,
    alphar: &mut [T],
    alphai: &mut [T],
    beta: &mut [T],
    anrm: T,
    anrmto: T,
    k: &Consts<T>,
) {
    let n = alphar.len();
    for i in 0..n {
        if alphai[i] == T::zero() {
            continue;
        }
        let up = anrmto / anrm;
        let down = anrm / anrmto;
        if alphar[i] != T::zero() && (alphar[i] / k.safmax > up || k.safmin / alphar[i] > down) {
            let factor = Scalar::abs(s[(i, i)] / alphar[i]);
            rescale_triple(i, factor, alphar, alphai, beta);
        } else if (alphai[i] / k.safmax > up || k.safmin / alphai[i] > down) && i + 1 < n {
            // Only the first member of a pair (ALPHAI > 0) can pass the
            // signed test, so A(i, i+1) exists; the bound check keeps the
            // access in range regardless.
            let factor = Scalar::abs(s[(i, i + 1)] / alphai[i]);
            rescale_triple(i, factor, alphar, alphai, beta);
        }
    }
}

/// `DGGES` loop 30 (ILBSCL): before undoing the scaling of `B`, rescale
/// complex-pair triples whose `BETA` would over- or underflow so that it is
/// on the order of `B(i,i)` of the triangular factor `t`.
pub(super) fn guard_unscale_b<T: Field + Real>(
    t: &Mat<T>,
    alphar: &mut [T],
    alphai: &mut [T],
    beta: &mut [T],
    bnrm: T,
    bnrmto: T,
    k: &Consts<T>,
) {
    for i in 0..alphar.len() {
        if alphai[i] == T::zero() {
            continue;
        }
        if beta[i] / k.safmax > bnrmto / bnrm || k.safmin / beta[i] > bnrm / bnrmto {
            let factor = Scalar::abs(t[(i, i)] / beta[i]);
            rescale_triple(i, factor, alphar, alphai, beta);
        }
    }
}
