//! Eigenvectors of a complex upper triangular Schur factor (LAPACK `ZTREVC`).
//!
//! Column `k` of the returned matrices belongs to the eigenvalue `T[k, k]`.
//! Right eigenvectors satisfy `T·x = λ·x`, left eigenvectors `yᴴ·T = λ·yᴴ`.
//! Near-singular pivots `T[j, j] − λ` are perturbed to `smin = max(ulp·|λ|,
//! smlnum)` and the working vector is rescaled whenever the solution would
//! otherwise overflow, so defective matrices yield finite eigenvectors with
//! small residuals. Each eigenvector is scaled so that its largest component
//! (by `|re| + |im|`) has magnitude 1.

use num_traits::{Float, One, Zero};
use oxiblas_core::scalar::{ComplexScalar, Field, Real, Scalar};
use oxiblas_matrix::Mat;

use super::qr::{cabs1, cdiv};
use crate::evd::schur::{big_number, small_number};

/// Machine constants shared by the solvers.
struct Constants<R> {
    ulp: R,
    smlnum: R,
    bignum: R,
}

impl<R: Real> Constants<R> {
    fn new(n: usize) -> Self {
        let ulp = <R as Scalar>::epsilon();
        let smlnum = small_number::<R>(n);
        let bignum = big_number(smlnum);
        Self {
            ulp,
            smlnum,
            bignum,
        }
    }
}

/// `|re| + |im|` 1-norms of the strictly upper triangular part of each column.
fn column_norms<T: Field + ComplexScalar>(t: &Mat<T>) -> Vec<T::Real> {
    let n = t.nrows();
    let mut norms = vec![T::Real::zero(); n];
    for (j, norm) in norms.iter_mut().enumerate() {
        for i in 0..j {
            *norm = *norm + cabs1(t[(i, j)]);
        }
    }
    norms
}

#[inline]
fn scale_slice<T: Field + ComplexScalar>(x: &mut [T], s: T::Real) {
    let s = T::from_real(s);
    for v in x.iter_mut() {
        *v = *v * s;
    }
}

/// Solves `d·x = scale·b` for `x`, choosing `scale ≤ 1` so that `x` cannot
/// overflow (LAPACK `DLALN2` for a complex 1×1 system). Returns
/// `(x, scale, |re x| + |im x|)`.
fn solve_1x1<T: Field + ComplexScalar>(d: T, b: T, bignum: T::Real) -> (T, T::Real, T::Real) {
    let one = T::Real::one();
    let dnorm = cabs1(d);
    let bnorm = cabs1(b);
    let mut scale = one;
    if dnorm < one && bnorm > one && bnorm > bignum * dnorm {
        scale = one / bnorm;
    }
    let x = cdiv(b * T::from_real(scale), d);
    (x, scale, cabs1(x))
}

/// Stores `w[range]` scaled to unit largest component into column `col` of `v`.
fn store_normalized<T: Field + ComplexScalar>(
    v: &mut Mat<T>,
    col: usize,
    w: &[T],
    range: core::ops::Range<usize>,
) {
    let mut emax = T::Real::zero();
    for z in &w[range.clone()] {
        emax = Float::max(emax, cabs1(*z));
    }
    let remax = T::from_real(T::Real::one() / emax);
    for r in range {
        v[(r, col)] = w[r] * remax;
    }
}

/// Right eigenvectors of the upper triangular `t` (`T·x_k = T[k, k]·x_k`);
/// entries below row `k` of column `k` are zero.
pub(crate) fn complex_right_eigenvectors<T: Field + ComplexScalar + bytemuck::Zeroable>(
    t: &Mat<T>,
) -> Mat<T> {
    let n = t.nrows();
    let mut v = Mat::zeros(n, n);
    if n == 0 {
        return v;
    }
    let c = Constants::<T::Real>::new(n);
    let cnorm = column_norms(t);
    let mut work = vec![T::zero(); n];
    for ki in (0..n).rev() {
        let lambda = t[(ki, ki)];
        let smin = Float::max(c.ulp * cabs1(lambda), c.smlnum);
        work[ki] = T::one();
        for k in 0..ki {
            work[k] = -t[(k, ki)];
        }
        // Solve (T[0..ki, 0..ki] − λ)·x = scale·rhs by back substitution.
        for j in (0..ki).rev() {
            let mut d = t[(j, j)] - lambda;
            if cabs1(d) < smin {
                d = T::from_real(smin);
            }
            let (mut x, mut scale, xnorm) = solve_1x1(d, work[j], c.bignum);
            // Keep |x|·cnorm[j] below overflow for the update of the rhs.
            if xnorm > T::Real::one() && cnorm[j] > c.bignum / xnorm {
                x = x * T::from_real(T::Real::one() / xnorm);
                scale = scale / xnorm;
            }
            if scale != T::Real::one() {
                scale_slice(&mut work[..=ki], scale);
            }
            work[j] = x;
            for r in 0..j {
                work[r] = work[r] - x * t[(r, j)];
            }
        }
        store_normalized(&mut v, ki, &work, 0..ki + 1);
    }
    v
}

/// Left eigenvectors of the upper triangular `t` (`y_kᴴ·T = T[k, k]·y_kᴴ`);
/// entries above row `k` of column `k` are zero.
pub(crate) fn complex_left_eigenvectors<T: Field + ComplexScalar + bytemuck::Zeroable>(
    t: &Mat<T>,
) -> Mat<T> {
    let n = t.nrows();
    let mut v = Mat::zeros(n, n);
    if n == 0 {
        return v;
    }
    let c = Constants::<T::Real>::new(n);
    let cnorm = column_norms(t);
    let mut work = vec![T::zero(); n];
    for ki in 0..n {
        let lambda = t[(ki, ki)];
        let smin = Float::max(c.ulp * cabs1(lambda), c.smlnum);
        // y[ki] = 1; solve (T[ki+1.., ki+1..] − λ)ᴴ·y = −conj(T[ki, ki+1..]).
        work[ki] = T::one();
        for k in (ki + 1)..n {
            work[k] = -t[(ki, k)].conj();
        }
        let mut vmax = T::Real::one();
        let mut vcrit = c.bignum;
        for j in (ki + 1)..n {
            // Rescale before forming the rhs if it could overflow.
            if cnorm[j] > vcrit {
                scale_slice(&mut work[ki..], T::Real::one() / vmax);
                vmax = T::Real::one();
            }
            let mut sum = T::zero();
            for r in (ki + 1)..j {
                sum = sum + t[(r, j)].conj() * work[r];
            }
            let rhs = work[j] - sum;
            let mut d = t[(j, j)] - lambda;
            if cabs1(d) < smin {
                d = T::from_real(smin);
            }
            let (x, scale, xnorm) = solve_1x1(d.conj(), rhs, c.bignum);
            if scale != T::Real::one() {
                scale_slice(&mut work[ki..], scale);
            }
            work[j] = x;
            vmax = Float::max(vmax, xnorm);
            vcrit = c.bignum / vmax;
        }
        store_normalized(&mut v, ki, &work, ki..n);
    }
    v
}
