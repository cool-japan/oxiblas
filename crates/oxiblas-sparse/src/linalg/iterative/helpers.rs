//! Helper functions shared across iterative solvers.
//!
//! # Scale-invariant breakdown tests
//!
//! Krylov methods are invariant under the rescaling `(A, b) -> (s A, s b)`: the
//! iterates do not change while every residual-like quantity scales by a power
//! of `s`. A breakdown (or "lucky breakdown") test must therefore never compare
//! a scaled quantity against a bare `epsilon()`; doing so declares spurious
//! breakdowns on small problems and misses genuine ones on large problems. The
//! tests here take one of two relative forms:
//!
//! * an inner product against the norms of its factors (Cauchy–Schwarz form),
//!   `|u·v| <= eps·‖u‖·‖v‖` — see [`negligible_inner`]; where the inner product
//!   must keep one sign throughout the iteration (`rᵀM⁻¹r` and `pᵀAp` in
//!   preconditioned CG), a negligible value (NaN included) or a value whose
//!   sign differs from its first one — see [`sign_breakdown`];
//! * a norm (or magnitude) against the norm it was derived from,
//!   `‖w‖ <= eps·‖w₀‖` — see [`negligible_norm`].
//!
//! When a breakdown is detected, the solver consults the true residual
//! `‖b − A x‖` ([`converged_true_residual`]) before deciding between reporting
//! convergence and reporting the breakdown.
//!
//! # Overflow-safe norms
//!
//! Every norm that enters a convergence or breakdown decision (residual and
//! vector norms, column norms of small dense matrices, Givens and Lanczos
//! magnitudes) is computed with the scaled 2-norm of
//! `linalg::precond::tolerance`: bit for bit the plain `sqrt(Σ v_i²)` whenever
//! that neither overflows nor underflows, and accurate when it would (a
//! right-hand side of size `1e160` used to give `‖b‖ = ∞`, so `tol·‖b‖ = ∞`
//! and every iteration "converged" at once). The norm of a vector scaled by a
//! power of two is exactly scaled, so the decisions stay scale-invariant.

use super::scaling::Operator;
use super::types::IterativeError;
use crate::linalg::precond::tolerance::{norm2, sum_of_squares_split};
use oxiblas_core::scalar::{Field, Real, Scalar};

pub(super) use crate::linalg::precond::tolerance::{
    SumOfSquares, dot_and_sums_of_squares, sum_of_squares,
};

/// Returns `true` when the inner product `inner = u·v` is negligible relative
/// to the norms of its factors: `|u·v| <= eps·‖u‖·‖v‖`.
///
/// The test is invariant under any rescaling of `u` and `v`. A zero factor
/// (`‖u‖ = 0` or `‖v‖ = 0`) always yields `true`.
pub(super) fn negligible_inner<T: Scalar<Real = T> + Clone + Field + Real>(
    inner: T,
    u_norm: T,
    v_norm: T,
) -> bool {
    Scalar::abs(inner) <= <T as Scalar>::epsilon() * u_norm * v_norm
}

/// The breakdown rule of preconditioned CG for an inner product that must
/// keep one sign: `rᵀz` (`z = M⁻¹ r`) and `pᵀAp`.
///
/// CG divides by both (`α = rᵀz / pᵀAp`, `β = r_{k+1}ᵀz_{k+1} / r_kᵀz_k`) and
/// converges as long as each keeps its sign: for SPD `A` and `M` both are
/// positive, and when `A` or `M` (or both) are negative definite the signs
/// flip consistently and the iterates are those of the corresponding SPD
/// problem. Returns `true` (breakdown) when
///
/// * `|inner| <= eps·‖u‖·‖v‖` — negligible in the Cauchy–Schwarz sense (a
///   singular `M` or `A` along the current vector), or NaN (non-finite
///   preconditioner output); or
/// * the sign of `inner` differs from `reference`, the sign at its first
///   check, which reveals an indefinite preconditioner or operator.
///
/// The first call records the sign in `reference`. Both tests are invariant
/// under any rescaling of `u` and `v` (by positive factors).
pub(super) fn sign_breakdown<T: Scalar<Real = T> + Clone + Field + Real>(
    inner: T,
    u_norm: T,
    v_norm: T,
    reference: &mut Option<bool>,
) -> bool {
    if !(Scalar::abs(inner) > <T as Scalar>::epsilon() * u_norm * v_norm) {
        return true;
    }
    let positive = inner > T::zero();
    match *reference {
        None => {
            *reference = Some(positive);
            false
        }
        Some(sign) => sign != positive,
    }
}

/// Returns `true` when `value` (a norm or magnitude) is negligible relative to
/// the norm `reference` it was derived from: `|value| <= eps·reference`.
///
/// The test is invariant under a common rescaling of both arguments. A zero
/// reference always yields `true` for a zero value.
pub(super) fn negligible_norm<T: Scalar<Real = T> + Clone + Field + Real>(
    value: T,
    reference: T,
) -> bool {
    Scalar::abs(value) <= <T as Scalar>::epsilon() * reference
}

/// Computes the true residual norm `‖b − A x‖` (overflow-safe).
pub(super) fn true_residual_norm<T: Scalar<Real = T> + Clone + Field + Real>(
    a: &Operator<'_, T>,
    b: &[T],
    x: &[T],
) -> T {
    a.true_residual_norm(b, x)
}

/// Breakdown verdict from the true residual.
///
/// Returns `Some(‖b − A x‖)` when the true residual meets `tol_abs` (the
/// breakdown is harmless: the iterate has already converged), and `None` when
/// the breakdown is genuine and must be reported.
pub(super) fn converged_true_residual<T: Scalar<Real = T> + Clone + Field + Real>(
    a: &Operator<'_, T>,
    b: &[T],
    x: &[T],
    tol_abs: T,
) -> Option<T> {
    let residual = true_residual_norm(a, b, x);
    if residual <= tol_abs {
        Some(residual)
    } else {
        None
    }
}

/// Builds an [`IterativeError::Breakdown`].
pub(super) fn breakdown(iteration: usize, description: &str) -> IterativeError {
    IterativeError::Breakdown {
        iteration,
        description: description.to_string(),
    }
}

/// Computes a Givens rotation `(c, s, r)` with `c·a + s·b = r` and
/// `−s·a + c·b = 0`.
///
/// Only exact zeros take the shortcuts: a test such as `|b| <= eps` would be an
/// absolute threshold on entries that scale with `A`. `r = ‖(a, b)‖` is the
/// overflow-safe norm (bit for bit `sqrt(a² + b²)` when that is
/// representable), and the formula is homogeneous, so the rotation is exactly
/// scale-invariant.
pub(super) fn givens_rotation_gmres<T: Scalar<Real = T> + Clone + Field + Real>(
    a: T,
    b: T,
) -> (T, T, T) {
    if b == T::zero() {
        return (T::one(), T::zero(), a);
    }

    if a == T::zero() {
        let sign = if b >= T::zero() {
            T::one()
        } else {
            T::zero() - T::one()
        };
        return (T::zero(), sign, Scalar::abs(b));
    }

    let r = norm2([a, b].into_iter());
    let c = a / r;
    let s = b / r;

    (c, s, r)
}

/// Solves `R y = g` for the leading `m x m` upper-triangular part of the
/// rotated Hessenberg matrix `h` (stored column by column: `h[col][row]`).
///
/// A diagonal entry is treated as zero (and `y_i` left at zero) only when it is
/// negligible relative to the norm of its own column, which equals the norm of
/// `A v_i` because Givens rotations preserve column norms.
pub(super) fn solve_upper_triangular<T: Scalar<Real = T> + Clone + Field + Real>(
    h: &[Vec<T>],
    g: &[T],
    m: usize,
) -> Vec<T> {
    let mut y = vec![T::zero(); m];

    for i in (0..m).rev() {
        let mut sum = g[i].clone();
        for j in (i + 1)..m {
            sum = sum - h[j][i].clone() * y[j].clone();
        }
        let column_norm = norm(&h[i][..=i]);
        if !negligible_norm(h[i][i].clone(), column_norm) {
            y[i] = sum / h[i][i].clone();
        }
    }

    y
}

pub(super) fn dot<T: Scalar + Clone + Field>(a: &[T], b: &[T]) -> T {
    assert_eq!(a.len(), b.len());
    let mut sum = T::zero();
    for i in 0..a.len() {
        sum = sum + a[i].clone() * b[i].clone();
    }
    sum
}

/// Overflow-safe 2-norm of `v` (bit for bit `sqrt(dot(v, v))` whenever that
/// is representable).
pub(super) fn norm<T: Scalar<Real = T> + Clone + Field + Real>(v: &[T]) -> T {
    norm2(v.iter().copied())
}

/// Forward substitution `M x = b` for the `s x s` matrix `m` (row-major,
/// `m[row][col]`; only the lower triangle is read).
///
/// A diagonal entry negligible relative to the norm of its column marks a
/// (numerically) unavailable direction; its component is set to zero rather
/// than left undivided, which would mix quantities of different scales.
pub(super) fn solve_lower_triangular_s<T: Scalar<Real = T> + Clone + Field + Real>(
    m: &[Vec<T>],
    b: &[T],
) -> Vec<T> {
    let s = b.len();
    let mut x = vec![T::zero(); s];

    for i in 0..s {
        let mut sum = b[i].clone();
        for j in 0..i {
            sum = sum - m[i][j].clone() * x[j].clone();
        }
        let column_norm = norm2(m.iter().take(s).map(|row| row[i]));
        if !negligible_norm(m[i][i].clone(), column_norm) {
            x[i] = sum / m[i][i].clone();
        }
    }

    x
}

/// Modified Gram–Schmidt QR of the `p` vectors in `vecs` (each of length `n`).
///
/// A column whose orthogonalized norm is negligible relative to its original
/// norm is numerically dependent on the previous ones; it is replaced by zero so
/// that rounding noise is never promoted to a basis direction.
pub(super) fn block_qr_gmres<T>(vecs: &[Vec<T>], n: usize, p: usize) -> (Vec<Vec<T>>, Vec<Vec<T>>)
where
    T: Scalar<Real = T> + Clone + Field + Real,
{
    let mut q: Vec<Vec<T>> = vecs.to_vec();
    let mut r: Vec<Vec<T>> = (0..p).map(|_| vec![T::zero(); p]).collect();

    for j in 0..p {
        let original_norm = norm(&q[j]);
        for i in 0..j {
            let dot_val = dot(&q[j], &q[i]);
            r[i][j] = dot_val.clone();
            for k in 0..n {
                q[j][k] = q[j][k].clone() - dot_val.clone() * q[i][k].clone();
            }
        }

        let nrm = norm(&q[j]);
        r[j][j] = nrm.clone();

        if negligible_norm(nrm.clone(), original_norm) {
            for k in 0..n {
                q[j][k] = T::zero();
            }
        } else {
            for k in 0..n {
                q[j][k] = q[j][k].clone() / nrm.clone();
            }
        }
    }

    (q, r)
}

pub(super) fn block_inner_prod_gmres<T>(v: &[Vec<T>], w: &[Vec<T>], p: usize) -> Vec<Vec<T>>
where
    T: Scalar<Real = T> + Clone + Field + Real,
{
    let mut result: Vec<Vec<T>> = (0..p).map(|_| vec![T::zero(); p]).collect();
    for i in 0..p {
        for j in 0..p {
            result[i][j] = dot(&v[i], &w[j]);
        }
    }
    result
}

/// Solves the block least-squares problem `min ‖H Y − RHS‖_F` by Householder
/// QR (`h` is `m x n_cols`, row-major; `rhs` is `m x p`).
///
/// Rank deficiency is judged relative to column norms (which Householder
/// reflections preserve), never against an absolute threshold.
pub(super) fn solve_block_ls_gmres<T>(
    h: &[Vec<T>],
    rhs: &[Vec<T>],
    m: usize,
    n_cols: usize,
    p: usize,
) -> Vec<Vec<T>>
where
    T: Scalar<Real = T> + Clone + Field + Real,
{
    if n_cols == 0 {
        return vec![];
    }

    let mut a: Vec<Vec<T>> = h.to_vec();
    let mut b_mat: Vec<Vec<T>> = rhs.to_vec();

    // Householder QR
    for j in 0..n_cols.min(m) {
        // Overflow-safe sums of squares of column j above and below the
        // diagonal, with a common scale so that they can be added.
        let (above, below) = sum_of_squares_split(
            a.iter().take(j).map(|row| row[j]),
            a.iter().take(m).skip(j).map(|row| row[j]),
        );
        let col_norm = below.norm();
        let full_norm = above.plus(&below).norm();

        // The part of column j below the diagonal is negligible: the column is
        // (numerically) in the span of the previous ones.
        if negligible_norm(col_norm.clone(), full_norm) {
            continue;
        }

        let sign = if a[j][j] >= T::zero() {
            T::one()
        } else {
            T::zero() - T::one()
        };
        let v0 = a[j][j].clone() + sign * col_norm.clone();

        let mut v_h: Vec<T> = vec![T::zero(); m - j];
        v_h[0] = T::one();
        // |v0| >= col_norm > 0 here, so the division is always safe.
        if v0 != T::zero() {
            for i in 1..(m - j) {
                v_h[i] = a[j + i][j].clone() / v0.clone();
            }
        }

        // v_h[0] = 1 and v_h is dimensionless, so vtv >= 1: this guard is
        // scale-invariant.
        let mut vtv = T::zero();
        for vi in &v_h {
            vtv = vtv + vi.clone() * vi.clone();
        }
        let tau = if Scalar::abs(vtv.clone()) > <T as Scalar>::epsilon() {
            (T::one() + T::one()) / vtv
        } else {
            T::zero()
        };

        for k in j..n_cols {
            let mut dot_v = T::zero();
            for i in 0..(m - j) {
                dot_v = dot_v + v_h[i].clone() * a[j + i][k].clone();
            }
            for i in 0..(m - j) {
                a[j + i][k] = a[j + i][k].clone() - tau.clone() * v_h[i].clone() * dot_v.clone();
            }
        }

        for k in 0..p {
            let mut dot_v = T::zero();
            for i in 0..(m - j) {
                dot_v = dot_v + v_h[i].clone() * b_mat[j + i][k].clone();
            }
            for i in 0..(m - j) {
                b_mat[j + i][k] =
                    b_mat[j + i][k].clone() - tau.clone() * v_h[i].clone() * dot_v.clone();
            }
        }
    }

    // Back substitution
    let mut y: Vec<Vec<T>> = vec![vec![T::zero(); p]; n_cols];

    for j in (0..n_cols).rev() {
        let column_norm = norm2(a.iter().take(m).map(|row| row[j]));
        let usable = !negligible_norm(a[j][j].clone(), column_norm);
        for k in 0..p {
            let mut sum = b_mat[j][k].clone();
            for i in (j + 1)..n_cols {
                sum = sum - a[j][i].clone() * y[i][k].clone();
            }
            if usable {
                y[j][k] = sum / a[j][j].clone();
            } else {
                y[j][k] = T::zero();
            }
        }
    }

    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_breakdown_rule() {
        let eps = f64::EPSILON;
        // First call records the sign; a negligible value is a breakdown.
        let mut sign = None;
        assert!(sign_breakdown(eps * 0.5, 1.0, 1.0, &mut sign));
        assert_eq!(sign, None);
        assert!(!sign_breakdown(-2.0, 1.0, 3.0, &mut sign));
        assert_eq!(sign, Some(false));
        // Consistently negative values keep going; a sign change breaks down.
        assert!(!sign_breakdown(-1e-3, 1.0, 1.0, &mut sign));
        assert!(sign_breakdown(1e-3, 1.0, 1.0, &mut sign));
        // NaN is always a breakdown.
        let mut fresh = None;
        assert!(sign_breakdown(f64::NAN, 1.0, 1.0, &mut fresh));
        let mut positive = Some(true);
        assert!(sign_breakdown(f64::NAN, 1.0, 1.0, &mut positive));
    }

    #[test]
    fn test_givens_rotation_safe_magnitude() {
        let (c, s, r) = givens_rotation_gmres(3.0_f64, 4.0);
        assert_eq!((c, s, r), (0.6, 0.8, 5.0));
        let big = 2.0_f64.powi(600);
        let (c, s, r) = givens_rotation_gmres(3.0 * big, 4.0 * big);
        assert_eq!((c, s, r), (0.6, 0.8, 5.0 * big));
    }
}
