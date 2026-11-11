//! Conjugate Gradient (CG) and Preconditioned CG solvers.

use super::helpers::{
    breakdown, converged_true_residual, dot_and_sums_of_squares, negligible_inner, sign_breakdown,
    sum_of_squares,
};
use super::scaling::{Operator, PrecondScaling, SystemScaling};
use super::types::{CgResult, IterativeError};
use crate::csr::CsrMatrix;
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Conjugate Gradient for symmetric positive definite `A`.
///
/// Converges when the recursively updated residual satisfies
/// `‖r‖ <= tol·‖b‖` (overflow-safe norms; see the [module
/// documentation](super) for systems far from unit scale).
///
/// # Breakdown
///
/// A step breaks down when `|pᵀAp| <= eps·‖p‖·‖Ap‖` (scale-invariant
/// Cauchy–Schwarz form). The true residual `‖b − A x‖` then decides: if it
/// meets `tol·‖b‖` the current iterate is returned as converged, otherwise
/// [`IterativeError::Breakdown`] is returned.
pub fn cg<T: Scalar<Real = T> + Clone + Field + Real>(
    a: &CsrMatrix<T>,
    b: &[T],
    x0: &[T],
    tol: T,
    max_iter: usize,
) -> Result<CgResult<T>, IterativeError> {
    let scaling = SystemScaling::for_rhs(a, b);
    let mut result = cg_scaled(
        &scaling.operator(a),
        &scaling.rhs(b),
        &scaling.initial_guess(x0),
        tol,
        max_iter,
    )?;
    scaling.solution(&mut result.x);
    result.residual_norm = scaling.residual(result.residual_norm);
    Ok(result)
}

/// [`cg`] on the (rescaled) system `a x = b`.
fn cg_scaled<T: Scalar<Real = T> + Clone + Field + Real>(
    a: &Operator<'_, T>,
    b: &[T],
    x0: &[T],
    tol: T,
    max_iter: usize,
) -> Result<CgResult<T>, IterativeError> {
    let n = a.nrows();

    if b.len() != n || x0.len() != n {
        return Err(IterativeError::DimensionMismatch {
            expected: n,
            actual: b.len().min(x0.len()),
        });
    }

    let mut x = x0.to_vec();
    let mut r = vec![T::zero(); n];
    let mut p = vec![T::zero(); n];
    let mut ap = vec![T::zero(); n];

    // r = b - A*x
    a.residual(b, &x, &mut r);

    // p = r
    p.clone_from_slice(&r);

    // r^T r (overflow-safe; bit-identical to the plain sum of squares when
    // that is representable) and ||r||
    let mut r_squares = sum_of_squares(r.iter().copied());
    let mut rr = r_squares.value();

    let b_norm = sum_of_squares(b.iter().copied()).norm();
    let tol_abs = tol.clone() * b_norm.clone();

    for iter in 0..max_iter {
        let r_norm = r_squares.norm();

        if r_norm <= tol_abs {
            return Ok(CgResult {
                x,
                iterations: iter,
                residual_norm: r_norm,
                converged: true,
            });
        }

        // ap = A * p
        a.apply(&p, &mut ap);

        // alpha = rr / (p^T * ap)
        let (pap, pp, apap) = dot_and_sums_of_squares(&p, &ap);

        if negligible_inner(pap.clone(), pp.norm(), apap.norm()) {
            if let Some(residual_norm) = converged_true_residual(a, b, &x, tol_abs.clone()) {
                return Ok(CgResult {
                    x,
                    iterations: iter,
                    residual_norm,
                    converged: true,
                });
            }
            return Err(breakdown(
                iter,
                "p^T * A * p negligible relative to ||p|| * ||A p||",
            ));
        }

        let alpha = rr.clone() / pap;

        // x = x + alpha * p
        for i in 0..n {
            x[i] = x[i].clone() + alpha.clone() * p[i].clone();
        }

        // r = r - alpha * ap
        for i in 0..n {
            r[i] = r[i].clone() - alpha.clone() * ap[i].clone();
        }

        // rr_new = r^T * r
        r_squares = sum_of_squares(r.iter().copied());
        let rr_new = r_squares.value();

        // beta = rr_new / rr
        let beta = rr_new.clone() / rr.clone();

        // p = r + beta * p
        for i in 0..n {
            p[i] = r[i].clone() + beta.clone() * p[i].clone();
        }

        rr = rr_new;
    }

    Ok(CgResult {
        x,
        iterations: max_iter,
        residual_norm: r_squares.norm(),
        converged: false,
    })
}

/// Preconditioned Conjugate Gradient; `precond` applies `M^{-1}`.
///
/// Converges when the recursively updated residual satisfies
/// `‖r‖ <= tol·‖b‖` (overflow-safe norms; see the [module
/// documentation](super) for systems far from unit scale, for which
/// `precond` receives rescaled residuals and must be linear).
///
/// # Breakdown
///
/// PCG divides by `rᵀz` (`z = M⁻¹ r`) and by `pᵀAp`, and converges as long as
/// each keeps its sign: positive for SPD `A` and `M`, and consistently
/// negative when `M`, `A` or both are negative definite (the iterates are then
/// exactly those of the corresponding SPD problem). Each of the two inner
/// products is therefore checked with a scale-invariant rule (`rᵀz` at the
/// start of every iteration that has not converged, `pᵀAp` after the product
/// with `A`), and either triggers a true-residual check, which yields a
/// converged result or [`IterativeError::Breakdown`]:
///
/// * the value is negligible, `|rᵀz| <= eps·‖r‖·‖z‖` (`|pᵀAp| <=
///   eps·‖p‖·‖Ap‖`), or NaN (non-finite preconditioner output): `M` or `A` is
///   singular along the current vector;
/// * its sign differs from its sign at the first iteration: `M` (`A`) is
///   indefinite.
pub fn pcg<T, F>(
    a: &CsrMatrix<T>,
    b: &[T],
    x0: &[T],
    precond: F,
    tol: T,
    max_iter: usize,
) -> Result<CgResult<T>, IterativeError>
where
    T: Scalar<Real = T> + Clone + Field + Real,
    F: Fn(&[T]) -> Vec<T>,
{
    let scaling = SystemScaling::for_rhs(a, b);
    let precond_scaling = PrecondScaling::new();
    let mut result = pcg_scaled(
        &scaling.operator(a),
        &scaling.rhs(b),
        &scaling.initial_guess(x0),
        |r: &[T]| precond_scaling.finish(r, precond(r)),
        tol,
        max_iter,
    )?;
    scaling.solution(&mut result.x);
    result.residual_norm = scaling.residual(result.residual_norm);
    Ok(result)
}

/// [`pcg`] on the (rescaled) system `a x = b`.
fn pcg_scaled<T, F>(
    a: &Operator<'_, T>,
    b: &[T],
    x0: &[T],
    precond: F,
    tol: T,
    max_iter: usize,
) -> Result<CgResult<T>, IterativeError>
where
    T: Scalar<Real = T> + Clone + Field + Real,
    F: Fn(&[T]) -> Vec<T>,
{
    let n = a.nrows();

    if b.len() != n || x0.len() != n {
        return Err(IterativeError::DimensionMismatch {
            expected: n,
            actual: b.len().min(x0.len()),
        });
    }

    let mut x = x0.to_vec();
    let mut r = vec![T::zero(); n];
    let mut ap = vec![T::zero(); n];

    // r = b - A*x
    a.residual(b, &x, &mut r);

    // z = M^{-1} * r
    let mut z = precond(&r);

    // p = z
    let mut p = z.clone();

    // rz = r^T z, with ||r|| and ||z|| for the r^T z breakdown test
    let (mut rz, mut r_squares, mut z_squares) = dot_and_sums_of_squares(&r, &z);

    let b_norm = sum_of_squares(b.iter().copied()).norm();
    let tol_abs = tol.clone() * b_norm.clone();

    // Signs of r^T z and p^T A p at their first checks.
    let mut rz_sign: Option<bool> = None;
    let mut pap_sign: Option<bool> = None;

    for iter in 0..max_iter {
        let r_norm = r_squares.norm();

        if r_norm <= tol_abs {
            return Ok(CgResult {
                x,
                iterations: iter,
                residual_norm: r_norm,
                converged: true,
            });
        }

        // r^T z = r^T M^{-1} r is the divisor of the next beta (and the
        // numerator of alpha): it must not be negligible and must keep its
        // sign.
        if sign_breakdown(rz.clone(), r_norm, z_squares.norm(), &mut rz_sign) {
            if let Some(residual_norm) = converged_true_residual(a, b, &x, tol_abs.clone()) {
                return Ok(CgResult {
                    x,
                    iterations: iter,
                    residual_norm,
                    converged: true,
                });
            }
            return Err(breakdown(
                iter,
                "r^T * M^{-1} * r negligible relative to ||r|| * ||M^{-1} r||, or of a \
                 different sign than at the first iteration (singular or indefinite \
                 preconditioner)",
            ));
        }

        // ap = A * p
        a.apply(&p, &mut ap);

        // alpha = rz / (p^T * ap)
        let (pap, pp, apap) = dot_and_sums_of_squares(&p, &ap);

        if sign_breakdown(pap.clone(), pp.norm(), apap.norm(), &mut pap_sign) {
            if let Some(residual_norm) = converged_true_residual(a, b, &x, tol_abs.clone()) {
                return Ok(CgResult {
                    x,
                    iterations: iter,
                    residual_norm,
                    converged: true,
                });
            }
            return Err(breakdown(
                iter,
                "p^T * A * p negligible relative to ||p|| * ||A p||, or of a different \
                 sign than at the first iteration (singular or indefinite operator)",
            ));
        }

        let alpha = rz.clone() / pap;

        // x = x + alpha * p
        for i in 0..n {
            x[i] = x[i].clone() + alpha.clone() * p[i].clone();
        }

        // r = r - alpha * ap
        for i in 0..n {
            r[i] = r[i].clone() - alpha.clone() * ap[i].clone();
        }

        // z = M^{-1} * r
        z = precond(&r);

        // rz_new = r^T * z (with the norms of r and z)
        let (rz_new, r_squares_new, z_squares_new) = dot_and_sums_of_squares(&r, &z);

        // beta = rz_new / rz (rz passed the breakdown test above)
        let beta = rz_new.clone() / rz.clone();

        // p = z + beta * p
        for i in 0..n {
            p[i] = z[i].clone() + beta.clone() * p[i].clone();
        }

        rz = rz_new;
        r_squares = r_squares_new;
        z_squares = z_squares_new;
    }

    Ok(CgResult {
        x,
        iterations: max_iter,
        residual_norm: r_squares.norm(),
        converged: false,
    })
}
