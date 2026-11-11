//! BiCGStab (Bi-Conjugate Gradient Stabilized) solver.

use super::helpers::{
    breakdown, converged_true_residual, dot, dot_and_sums_of_squares, negligible_inner, norm,
};
use super::scaling::{Operator, SystemScaling};
use super::types::{CgResult, IterativeError};
use crate::csr::CsrMatrix;
use oxiblas_core::scalar::{Field, Real, Scalar};

/// BiCGStab for general (non-symmetric) `A`, with the shadow residual
/// `r̂ = r₀`.
///
/// Converges when the recursively updated residual (or the intermediate
/// residual `s`) satisfies `‖·‖ <= tol·‖b‖` (overflow-safe norms; see the
/// [module documentation](super) for systems far from unit scale).
///
/// # Breakdown
///
/// All tests are scale-invariant Cauchy–Schwarz forms:
/// * `|r̂ᵀv| <= eps·‖r̂‖·‖v‖` (the step length `alpha` is undefined);
/// * `|tᵀs| <= eps·‖t‖·‖s‖` (the stabilization parameter `omega` vanishes,
///   which includes `t = A s = 0`);
/// * `|r̂ᵀr| <= eps·‖r̂‖·‖r‖` (the next `rho` vanishes).
///
/// On a breakdown the true residual `‖b − A x‖` decides: a converged result if
/// it meets `tol·‖b‖`, [`IterativeError::Breakdown`] otherwise.
pub fn bicgstab<T: Scalar<Real = T> + Clone + Field + Real>(
    a: &CsrMatrix<T>,
    b: &[T],
    x0: &[T],
    tol: T,
    max_iter: usize,
) -> Result<CgResult<T>, IterativeError> {
    let scaling = SystemScaling::for_rhs(a, b);
    let mut result = bicgstab_scaled(
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

/// [`bicgstab`] on the (rescaled) system `a x = b`.
fn bicgstab_scaled<T: Scalar<Real = T> + Clone + Field + Real>(
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
    let mut v = vec![T::zero(); n];
    let mut s = vec![T::zero(); n];
    let mut t = vec![T::zero(); n];

    // r = b - A*x
    a.residual(b, &x, &mut r);

    // r_hat = r (shadow residual, kept constant)
    let r_hat = r.clone();
    let r_hat_norm = norm(&r_hat);

    // p = r
    p.clone_from_slice(&r);

    let mut rho = dot(&r_hat, &r);
    let mut r_norm = norm(&r);

    let b_norm = norm(b);
    let tol_abs = tol.clone() * b_norm.clone();

    for iter in 0..max_iter {
        if r_norm <= tol_abs {
            return Ok(CgResult {
                x,
                iterations: iter,
                residual_norm: r_norm,
                converged: true,
            });
        }

        // v = A * p
        a.apply(&p, &mut v);

        // alpha = rho / (r_hat^T * v)
        let (rv, _, vv) = dot_and_sums_of_squares(&r_hat, &v);

        if negligible_inner(rv.clone(), r_hat_norm.clone(), vv.norm()) {
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
                "r_hat^T * v negligible relative to ||r_hat|| * ||v||",
            ));
        }

        let alpha = rho.clone() / rv;

        // s = r - alpha * v
        for i in 0..n {
            s[i] = r[i].clone() - alpha.clone() * v[i].clone();
        }

        // Check if s is small enough
        let s_norm = norm(&s);
        if s_norm <= tol_abs {
            // x = x + alpha * p
            for i in 0..n {
                x[i] = x[i].clone() + alpha.clone() * p[i].clone();
            }
            return Ok(CgResult {
                x,
                iterations: iter,
                residual_norm: s_norm,
                converged: true,
            });
        }

        // t = A * s
        a.apply(&s, &mut t);

        // omega = (t^T * s) / (t^T * t)
        let (ts, tt, _) = dot_and_sums_of_squares(&t, &s);

        if negligible_inner(ts.clone(), tt.norm(), s_norm.clone()) {
            // The half step x + alpha*p (whose residual is s) is the best
            // available iterate.
            for i in 0..n {
                x[i] = x[i].clone() + alpha.clone() * p[i].clone();
            }
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
                "t^T * s negligible relative to ||t|| * ||s|| (omega vanishes)",
            ));
        }

        let omega = ts / tt.value();

        // x = x + alpha * p + omega * s
        for i in 0..n {
            x[i] = x[i].clone() + alpha.clone() * p[i].clone() + omega.clone() * s[i].clone();
        }

        // r = s - omega * t
        for i in 0..n {
            r[i] = s[i].clone() - omega.clone() * t[i].clone();
        }

        // rho_new = r_hat^T * r
        let (rho_new, _, rr) = dot_and_sums_of_squares(&r_hat, &r);
        r_norm = rr.norm();

        if negligible_inner(rho_new.clone(), r_hat_norm.clone(), r_norm.clone()) {
            if let Some(residual_norm) = converged_true_residual(a, b, &x, tol_abs.clone()) {
                return Ok(CgResult {
                    x,
                    iterations: iter + 1,
                    residual_norm,
                    converged: true,
                });
            }
            return Err(breakdown(
                iter,
                "r_hat^T * r negligible relative to ||r_hat|| * ||r|| (rho vanishes)",
            ));
        }

        // beta = (rho_new / rho) * (alpha / omega)
        let beta = (rho_new.clone() / rho.clone()) * (alpha / omega.clone());

        // p = r + beta * (p - omega * v)
        for i in 0..n {
            p[i] = r[i].clone() + beta.clone() * (p[i].clone() - omega.clone() * v[i].clone());
        }

        rho = rho_new;
    }

    Ok(CgResult {
        x,
        iterations: max_iter,
        residual_norm: r_norm,
        converged: false,
    })
}
