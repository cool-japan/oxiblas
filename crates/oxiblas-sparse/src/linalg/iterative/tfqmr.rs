//! TFQMR (Transpose-Free Quasi-Minimal Residual) solvers.

use super::helpers::{
    breakdown, converged_true_residual, dot, dot_and_sums_of_squares, negligible_inner, norm,
};
use super::scaling::{Operator, PrecondScaling, SystemScaling};
use super::types::{IterativeError, TfqmrResult};
use crate::csr::CsrMatrix;
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Maps a result of the rescaled system back to the original one.
fn unscale_tfqmr<T: Scalar<Real = T> + Clone + Field + Real>(
    scaling: &SystemScaling<T>,
    mut result: TfqmrResult<T>,
) -> TfqmrResult<T> {
    scaling.solution(&mut result.x);
    result.residual_norm = scaling.residual(result.residual_norm);
    scaling.residuals(&mut result.residual_history);
    result
}

/// TFQMR for general (non-symmetric) `A`, with the shadow residual `r̃ = r₀`.
///
/// Converges when the quasi-residual `tau` satisfies `tau <= tol·‖b‖`
/// (`‖r_m‖ <= sqrt(m + 1)·tau_m`). `iterations` counts half-steps. Norms are
/// overflow-safe; see the [module documentation](super) for systems far
/// from unit scale.
///
/// # Breakdown
///
/// Both Lanczos breakdowns are detected with scale-invariant Cauchy–Schwarz
/// tests: `|r̃ᵀv| <= eps·‖r̃‖·‖v‖` (`sigma` vanishes) and
/// `|r̃ᵀw| <= eps·‖r̃‖·‖w‖` (the next `rho` vanishes). The true residual
/// `‖b − A x‖` then decides: a converged result if it meets `tol·‖b‖`,
/// [`IterativeError::Breakdown`] otherwise.
pub fn tfqmr<T: Scalar<Real = T> + Clone + Field + Real>(
    a: &CsrMatrix<T>,
    b: &[T],
    x0: &[T],
    tol: T,
    max_iter: usize,
) -> Result<TfqmrResult<T>, IterativeError> {
    let scaling = SystemScaling::for_rhs(a, b);
    let result = tfqmr_scaled(
        &scaling.operator(a),
        &scaling.rhs(b),
        &scaling.initial_guess(x0),
        tol,
        max_iter,
    )?;
    Ok(unscale_tfqmr(&scaling, result))
}

/// [`tfqmr`] on the (rescaled) system `a x = b`.
fn tfqmr_scaled<T: Scalar<Real = T> + Clone + Field + Real>(
    a: &Operator<'_, T>,
    b: &[T],
    x0: &[T],
    tol: T,
    max_iter: usize,
) -> Result<TfqmrResult<T>, IterativeError> {
    let n = a.nrows();

    if b.len() != n || x0.len() != n {
        return Err(IterativeError::DimensionMismatch {
            expected: n,
            actual: b.len().min(x0.len()),
        });
    }

    let mut x = x0.to_vec();
    let mut residual_history = Vec::new();

    // Initial residual: r0 = b - A*x0
    let mut r = vec![T::zero(); n];
    a.apply(&x, &mut r);
    for i in 0..n {
        r[i] = b[i].clone() - r[i].clone();
    }

    let b_norm = norm(b);
    let tol_abs = tol.clone() * b_norm.clone();

    // Shadow residual (kept constant)
    let r_tilde = r.clone();
    let r_tilde_norm = norm(&r_tilde);

    // Initialize working vectors
    let mut w = r.clone();
    let mut y1 = r.clone();
    let mut y2 = vec![T::zero(); n];
    let mut d = vec![T::zero(); n];
    let mut v = vec![T::zero(); n];
    let mut u = vec![T::zero(); n];

    // v = A * y1
    a.apply(&y1, &mut v);

    // u = v
    u.clone_from_slice(&v);

    // rho = (r_tilde, r)
    let mut rho = dot(&r_tilde, &r);

    // tau = ||r||
    let mut tau = norm(&r);
    residual_history.push(tau.clone());

    if tau <= tol_abs {
        return Ok(TfqmrResult {
            x,
            iterations: 0,
            residual_norm: tau,
            converged: true,
            residual_history,
        });
    }

    // theta and eta initialized
    let mut theta = T::zero();
    let mut eta = T::zero();

    for iter in 0..max_iter {
        // sigma = (r_tilde, v)
        let (sigma, _, vv) = dot_and_sums_of_squares(&r_tilde, &v);

        if negligible_inner(sigma.clone(), r_tilde_norm.clone(), vv.norm()) {
            if let Some(residual_norm) = converged_true_residual(a, b, &x, tol_abs.clone()) {
                return Ok(TfqmrResult {
                    x,
                    iterations: 2 * iter,
                    residual_norm,
                    converged: true,
                    residual_history,
                });
            }
            return Err(breakdown(
                iter,
                "sigma = (r_tilde, v) negligible relative to ||r_tilde|| * ||v||",
            ));
        }

        // alpha = rho / sigma (nonzero: rho passed the breakdown test)
        let alpha = rho.clone() / sigma;

        // --- First half-iteration (odd step) ---

        // w = w - alpha * u
        for i in 0..n {
            w[i] = w[i].clone() - alpha.clone() * u[i].clone();
        }

        // d_new = y1 + (theta^2 * eta / alpha) * d
        let theta_sq_eta = theta.clone() * theta.clone() * eta.clone();
        let theta_sq_eta_over_alpha = if alpha != T::zero() {
            theta_sq_eta / alpha.clone()
        } else {
            T::zero()
        };

        for i in 0..n {
            d[i] = y1[i].clone() + theta_sq_eta_over_alpha.clone() * d[i].clone();
        }

        // theta_new = ||w|| / tau
        let w_norm = norm(&w);
        let theta_new = if tau > T::zero() {
            w_norm.clone() / tau.clone()
        } else {
            T::zero()
        };

        // c = 1 / sqrt(1 + theta_new^2)
        let c_sq_inv = T::one() + theta_new.clone() * theta_new.clone();
        let c = T::one() / Real::sqrt(c_sq_inv);

        // tau_new = tau * theta_new * c
        let tau_new = tau.clone() * theta_new.clone() * c.clone();

        // eta_new = c^2 * alpha
        let eta_new = c.clone() * c.clone() * alpha.clone();

        // x = x + eta_new * d
        for i in 0..n {
            x[i] = x[i].clone() + eta_new.clone() * d[i].clone();
        }

        theta = theta_new;
        tau = tau_new.clone();
        eta = eta_new.clone();

        residual_history.push(tau.clone());

        // Check convergence
        if tau <= tol_abs {
            return Ok(TfqmrResult {
                x,
                iterations: 2 * iter + 1,
                residual_norm: tau,
                converged: true,
                residual_history,
            });
        }

        // --- Second half-iteration (even step) ---

        // y2 = y1 - alpha * v (CGS update for y)
        for i in 0..n {
            y2[i] = y1[i].clone() - alpha.clone() * v[i].clone();
        }

        // d_new = y2 + (theta^2 * eta / alpha) * d
        let theta_sq_eta_2 = theta.clone() * theta.clone() * eta.clone();
        let theta_sq_eta_over_alpha_2 = if alpha != T::zero() {
            theta_sq_eta_2 / alpha.clone()
        } else {
            T::zero()
        };

        for i in 0..n {
            d[i] = y2[i].clone() + theta_sq_eta_over_alpha_2.clone() * d[i].clone();
        }

        // u_new = A * y2
        let mut u_new = vec![T::zero(); n];
        a.apply(&y2, &mut u_new);

        // w = w - alpha * u_new
        for i in 0..n {
            w[i] = w[i].clone() - alpha.clone() * u_new[i].clone();
        }

        // theta_new = ||w|| / tau
        let w_norm_2 = norm(&w);
        let theta_new_2 = if tau > T::zero() {
            w_norm_2.clone() / tau.clone()
        } else {
            T::zero()
        };

        // c = 1 / sqrt(1 + theta_new^2)
        let c_sq_inv_2 = T::one() + theta_new_2.clone() * theta_new_2.clone();
        let c_2 = T::one() / Real::sqrt(c_sq_inv_2);

        // tau_new = tau * theta_new * c
        let tau_new_2 = tau.clone() * theta_new_2.clone() * c_2.clone();

        // eta_new = c^2 * alpha
        let eta_new_2 = c_2.clone() * c_2.clone() * alpha.clone();

        // x = x + eta_new * d
        for i in 0..n {
            x[i] = x[i].clone() + eta_new_2.clone() * d[i].clone();
        }

        theta = theta_new_2;
        tau = tau_new_2.clone();
        eta = eta_new_2;

        residual_history.push(tau.clone());

        // Check convergence
        if tau <= tol_abs {
            return Ok(TfqmrResult {
                x,
                iterations: 2 * iter + 2,
                residual_norm: tau,
                converged: true,
                residual_history,
            });
        }

        // Prepare for next iteration
        // rho_new = (r_tilde, w)
        let rho_new = dot(&r_tilde, &w);

        if negligible_inner(rho_new.clone(), r_tilde_norm.clone(), w_norm_2) {
            if let Some(residual_norm) = converged_true_residual(a, b, &x, tol_abs.clone()) {
                return Ok(TfqmrResult {
                    x,
                    iterations: 2 * iter + 2,
                    residual_norm,
                    converged: true,
                    residual_history,
                });
            }
            return Err(breakdown(
                iter,
                "rho = (r_tilde, w) negligible relative to ||r_tilde|| * ||w||",
            ));
        }

        // beta = rho_new / rho
        let beta = rho_new.clone() / rho.clone();

        // y1 = w + beta * y2
        for i in 0..n {
            y1[i] = w[i].clone() + beta.clone() * y2[i].clone();
        }

        // u = A * y1
        a.apply(&y1, &mut u);

        // v = u + beta * (u_new + beta * v)
        for i in 0..n {
            v[i] = u[i].clone() + beta.clone() * (u_new[i].clone() + beta.clone() * v[i].clone());
        }

        rho = rho_new;
    }

    // Compute final actual residual
    let mut r_final = vec![T::zero(); n];
    a.apply(&x, &mut r_final);
    for i in 0..n {
        r_final[i] = b[i].clone() - r_final[i].clone();
    }
    let final_residual = norm(&r_final);

    Ok(TfqmrResult {
        x,
        iterations: 2 * max_iter,
        residual_norm: final_residual,
        converged: false,
        residual_history,
    })
}

/// Right-preconditioned TFQMR; `precond` applies `M^{-1}`.
///
/// Runs TFQMR on `A M^{-1}` (all Krylov vectors live in residual space, so
/// `tau` estimates the true residual `‖b − A x‖`) and maps only the solution
/// update through `M^{-1}`: the smoothing direction is kept as `M^{-1} d`.
/// With the identity preconditioner this reproduces [`tfqmr`] exactly. For a
/// system far from unit scale `precond` receives vectors of the rescaled
/// system and must be linear (see the [module documentation](super)).
///
/// # Breakdown
///
/// Same scale-invariant rules as [`tfqmr`].
pub fn ptfqmr<T, F>(
    a: &CsrMatrix<T>,
    b: &[T],
    x0: &[T],
    precond: F,
    tol: T,
    max_iter: usize,
) -> Result<TfqmrResult<T>, IterativeError>
where
    T: Scalar<Real = T> + Clone + Field + Real,
    F: Fn(&[T]) -> Vec<T>,
{
    let scaling = SystemScaling::for_rhs(a, b);
    let precond_scaling = PrecondScaling::new();
    let result = ptfqmr_scaled(
        &scaling.operator(a),
        &scaling.rhs(b),
        &scaling.initial_guess(x0),
        |y: &[T]| precond_scaling.finish(y, precond(y)),
        tol,
        max_iter,
    )?;
    Ok(unscale_tfqmr(&scaling, result))
}

/// [`ptfqmr`] on the (rescaled) system `a x = b`.
fn ptfqmr_scaled<T, F>(
    a: &Operator<'_, T>,
    b: &[T],
    x0: &[T],
    precond: F,
    tol: T,
    max_iter: usize,
) -> Result<TfqmrResult<T>, IterativeError>
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
    let mut residual_history = Vec::new();

    // Initial residual: r0 = b - A*x0
    let mut r = vec![T::zero(); n];
    a.apply(&x, &mut r);
    for i in 0..n {
        r[i] = b[i].clone() - r[i].clone();
    }

    let b_norm = norm(b);
    let tol_abs = tol.clone() * b_norm.clone();

    // Shadow residual (kept constant)
    let r_tilde = r.clone();
    let r_tilde_norm = norm(&r_tilde);

    // Residual-space TFQMR vectors (u_m in the usual notation) and their
    // preconditioned images z = M^{-1} y.
    let mut w = r.clone();
    let mut y1 = r.clone();
    let mut z1 = precond(&y1);
    let mut y2 = vec![T::zero(); n];
    // Smoothing direction mapped to x-space: d = M^{-1} d_residual.
    let mut d = vec![T::zero(); n];
    let mut v = vec![T::zero(); n];
    let mut u = vec![T::zero(); n];

    // v = A * M^{-1} * y1
    a.apply(&z1, &mut v);

    // u = v
    u.clone_from_slice(&v);

    // rho = (r_tilde, r)
    let mut rho = dot(&r_tilde, &r);

    // tau = ||r||
    let mut tau = norm(&r);
    residual_history.push(tau.clone());

    if tau <= tol_abs {
        return Ok(TfqmrResult {
            x,
            iterations: 0,
            residual_norm: tau,
            converged: true,
            residual_history,
        });
    }

    let mut theta = T::zero();
    let mut eta = T::zero();

    for iter in 0..max_iter {
        let (sigma, _, vv) = dot_and_sums_of_squares(&r_tilde, &v);

        if negligible_inner(sigma.clone(), r_tilde_norm.clone(), vv.norm()) {
            if let Some(residual_norm) = converged_true_residual(a, b, &x, tol_abs.clone()) {
                return Ok(TfqmrResult {
                    x,
                    iterations: 2 * iter,
                    residual_norm,
                    converged: true,
                    residual_history,
                });
            }
            return Err(breakdown(
                iter,
                "sigma = (r_tilde, v) negligible relative to ||r_tilde|| * ||v||",
            ));
        }

        // alpha = rho / sigma (nonzero: rho passed the breakdown test)
        let alpha = rho.clone() / sigma;

        // --- First half-iteration ---
        for i in 0..n {
            w[i] = w[i].clone() - alpha.clone() * u[i].clone();
        }

        let theta_sq_eta = theta.clone() * theta.clone() * eta.clone();
        let theta_sq_eta_over_alpha = if alpha != T::zero() {
            theta_sq_eta / alpha.clone()
        } else {
            T::zero()
        };

        // d = M^{-1} y1 + (theta^2 * eta / alpha) * d
        for i in 0..n {
            d[i] = z1[i].clone() + theta_sq_eta_over_alpha.clone() * d[i].clone();
        }

        let w_norm = norm(&w);
        let theta_new = if tau > T::zero() {
            w_norm.clone() / tau.clone()
        } else {
            T::zero()
        };
        let c_sq_inv = T::one() + theta_new.clone() * theta_new.clone();
        let c = T::one() / Real::sqrt(c_sq_inv);
        let tau_new = tau.clone() * theta_new.clone() * c.clone();
        let eta_new = c.clone() * c.clone() * alpha.clone();

        for i in 0..n {
            x[i] = x[i].clone() + eta_new.clone() * d[i].clone();
        }

        theta = theta_new;
        tau = tau_new.clone();
        eta = eta_new.clone();
        residual_history.push(tau.clone());

        if tau <= tol_abs {
            return Ok(TfqmrResult {
                x,
                iterations: 2 * iter + 1,
                residual_norm: tau,
                converged: true,
                residual_history,
            });
        }

        // --- Second half-iteration ---
        // y2 = y1 - alpha * v (residual space), z2 = M^{-1} y2
        for i in 0..n {
            y2[i] = y1[i].clone() - alpha.clone() * v[i].clone();
        }
        let z2 = precond(&y2);

        let theta_sq_eta_2 = theta.clone() * theta.clone() * eta.clone();
        let theta_sq_eta_over_alpha_2 = if alpha != T::zero() {
            theta_sq_eta_2 / alpha.clone()
        } else {
            T::zero()
        };

        // d = M^{-1} y2 + (theta^2 * eta / alpha) * d
        for i in 0..n {
            d[i] = z2[i].clone() + theta_sq_eta_over_alpha_2.clone() * d[i].clone();
        }

        // u_new = A * M^{-1} * y2
        let mut u_new = vec![T::zero(); n];
        a.apply(&z2, &mut u_new);

        for i in 0..n {
            w[i] = w[i].clone() - alpha.clone() * u_new[i].clone();
        }

        let w_norm_2 = norm(&w);
        let theta_new_2 = if tau > T::zero() {
            w_norm_2.clone() / tau.clone()
        } else {
            T::zero()
        };
        let c_sq_inv_2 = T::one() + theta_new_2.clone() * theta_new_2.clone();
        let c_2 = T::one() / Real::sqrt(c_sq_inv_2);
        let tau_new_2 = tau.clone() * theta_new_2.clone() * c_2.clone();
        let eta_new_2 = c_2.clone() * c_2.clone() * alpha.clone();

        for i in 0..n {
            x[i] = x[i].clone() + eta_new_2.clone() * d[i].clone();
        }

        theta = theta_new_2;
        tau = tau_new_2.clone();
        eta = eta_new_2;
        residual_history.push(tau.clone());

        if tau <= tol_abs {
            return Ok(TfqmrResult {
                x,
                iterations: 2 * iter + 2,
                residual_norm: tau,
                converged: true,
                residual_history,
            });
        }

        // Prepare for next iteration
        let rho_new = dot(&r_tilde, &w);

        if negligible_inner(rho_new.clone(), r_tilde_norm.clone(), w_norm_2) {
            if let Some(residual_norm) = converged_true_residual(a, b, &x, tol_abs.clone()) {
                return Ok(TfqmrResult {
                    x,
                    iterations: 2 * iter + 2,
                    residual_norm,
                    converged: true,
                    residual_history,
                });
            }
            return Err(breakdown(
                iter,
                "rho = (r_tilde, w) negligible relative to ||r_tilde|| * ||w||",
            ));
        }

        let beta = rho_new.clone() / rho.clone();

        // y1 = w + beta * y2 (residual space), z1 = M^{-1} y1
        for i in 0..n {
            y1[i] = w[i].clone() + beta.clone() * y2[i].clone();
        }
        z1 = precond(&y1);

        // u = A * M^{-1} * y1
        a.apply(&z1, &mut u);

        for i in 0..n {
            v[i] = u[i].clone() + beta.clone() * (u_new[i].clone() + beta.clone() * v[i].clone());
        }

        rho = rho_new;
    }

    // Compute final actual residual
    let mut r_final = vec![T::zero(); n];
    a.apply(&x, &mut r_final);
    for i in 0..n {
        r_final[i] = b[i].clone() - r_final[i].clone();
    }
    let final_residual = norm(&r_final);

    Ok(TfqmrResult {
        x,
        iterations: 2 * max_iter,
        residual_norm: final_residual,
        converged: false,
        residual_history,
    })
}
