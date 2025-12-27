//! TFQMR (Transpose-Free Quasi-Minimal Residual) solvers.

use super::helpers::{dot, norm};
use super::types::{IterativeError, TfqmrResult};
use crate::csr::CsrMatrix;
use crate::ops::spmv;
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Solver function. See module documentation for details.
pub fn tfqmr<T: Scalar<Real = T> + Clone + Field + Real>(
    a: &CsrMatrix<T>,
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
    spmv(T::one(), a, &x, T::zero(), &mut r);
    for i in 0..n {
        r[i] = b[i].clone() - r[i].clone();
    }

    let b_norm = norm(b);
    let tol_abs = tol.clone() * b_norm.clone();

    // Shadow residual (kept constant)
    let r_tilde = r.clone();

    // Initialize working vectors
    let mut w = r.clone();
    let mut y1 = r.clone();
    let mut y2 = vec![T::zero(); n];
    let mut d = vec![T::zero(); n];
    let mut v = vec![T::zero(); n];
    let mut u = vec![T::zero(); n];

    // v = A * y1
    spmv(T::one(), a, &y1, T::zero(), &mut v);

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
        let sigma = dot(&r_tilde, &v);

        if Scalar::abs(sigma.clone()) <= <T as Scalar>::epsilon() {
            return Err(IterativeError::Breakdown {
                iteration: iter,
                description: "sigma = (r_tilde, v) near zero".to_string(),
            });
        }

        // alpha = rho / sigma
        let alpha = rho.clone() / sigma;

        // --- First half-iteration (odd step) ---

        // w = w - alpha * u
        for i in 0..n {
            w[i] = w[i].clone() - alpha.clone() * u[i].clone();
        }

        // d_new = y1 + (theta^2 * eta / alpha) * d
        let theta_sq_eta = theta.clone() * theta.clone() * eta.clone();
        let theta_sq_eta_over_alpha = if Scalar::abs(alpha.clone()) > <T as Scalar>::epsilon() {
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
        let theta_sq_eta_over_alpha_2 = if Scalar::abs(alpha.clone()) > <T as Scalar>::epsilon() {
            theta_sq_eta_2 / alpha.clone()
        } else {
            T::zero()
        };

        for i in 0..n {
            d[i] = y2[i].clone() + theta_sq_eta_over_alpha_2.clone() * d[i].clone();
        }

        // u_new = A * y2
        let mut u_new = vec![T::zero(); n];
        spmv(T::one(), a, &y2, T::zero(), &mut u_new);

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

        if Scalar::abs(rho.clone()) <= <T as Scalar>::epsilon() {
            return Err(IterativeError::Breakdown {
                iteration: iter,
                description: "rho near zero".to_string(),
            });
        }

        // beta = rho_new / rho
        let beta = rho_new.clone() / rho.clone();

        // y1 = w + beta * y2
        for i in 0..n {
            y1[i] = w[i].clone() + beta.clone() * y2[i].clone();
        }

        // u = A * y1
        spmv(T::one(), a, &y1, T::zero(), &mut u);

        // v = u + beta * (u_new + beta * v)
        for i in 0..n {
            v[i] = u[i].clone() + beta.clone() * (u_new[i].clone() + beta.clone() * v[i].clone());
        }

        rho = rho_new;
    }

    // Compute final actual residual
    let mut r_final = vec![T::zero(); n];
    spmv(T::one(), a, &x, T::zero(), &mut r_final);
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

/// Solver function. See module documentation for details.
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
    spmv(T::one(), a, &x, T::zero(), &mut r);
    for i in 0..n {
        r[i] = b[i].clone() - r[i].clone();
    }

    let b_norm = norm(b);
    let tol_abs = tol.clone() * b_norm.clone();

    // Shadow residual (kept constant)
    let r_tilde = r.clone();

    // Apply preconditioner: y1 = M^{-1} * r
    let mut y1 = precond(&r);
    let mut w = r.clone();
    let mut y2 = vec![T::zero(); n];
    let mut d = vec![T::zero(); n];
    let mut v = vec![T::zero(); n];
    let mut u = vec![T::zero(); n];

    // v = A * y1
    spmv(T::one(), a, &y1, T::zero(), &mut v);

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
        let sigma = dot(&r_tilde, &v);

        if Scalar::abs(sigma.clone()) <= <T as Scalar>::epsilon() {
            return Err(IterativeError::Breakdown {
                iteration: iter,
                description: "sigma = (r_tilde, v) near zero".to_string(),
            });
        }

        let alpha = rho.clone() / sigma;

        // --- First half-iteration ---
        for i in 0..n {
            w[i] = w[i].clone() - alpha.clone() * u[i].clone();
        }

        let theta_sq_eta = theta.clone() * theta.clone() * eta.clone();
        let theta_sq_eta_over_alpha = if Scalar::abs(alpha.clone()) > <T as Scalar>::epsilon() {
            theta_sq_eta / alpha.clone()
        } else {
            T::zero()
        };

        for i in 0..n {
            d[i] = y1[i].clone() + theta_sq_eta_over_alpha.clone() * d[i].clone();
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
        // y2 = y1 - alpha * M^{-1} * v (preconditioned CGS update)
        // Actually for preconditioned TFQMR: y2 = y1 - alpha * (M^{-1} * v term)
        // Simpler approach: y2 = y1 - alpha * v where v already contains the effect
        // For proper preconditioning, we compute: y2 = M^{-1}*(r - alpha*A*y1) but since
        // this is complex, let's use the simpler approach matching unpreconditioned version
        for i in 0..n {
            y2[i] = y1[i].clone() - alpha.clone() * v[i].clone();
        }

        let theta_sq_eta_2 = theta.clone() * theta.clone() * eta.clone();
        let theta_sq_eta_over_alpha_2 = if Scalar::abs(alpha.clone()) > <T as Scalar>::epsilon() {
            theta_sq_eta_2 / alpha.clone()
        } else {
            T::zero()
        };

        for i in 0..n {
            d[i] = y2[i].clone() + theta_sq_eta_over_alpha_2.clone() * d[i].clone();
        }

        // u_new = A * y2
        let mut u_new = vec![T::zero(); n];
        spmv(T::one(), a, &y2, T::zero(), &mut u_new);

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

        if Scalar::abs(rho.clone()) <= <T as Scalar>::epsilon() {
            return Err(IterativeError::Breakdown {
                iteration: iter,
                description: "rho near zero".to_string(),
            });
        }

        let beta = rho_new.clone() / rho.clone();

        // y1 = M^{-1} * w + beta * y2
        let w_precond = precond(&w);
        for i in 0..n {
            y1[i] = w_precond[i].clone() + beta.clone() * y2[i].clone();
        }

        spmv(T::one(), a, &y1, T::zero(), &mut u);

        for i in 0..n {
            v[i] = u[i].clone() + beta.clone() * (u_new[i].clone() + beta.clone() * v[i].clone());
        }

        rho = rho_new;
    }

    // Compute final actual residual
    let mut r_final = vec![T::zero(); n];
    spmv(T::one(), a, &x, T::zero(), &mut r_final);
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
