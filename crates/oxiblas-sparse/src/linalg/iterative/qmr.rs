//! QMR (Quasi-Minimal Residual) solvers.

use super::helpers::{dot, norm};
use super::types::{IterativeError, QmrResult};
use crate::csr::CsrMatrix;
use crate::ops::{spmv, spmv_transpose};
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Solver function. See module documentation for details.
pub fn qmr<T: Scalar<Real = T> + Clone + Field + Real>(
    a: &CsrMatrix<T>,
    b: &[T],
    x0: &[T],
    tol: T,
    max_iter: usize,
) -> Result<QmrResult<T>, IterativeError> {
    let n = a.nrows();

    if b.len() != n || x0.len() != n {
        return Err(IterativeError::DimensionMismatch {
            expected: n,
            actual: b.len().min(x0.len()),
        });
    }

    if a.nrows() != a.ncols() {
        return Err(IterativeError::DimensionMismatch {
            expected: a.nrows(),
            actual: a.ncols(),
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
    if b_norm <= T::zero() {
        return Ok(QmrResult {
            x: vec![T::zero(); n],
            iterations: 0,
            residual_norm: T::zero(),
            converged: true,
            residual_history: vec![T::zero()],
        });
    }
    let tol_abs = tol.clone() * b_norm.clone();

    let r_norm = norm(&r);
    residual_history.push(r_norm.clone());

    if r_norm <= tol_abs {
        return Ok(QmrResult {
            x,
            iterations: 0,
            residual_norm: r_norm,
            converged: true,
            residual_history,
        });
    }

    // BiCG vectors
    let r_tilde = r.clone(); // Shadow residual (kept constant)
    let mut p = r.clone();
    let mut p_tilde = r.clone();

    // QMR smoothing variables
    let mut d = vec![T::zero(); n];

    // Scalars
    let mut rho = dot(&r_tilde, &r);
    let mut tau = r_norm.clone();
    let mut theta = T::zero();
    let mut eta = T::zero();

    for iter in 0..max_iter {
        // v = A * p
        let mut v = vec![T::zero(); n];
        spmv(T::one(), a, &p, T::zero(), &mut v);

        // sigma = <r_tilde, v>
        let sigma = dot(&r_tilde, &v);

        // Compute alpha
        let alpha = if Scalar::abs(sigma.clone())
            > <T as Scalar>::epsilon() * T::from_f64(1e10).unwrap_or_else(T::zero)
        {
            rho.clone() / sigma.clone()
        } else {
            // Stagnation - return best current solution
            let mut r_final = vec![T::zero(); n];
            spmv(T::one(), a, &x, T::zero(), &mut r_final);
            for i in 0..n {
                r_final[i] = b[i].clone() - r_final[i].clone();
            }
            let actual_residual = norm(&r_final);

            return Ok(QmrResult {
                x,
                iterations: iter,
                residual_norm: actual_residual,
                converged: actual_residual <= tol_abs,
                residual_history,
            });
        };

        // Update residual: r = r - alpha * v
        for i in 0..n {
            r[i] = r[i].clone() - alpha.clone() * v[i].clone();
        }

        // QMR smoothing step
        let r_norm_new = norm(&r);
        let theta_new = if tau > T::zero() {
            r_norm_new.clone() / tau.clone()
        } else {
            T::one()
        };
        let c_sq = T::one() / (T::one() + theta_new.clone() * theta_new.clone());
        let c = Real::sqrt(c_sq.clone());
        let tau_new = tau.clone() * theta_new.clone() * c.clone();
        let eta_new = c_sq.clone() * alpha.clone();

        // Update d: d = p + (theta^2 * eta / alpha) * d
        let coeff = if Scalar::abs(alpha.clone()) > <T as Scalar>::epsilon() {
            theta.clone() * theta.clone() * eta.clone() / alpha.clone()
        } else {
            T::zero()
        };
        for i in 0..n {
            d[i] = p[i].clone() + coeff.clone() * d[i].clone();
        }

        // Update x: x = x + eta * d
        for i in 0..n {
            x[i] = x[i].clone() + eta_new.clone() * d[i].clone();
        }

        theta = theta_new;
        tau = tau_new.clone();
        eta = eta_new;

        residual_history.push(tau.clone());

        // Check convergence
        if tau <= tol_abs {
            let mut r_final = vec![T::zero(); n];
            spmv(T::one(), a, &x, T::zero(), &mut r_final);
            for i in 0..n {
                r_final[i] = b[i].clone() - r_final[i].clone();
            }
            let actual_residual = norm(&r_final);

            if actual_residual <= tol_abs {
                return Ok(QmrResult {
                    x,
                    iterations: iter + 1,
                    residual_norm: actual_residual,
                    converged: true,
                    residual_history,
                });
            }
        }

        // Update for transpose direction: p_tilde_new
        let mut v_tilde = vec![T::zero(); n];
        spmv_transpose(T::one(), a, &p_tilde, T::zero(), &mut v_tilde);

        // rho_new = <r_tilde, r>
        let rho_new = dot(&r_tilde, &r);

        // Compute beta
        let beta = if Scalar::abs(rho.clone())
            > <T as Scalar>::epsilon() * T::from_f64(1e10).unwrap_or_else(T::zero)
        {
            rho_new.clone() / rho.clone()
        } else {
            T::from_f64(0.1).unwrap_or_else(T::zero) // Small beta to continue
        };

        // Update p: p = r + beta * p
        for i in 0..n {
            p[i] = r[i].clone() + beta.clone() * p[i].clone();
        }

        // Update p_tilde (for transpose)
        for i in 0..n {
            p_tilde[i] = p_tilde[i].clone() - alpha.clone() * v_tilde[i].clone();
            p_tilde[i] = p_tilde[i].clone() * beta.clone() + r[i].clone();
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

    Ok(QmrResult {
        x,
        iterations: max_iter,
        residual_norm: final_residual,
        converged: false,
        residual_history,
    })
}

/// Solver function. See module documentation for details.
pub fn pqmr<T, FL, FR>(
    a: &CsrMatrix<T>,
    b: &[T],
    x0: &[T],
    left_precond: FL,
    right_precond: FR,
    tol: T,
    max_iter: usize,
) -> Result<QmrResult<T>, IterativeError>
where
    T: Scalar<Real = T> + Clone + Field + Real,
    FL: Fn(&[T]) -> Vec<T>,
    FR: Fn(&[T]) -> Vec<T>,
{
    let n = a.nrows();

    if b.len() != n || x0.len() != n {
        return Err(IterativeError::DimensionMismatch {
            expected: n,
            actual: b.len().min(x0.len()),
        });
    }

    if a.nrows() != a.ncols() {
        return Err(IterativeError::DimensionMismatch {
            expected: a.nrows(),
            actual: a.ncols(),
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

    // Apply left preconditioner: r_hat = M1^{-1} * r
    let r_hat = left_precond(&r);

    let b_precond = left_precond(b);
    let b_norm = norm(&b_precond);
    if b_norm <= T::zero() {
        return Ok(QmrResult {
            x: vec![T::zero(); n],
            iterations: 0,
            residual_norm: T::zero(),
            converged: true,
            residual_history: vec![T::zero()],
        });
    }
    let tol_abs = tol.clone() * b_norm.clone();

    let r_hat_norm = norm(&r_hat);
    residual_history.push(r_hat_norm.clone());

    if r_hat_norm <= tol_abs {
        return Ok(QmrResult {
            x,
            iterations: 0,
            residual_norm: r_hat_norm,
            converged: true,
            residual_history,
        });
    }

    // Initialize bi-Lanczos vectors
    let mut v = r_hat
        .iter()
        .map(|ri| ri.clone() / r_hat_norm.clone())
        .collect::<Vec<_>>();
    let mut w = v.clone();

    let mut v_old = vec![T::zero(); n];
    let mut w_old = vec![T::zero(); n];

    let mut rho = r_hat_norm.clone();
    let mut xi = r_hat_norm.clone();
    let mut gamma = T::one();
    let mut eta = T::from_f64(-1.0).unwrap_or_else(T::zero);
    let mut delta: T;
    let mut eps: T;
    let mut _theta = T::zero();
    let mut theta_old = T::zero();

    let mut p = vec![T::zero(); n];
    let mut p_old = vec![T::zero(); n];

    let mut y;
    let mut z;

    for iter in 0..max_iter {
        // Apply right preconditioner then matrix: y = M1^{-1} * A * M2^{-1} * v
        let v_right = right_precond(&v);
        let mut av = vec![T::zero(); n];
        spmv(T::one(), a, &v_right, T::zero(), &mut av);
        y = left_precond(&av);

        // For transpose: z = M2^{-T} * A^T * M1^{-T} * w
        // Simplified: assume symmetric preconditioners for now
        let w_right = left_precond(&w); // M1^{-T} ≈ M1^{-1} for symmetric
        let mut atw = vec![T::zero(); n];
        spmv_transpose(T::one(), a, &w_right, T::zero(), &mut atw);
        z = right_precond(&atw); // M2^{-T} ≈ M2^{-1} for symmetric

        delta = dot(&w, &y);

        if Scalar::abs(delta.clone()) <= <T as Scalar>::epsilon() {
            return Err(IterativeError::Breakdown {
                iteration: iter,
                description: "delta near zero".to_string(),
            });
        }

        let alpha = delta.clone() / (rho.clone() * xi.clone());

        if iter == 0 {
            for i in 0..n {
                p[i] = v_right[i].clone();
            }
        } else {
            let c = theta_old.clone() * gamma.clone() * alpha.clone();
            let p_right = right_precond(&p_old);
            for i in 0..n {
                p[i] = v_right[i].clone() - c.clone() * p_right[i].clone();
            }
        }

        for i in 0..n {
            y[i] = y[i].clone() - alpha.clone() * v_old[i].clone();
        }

        let y_norm = norm(&y);
        eps = y_norm.clone();

        if eps <= <T as Scalar>::epsilon() {
            for i in 0..n {
                x[i] = x[i].clone() + (T::one() / gamma.clone()) * p[i].clone();
            }
            let mut r_final = vec![T::zero(); n];
            spmv(T::one(), a, &x, T::zero(), &mut r_final);
            for i in 0..n {
                r_final[i] = b[i].clone() - r_final[i].clone();
            }
            let final_norm = norm(&r_final);
            residual_history.push(final_norm.clone());

            return Ok(QmrResult {
                x,
                iterations: iter + 1,
                residual_norm: final_norm,
                converged: true,
                residual_history,
            });
        }

        let theta_new = eps.clone() / (gamma.clone() * Scalar::abs(alpha.clone()));
        let c_sq = T::one() / (T::one() + theta_new.clone() * theta_new.clone());
        let c = Real::sqrt(c_sq.clone());
        let gamma_new = gamma.clone() * c.clone();
        let eta_new =
            T::from_f64(-1.0).unwrap_or_else(T::zero) * eta.clone() * rho.clone() * c_sq.clone()
                / alpha.clone();

        for i in 0..n {
            x[i] = x[i].clone() + eta_new.clone() * p[i].clone();
        }

        let quasi_residual = r_hat_norm.clone()
            * Real::sqrt(T::one() + theta_new.clone() * theta_new.clone())
            * gamma_new.clone();
        residual_history.push(quasi_residual.clone());

        if quasi_residual <= tol_abs {
            let mut r_final = vec![T::zero(); n];
            spmv(T::one(), a, &x, T::zero(), &mut r_final);
            for i in 0..n {
                r_final[i] = b[i].clone() - r_final[i].clone();
            }
            let actual_residual = norm(&r_final);

            if actual_residual <= tol_abs * b_norm.clone() / norm(&b_precond) {
                return Ok(QmrResult {
                    x,
                    iterations: iter + 1,
                    residual_norm: actual_residual,
                    converged: true,
                    residual_history,
                });
            }
        }

        for i in 0..n {
            z[i] = z[i].clone() - alpha.clone() * w_old[i].clone();
        }

        let z_norm = norm(&z);

        if z_norm <= <T as Scalar>::epsilon() {
            return Err(IterativeError::Breakdown {
                iteration: iter,
                description: "z norm near zero".to_string(),
            });
        }

        let rho_new_unnorm = dot(&y, &z);

        if Scalar::abs(rho_new_unnorm.clone()) <= <T as Scalar>::epsilon() {
            return Err(IterativeError::Breakdown {
                iteration: iter,
                description: "rho_new near zero".to_string(),
            });
        }

        let rho_new = eps.clone();
        let xi_new = z_norm.clone();

        v_old.clone_from_slice(&v);
        w_old.clone_from_slice(&w);
        p_old.clone_from_slice(&p);

        for i in 0..n {
            v[i] = y[i].clone() / rho_new.clone();
            w[i] = z[i].clone() / xi_new.clone();
        }

        theta_old = theta_new.clone();
        _theta = theta_new;
        gamma = gamma_new;
        eta = eta_new;
        rho = rho_new;
        xi = xi_new;
    }

    let mut r_final = vec![T::zero(); n];
    spmv(T::one(), a, &x, T::zero(), &mut r_final);
    for i in 0..n {
        r_final[i] = b[i].clone() - r_final[i].clone();
    }
    let final_residual = norm(&r_final);

    Ok(QmrResult {
        x,
        iterations: max_iter,
        residual_norm: final_residual,
        converged: false,
        residual_history,
    })
}
