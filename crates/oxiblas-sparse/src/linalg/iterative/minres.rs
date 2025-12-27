//! MINRES (Minimum Residual) and Preconditioned MINRES solvers.

use super::helpers::{dot, norm};
use super::types::{IterativeError, MinresResult};
use crate::csr::CsrMatrix;
use crate::ops::spmv;
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Solver function. See module documentation for details.
pub fn minres<T: Scalar<Real = T> + Clone + Field + Real>(
    a: &CsrMatrix<T>,
    b: &[T],
    x0: &[T],
    tol: T,
    max_iter: usize,
) -> Result<MinresResult<T>, IterativeError> {
    let n = a.nrows();

    if b.len() != n || x0.len() != n {
        return Err(IterativeError::DimensionMismatch {
            expected: n,
            actual: b.len().min(x0.len()),
        });
    }

    let mut x = x0.to_vec();
    let mut residual_history = Vec::with_capacity(max_iter);

    // r = b - A*x0
    let mut r = vec![T::zero(); n];
    spmv(T::one(), a, &x, T::zero(), &mut r);
    for i in 0..n {
        r[i] = b[i].clone() - r[i].clone();
    }

    let b_norm = norm(b);
    let beta1 = norm(&r);
    residual_history.push(beta1.clone());

    // Check for immediate convergence
    if beta1 <= tol.clone() * b_norm.clone() {
        return Ok(MinresResult {
            x,
            iterations: 0,
            residual_norm: beta1,
            converged: true,
            residual_history,
        });
    }

    let tol_abs = tol.clone() * b_norm.clone();

    // Initialize Lanczos vectors
    let mut v_old = vec![T::zero(); n];
    let mut v = vec![T::zero(); n];
    for i in 0..n {
        v[i] = r[i].clone() / beta1.clone();
    }

    // Lanczos scalars
    let mut beta = beta1.clone();

    // Givens rotation parameters (need two previous rotations)
    // c = cosine, s = sine
    let mut c = T::one(); // current rotation cosine
    let mut s = T::zero(); // current rotation sine
    let mut c_old = T::one(); // previous rotation cosine
    let mut s_old = T::zero(); // previous rotation sine

    // Direction vectors for solution update (need two previous)
    let mut w = vec![T::zero(); n];
    let mut w_old = vec![T::zero(); n];

    // Right-hand side of transformed system
    let mut eta = beta1.clone();

    for k in 0..max_iter {
        // Lanczos step: compute v_new = A*v - alpha*v - beta*v_old
        let mut v_new = vec![T::zero(); n];
        spmv(T::one(), a, &v, T::zero(), &mut v_new);

        let alpha = dot(&v, &v_new);

        // v_new = v_new - alpha*v - beta*v_old
        for i in 0..n {
            v_new[i] =
                v_new[i].clone() - alpha.clone() * v[i].clone() - beta.clone() * v_old[i].clone();
        }

        let beta_new = norm(&v_new);

        // =====================================
        // QR factorization update via Givens rotations
        // =====================================
        // We process column k of the tridiagonal matrix T:
        //   [ ...         ]
        //   [ beta    r2  ]  <- row k-1
        //   [ alpha   r1  ]  <- row k   (after applying old rotations)
        //   [ beta_new 0  ]  <- row k+1 (will be eliminated)

        // Apply previous rotations to get intermediate values
        // r2 = element at (k-1, k) after applying G_{k-2}
        // r1 = element at (k, k) after applying G_{k-1}
        let c_oold = c_old.clone();
        let s_oold = s_old.clone();
        c_old = c.clone();
        s_old = s.clone();

        // Apply G_{k-2} and G_{k-1} to [beta, alpha]^T
        // After G_{k-2}: affects rows k-2, k-1 (position k-1 is beta)
        // After G_{k-1}: affects rows k-1, k (positions are the result from G_{k-2} and alpha)
        let r1 = c_old.clone() * alpha.clone() - c_oold.clone() * s_old.clone() * beta.clone();
        let r2 = s_old.clone() * alpha.clone() + c_oold.clone() * c_old.clone() * beta.clone();
        let r3 = s_oold.clone() * beta.clone();

        // Compute new Givens rotation to eliminate beta_new
        // [c  s] [r1      ]   [rho]
        // [-s c] [beta_new] = [0  ]
        let rho = Real::sqrt(r1.clone() * r1.clone() + beta_new.clone() * beta_new.clone());

        let c_new: T;
        let s_new: T;
        if Scalar::abs(rho.clone()) <= <T as Scalar>::epsilon() {
            c_new = T::one();
            s_new = T::zero();
        } else {
            c_new = r1.clone() / rho.clone();
            s_new = beta_new.clone() / rho.clone();
        }

        // Update direction vector: w_new = (v - r3*w_oold - r2*w_old) / rho
        // Here w is w_old from prev iteration, w_old is w_oold from prev iteration
        let mut w_new = vec![T::zero(); n];
        if Scalar::abs(rho.clone()) > <T as Scalar>::epsilon() {
            for i in 0..n {
                w_new[i] =
                    (v[i].clone() - r3.clone() * w_old[i].clone() - r2.clone() * w[i].clone())
                        / rho.clone();
            }
        }

        // Update right-hand side and solution
        // eta is the residual in the transformed system
        let eta_bar = c_new.clone() * eta.clone(); // contribution to solution
        let eta_new = T::zero() - s_new.clone() * eta.clone(); // remaining residual

        // Update solution: x = x + eta_bar * w_new
        for i in 0..n {
            x[i] = x[i].clone() + eta_bar.clone() * w_new[i].clone();
        }

        // Residual estimate: |eta_new|
        let res_norm = Scalar::abs(eta_new.clone());
        residual_history.push(res_norm.clone());

        // Check convergence
        if res_norm <= tol_abs {
            return Ok(MinresResult {
                x,
                iterations: k + 1,
                residual_norm: res_norm,
                converged: true,
                residual_history,
            });
        }

        // Check for breakdown
        if Scalar::abs(beta_new.clone()) <= <T as Scalar>::epsilon() {
            return Ok(MinresResult {
                x,
                iterations: k + 1,
                residual_norm: res_norm,
                converged: true,
                residual_history,
            });
        }

        // Prepare for next iteration
        for i in 0..n {
            v_old[i] = v[i].clone();
            v[i] = v_new[i].clone() / beta_new.clone();
            w_old[i] = w[i].clone();
            w[i] = w_new[i].clone();
        }

        beta = beta_new;
        c = c_new;
        s = s_new;
        eta = eta_new;
    }

    // Compute final residual
    let mut r_final = vec![T::zero(); n];
    spmv(T::one(), a, &x, T::zero(), &mut r_final);
    for i in 0..n {
        r_final[i] = b[i].clone() - r_final[i].clone();
    }
    let final_residual = norm(&r_final);

    Ok(MinresResult {
        x,
        iterations: max_iter,
        residual_norm: final_residual,
        converged: false,
        residual_history,
    })
}

/// Solver function. See module documentation for details.
pub fn pminres<T, F>(
    a: &CsrMatrix<T>,
    b: &[T],
    x0: &[T],
    precond: F,
    tol: T,
    max_iter: usize,
) -> Result<MinresResult<T>, IterativeError>
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
    let mut residual_history = Vec::with_capacity(max_iter);

    // r = b - A*x0
    let mut r = vec![T::zero(); n];
    spmv(T::one(), a, &x, T::zero(), &mut r);
    for i in 0..n {
        r[i] = b[i].clone() - r[i].clone();
    }

    let b_norm = norm(b);

    // y = M^{-1} * r (preconditioned residual)
    let mut y = precond(&r);

    // beta1 = sqrt(r^T * y) = sqrt(r^T * M^{-1} * r) (M-norm of r)
    let ry = dot(&r, &y);
    if ry <= T::zero() {
        return Err(IterativeError::Breakdown {
            iteration: 0,
            description: "Preconditioner not positive definite".to_string(),
        });
    }
    let beta1 = Real::sqrt(ry);
    residual_history.push(norm(&r));

    // Check for immediate convergence
    let r_norm = norm(&r);
    if r_norm <= tol.clone() * b_norm.clone() {
        return Ok(MinresResult {
            x,
            iterations: 0,
            residual_norm: r_norm,
            converged: true,
            residual_history,
        });
    }

    let tol_abs = tol.clone() * b_norm.clone();

    // Initialize preconditioned Lanczos vectors
    // v = M^{-1} r / beta1 (preconditioned direction)
    // p = r / beta1 (residual direction, p = M * v in exact arithmetic)
    let mut v_old = vec![T::zero(); n];
    let mut v = vec![T::zero(); n];
    let mut p_old = vec![T::zero(); n];
    let mut p = vec![T::zero(); n];
    for i in 0..n {
        v[i] = y[i].clone() / beta1.clone();
        p[i] = r[i].clone() / beta1.clone();
    }

    // Lanczos scalars
    let mut beta = beta1.clone();

    // Givens rotation parameters (need two previous rotations)
    let mut c = T::one();
    let mut s = T::zero();
    let mut c_old = T::one();
    let mut s_old = T::zero();

    // Direction vectors for solution update
    let mut w = vec![T::zero(); n];
    let mut w_old = vec![T::zero(); n];

    // Right-hand side of transformed system
    let mut eta = beta1.clone();

    for k in 0..max_iter {
        // Preconditioned Lanczos step
        // Compute A * v
        let mut av = vec![T::zero(); n];
        spmv(T::one(), a, &v, T::zero(), &mut av);

        // alpha = v^T * A * v = (M^{-1} r)^T * A * (M^{-1} r) / beta^2
        // But we compute it as: alpha = <p, Av> where p = M*v
        let alpha = dot(&p, &av);

        // Update: av = A*v - alpha*p - beta*p_old
        // This maintains the relation that p holds M*v
        for i in 0..n {
            av[i] = av[i].clone() - alpha.clone() * p[i].clone() - beta.clone() * p_old[i].clone();
        }

        // Apply preconditioner: y = M^{-1} * av
        y = precond(&av);

        // beta_new = sqrt(av^T * y) = sqrt(av^T * M^{-1} * av)
        let avy = dot(&av, &y);
        if avy < T::zero() {
            return Err(IterativeError::Breakdown {
                iteration: k,
                description: "Preconditioner not positive definite".to_string(),
            });
        }
        let beta_new = Real::sqrt(avy);

        // =====================================
        // QR factorization update (same as unpreconditioned)
        // =====================================
        let c_oold = c_old.clone();
        let s_oold = s_old.clone();
        c_old = c.clone();
        s_old = s.clone();

        // Apply previous rotations
        let r1 = c_old.clone() * alpha.clone() - c_oold.clone() * s_old.clone() * beta.clone();
        let r2 = s_old.clone() * alpha.clone() + c_oold.clone() * c_old.clone() * beta.clone();
        let r3 = s_oold.clone() * beta.clone();

        // Compute new rotation
        let rho = Real::sqrt(r1.clone() * r1.clone() + beta_new.clone() * beta_new.clone());

        let c_new: T;
        let s_new: T;
        if Scalar::abs(rho.clone()) <= <T as Scalar>::epsilon() {
            c_new = T::one();
            s_new = T::zero();
        } else {
            c_new = r1.clone() / rho.clone();
            s_new = beta_new.clone() / rho.clone();
        }

        // Update direction vector
        let mut w_new = vec![T::zero(); n];
        if Scalar::abs(rho.clone()) > <T as Scalar>::epsilon() {
            for i in 0..n {
                w_new[i] =
                    (v[i].clone() - r3.clone() * w_old[i].clone() - r2.clone() * w[i].clone())
                        / rho.clone();
            }
        }

        // Update solution
        let eta_bar = c_new.clone() * eta.clone();
        let eta_new = T::zero() - s_new.clone() * eta.clone();

        for i in 0..n {
            x[i] = x[i].clone() + eta_bar.clone() * w_new[i].clone();
        }

        // Compute actual residual for convergence check
        let mut r_curr = vec![T::zero(); n];
        spmv(T::one(), a, &x, T::zero(), &mut r_curr);
        for i in 0..n {
            r_curr[i] = b[i].clone() - r_curr[i].clone();
        }
        let res_norm = norm(&r_curr);
        residual_history.push(res_norm.clone());

        if res_norm <= tol_abs {
            return Ok(MinresResult {
                x,
                iterations: k + 1,
                residual_norm: res_norm,
                converged: true,
                residual_history,
            });
        }

        if Scalar::abs(beta_new.clone()) <= <T as Scalar>::epsilon() {
            return Ok(MinresResult {
                x,
                iterations: k + 1,
                residual_norm: res_norm,
                converged: true,
                residual_history,
            });
        }

        // Prepare for next iteration
        for i in 0..n {
            v_old[i] = v[i].clone();
            v[i] = y[i].clone() / beta_new.clone();
            p_old[i] = p[i].clone();
            p[i] = av[i].clone() / beta_new.clone();
            w_old[i] = w[i].clone();
            w[i] = w_new[i].clone();
        }

        beta = beta_new;
        c = c_new;
        s = s_new;
        eta = eta_new;
    }

    // Final residual
    let mut r_final = vec![T::zero(); n];
    spmv(T::one(), a, &x, T::zero(), &mut r_final);
    for i in 0..n {
        r_final[i] = b[i].clone() - r_final[i].clone();
    }
    let final_residual = norm(&r_final);

    Ok(MinresResult {
        x,
        iterations: max_iter,
        residual_norm: final_residual,
        converged: false,
        residual_history,
    })
}
