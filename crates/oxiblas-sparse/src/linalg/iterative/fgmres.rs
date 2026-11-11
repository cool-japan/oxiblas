//! FGMRES (Flexible GMRES) and FGMRES-IR solvers.

use super::gmres::gmres;
use super::helpers::{
    converged_true_residual, dot, givens_rotation_gmres, negligible_norm, norm,
    solve_upper_triangular,
};
use super::scaling::{Operator, PrecondScaling, SystemScaling};
use super::types::{FgmresResult, IterativeError};
use crate::csr::CsrMatrix;
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Flexible GMRES: the (possibly variable or stateful) preconditioner may
/// change from one Arnoldi step to the next; the preconditioned vectors are
/// stored explicitly for the solution update.
///
/// Converges when the least-squares residual estimate satisfies
/// `<= tol·‖b‖` and the true residual `‖b − A x‖` confirms it (the reported
/// `residual_norm` is then the true residual); otherwise the method restarts
/// from the true residual, as [`super::gmres::gmres`] does. Norms are
/// overflow-safe; for a system far from unit scale `precond` receives
/// vectors of the rescaled system and must be homogeneous
/// (`M_j⁻¹(c·v) = c·M_j⁻¹ v`; see the [module documentation](super)).
///
/// # Breakdown
///
/// Same lucky-breakdown rule as [`super::gmres::gmres`]: the sub-diagonal entry
/// `h_{j+1,j}` negligible relative to `‖A z_j‖` triggers a true-residual check
/// (converged, or restart from the true residual).
pub fn fgmres<T, F>(
    a: &CsrMatrix<T>,
    b: &[T],
    x0: &[T],
    precond: &mut F,
    restart: usize,
    tol: T,
    max_iter: usize,
) -> Result<FgmresResult<T>, IterativeError>
where
    T: Scalar<Real = T> + Clone + Field + Real,
    F: FnMut(&[T]) -> Vec<T>,
{
    let scaling = SystemScaling::for_rhs(a, b);
    let precond_scaling = PrecondScaling::new();
    let mut scaled_precond = |v: &[T]| precond_scaling.finish(v, precond(v));
    let mut result = fgmres_scaled(
        &scaling.operator(a),
        &scaling.rhs(b),
        &scaling.initial_guess(x0),
        &mut scaled_precond,
        restart,
        tol,
        max_iter,
    )?;
    scaling.solution(&mut result.x);
    result.residual_norm = scaling.residual(result.residual_norm);
    scaling.residuals(&mut result.residual_history);
    Ok(result)
}

/// [`fgmres`] on the (rescaled) system `a x = b`.
fn fgmres_scaled<T, F>(
    a: &Operator<'_, T>,
    b: &[T],
    x0: &[T],
    precond: &mut F,
    restart: usize,
    tol: T,
    max_iter: usize,
) -> Result<FgmresResult<T>, IterativeError>
where
    T: Scalar<Real = T> + Clone + Field + Real,
    F: FnMut(&[T]) -> Vec<T>,
{
    let n = a.nrows();

    if b.len() != n || x0.len() != n {
        return Err(IterativeError::DimensionMismatch {
            expected: n,
            actual: b.len().min(x0.len()),
        });
    }

    let restart = restart.min(n).max(1);
    let mut x = x0.to_vec();
    let mut residual_history = Vec::new();
    let mut total_iter = 0;
    let mut restarts = 0;

    let b_norm = norm(b);
    let tol_abs = tol.clone() * b_norm.clone();

    // Outer loop (restarts)
    while total_iter < max_iter {
        // Compute initial residual r = b - A*x
        let mut r = vec![T::zero(); n];
        a.apply(&x, &mut r);
        for i in 0..n {
            r[i] = b[i].clone() - r[i].clone();
        }

        let beta = norm(&r);
        residual_history.push(beta.clone());

        if beta <= tol_abs {
            return Ok(FgmresResult {
                x,
                iterations: total_iter,
                restarts,
                residual_norm: beta,
                converged: true,
                residual_history,
            });
        }

        // Arnoldi process with variable preconditioning
        // V: orthonormal basis vectors (n x (restart+1))
        // Z: preconditioned vectors (n x restart) - stored explicitly for variable precond
        // H: upper Hessenberg matrix ((restart+1) x restart)
        let mut v: Vec<Vec<T>> = Vec::with_capacity(restart + 1);
        let mut z: Vec<Vec<T>> = Vec::with_capacity(restart);

        // v[0] = r / beta
        let mut v0 = vec![T::zero(); n];
        for i in 0..n {
            v0[i] = r[i].clone() / beta.clone();
        }
        v.push(v0);

        // Upper Hessenberg matrix stored column by column
        let mut h: Vec<Vec<T>> = Vec::with_capacity(restart);

        // Givens rotation parameters
        let mut cs: Vec<T> = Vec::with_capacity(restart);
        let mut sn: Vec<T> = Vec::with_capacity(restart);

        // Right-hand side for least squares (starts as [beta, 0, 0, ...])
        let mut g = vec![T::zero(); restart + 1];
        g[0] = beta.clone();

        // Set when a lucky breakdown has already folded the cycle into x.
        let mut solution_updated = false;

        // Inner loop (Arnoldi iteration with variable preconditioning)
        for j in 0..restart {
            if total_iter >= max_iter {
                break;
            }

            // z[j] = M_j^{-1} * v[j] - variable preconditioner applied here
            let z_j = precond(&v[j]);
            z.push(z_j.clone());

            // w = A * z[j]
            let mut w = vec![T::zero(); n];
            a.apply(&z_j, &mut w);
            // ||A z_j||: the norm the sub-diagonal entry is derived from
            let w_norm = norm(&w);

            // Modified Gram-Schmidt orthogonalization
            let mut h_col = vec![T::zero(); j + 2];
            for i in 0..=j {
                h_col[i] = dot(&v[i], &w);
                for k in 0..n {
                    w[k] = w[k].clone() - h_col[i].clone() * v[i][k].clone();
                }
            }
            // h_{j+1,j}, kept before the Givens rotation overwrites h_col[j + 1]
            let h_sub = norm(&w);
            h_col[j + 1] = h_sub.clone();

            // Apply previous Givens rotations to new column
            for i in 0..j {
                let temp = cs[i].clone() * h_col[i].clone() + sn[i].clone() * h_col[i + 1].clone();
                h_col[i + 1] = T::zero() - sn[i].clone() * h_col[i].clone()
                    + cs[i].clone() * h_col[i + 1].clone();
                h_col[i] = temp;
            }

            // Compute new Givens rotation
            let (c, s, r_val) = givens_rotation_gmres(h_col[j].clone(), h_col[j + 1].clone());
            cs.push(c.clone());
            sn.push(s.clone());

            // Apply new rotation to h_col and g
            h_col[j] = r_val;
            h_col[j + 1] = T::zero();

            let temp = c.clone() * g[j].clone() + s.clone() * g[j + 1].clone();
            g[j + 1] = T::zero() - s.clone() * g[j].clone() + c.clone() * g[j + 1].clone();
            g[j] = temp;

            h.push(h_col);

            // Check convergence
            let res_norm = Scalar::abs(g[j + 1].clone());
            residual_history.push(res_norm.clone());
            total_iter += 1;

            if res_norm <= tol_abs {
                // Converged estimate - solve upper triangular system H*y = g
                let y = solve_upper_triangular(&h, &g, j + 1);

                // Update solution: x = x + Z*y (using preconditioned vectors)
                for i in 0..=j {
                    for k in 0..n {
                        x[k] = x[k].clone() + y[i].clone() * z[i][k].clone();
                    }
                }

                // Confirm with the true residual before reporting: the estimate
                // |g_{j+1}| is exactly 0 when A v_j = 0 or h_{j+1,j} = 0 (the
                // rotation then has s = 0) even if the triangular solve had to
                // drop a singular direction, and in finite precision it drifts
                // away from ||b - A x||.
                if let Some(residual_norm) = converged_true_residual(a, b, &x, tol_abs.clone()) {
                    return Ok(FgmresResult {
                        x,
                        iterations: total_iter,
                        restarts,
                        residual_norm,
                        converged: true,
                        residual_history,
                    });
                }
                // Not converged: restart from the true residual.
                solution_updated = true;
                break;
            }

            // Lucky breakdown: the Krylov subspace is (numerically) invariant
            if negligible_norm(h_sub.clone(), w_norm) {
                let y = solve_upper_triangular(&h, &g, j + 1);
                for i in 0..=j {
                    for k in 0..n {
                        x[k] = x[k].clone() + y[i].clone() * z[i][k].clone();
                    }
                }
                if let Some(residual_norm) = converged_true_residual(a, b, &x, tol_abs.clone()) {
                    return Ok(FgmresResult {
                        x,
                        iterations: total_iter,
                        restarts,
                        residual_norm,
                        converged: true,
                        residual_history,
                    });
                }
                // Not converged: restart from the true residual.
                solution_updated = true;
                break;
            }

            // Normalize w to get v[j+1]
            let mut v_new = vec![T::zero(); n];
            for k in 0..n {
                v_new[k] = w[k].clone() / h_sub.clone();
            }
            v.push(v_new);
        }

        if !solution_updated && !h.is_empty() {
            // Solve upper triangular system before restart
            let m = h.len();
            let y = solve_upper_triangular(&h, &g, m);

            // Update solution: x = x + Z*y
            for i in 0..m {
                if i < z.len() {
                    for k in 0..n {
                        x[k] = x[k].clone() + y[i].clone() * z[i][k].clone();
                    }
                }
            }
        }

        restarts += 1;
    }

    // Final residual computation
    let mut r = vec![T::zero(); n];
    a.apply(&x, &mut r);
    for i in 0..n {
        r[i] = b[i].clone() - r[i].clone();
    }
    let final_residual = norm(&r);

    Ok(FgmresResult {
        x,
        iterations: total_iter,
        restarts,
        residual_norm: final_residual,
        converged: false,
        residual_history,
    })
}

/// FGMRES with an inner GMRES solve as the (variable) preconditioner.
///
/// If an inner solve fails, the preconditioner falls back to the identity.
pub fn fgmres_ir<T: Scalar<Real = T> + Clone + Field + Real>(
    a: &CsrMatrix<T>,
    b: &[T],
    x0: &[T],
    inner_restart: usize,
    inner_tol: T,
    inner_max_iter: usize,
    outer_restart: usize,
    outer_tol: T,
    outer_max_iter: usize,
) -> Result<FgmresResult<T>, IterativeError> {
    // Inner GMRES preconditioner: approximately solve A*z = r
    let mut inner_precond = |r: &[T]| -> Vec<T> {
        let zero = vec![T::zero(); r.len()];
        match gmres(
            a,
            r,
            &zero,
            inner_restart,
            inner_tol.clone(),
            inner_max_iter,
        ) {
            Ok(result) => result.x,
            Err(_) => r.to_vec(), // Fallback to identity if inner GMRES fails
        }
    };

    fgmres(
        a,
        b,
        x0,
        &mut inner_precond,
        outer_restart,
        outer_tol,
        outer_max_iter,
    )
}
