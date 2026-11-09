//! MINRES (Minimum Residual) and Preconditioned MINRES solvers.

use super::helpers::{breakdown, converged_true_residual, dot, negligible_norm, norm};
use super::scaling::{Operator, PrecondScaling, SystemScaling};
use super::types::{IterativeError, MinresResult};
use crate::csr::CsrMatrix;
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Maps a result of the rescaled system back to the original one.
fn unscale_minres<T: Scalar<Real = T> + Clone + Field + Real>(
    scaling: &SystemScaling<T>,
    mut result: MinresResult<T>,
) -> MinresResult<T> {
    scaling.solution(&mut result.x);
    result.residual_norm = scaling.residual(result.residual_norm);
    scaling.residuals(&mut result.residual_history);
    result
}

/// MINRES for symmetric (possibly indefinite) `A`.
///
/// Converges when the recursive residual estimate `|eta|` satisfies
/// `<= tol·‖b‖`. Norms (including the Lanczos column norms and the rotated
/// diagonal) are overflow-safe; see the [module documentation](super) for
/// systems far from unit scale.
///
/// # Breakdown
///
/// The entries of each Lanczos column `(beta_k, alpha_k, beta_{k+1})` are
/// judged against that column's norm `‖A v_k‖` (scale-invariant):
/// * `beta_{k+1}` negligible: the Krylov subspace is invariant and the
///   recurrence cannot continue;
/// * the rotated diagonal `rho` negligible: the projected tridiagonal matrix is
///   singular on an invariant subspace.
///
/// Either way the true residual `‖b − A x‖` decides: a converged result if it
/// meets `tol·‖b‖`, [`IterativeError::Breakdown`] otherwise.
pub fn minres<T: Scalar<Real = T> + Clone + Field + Real>(
    a: &CsrMatrix<T>,
    b: &[T],
    x0: &[T],
    tol: T,
    max_iter: usize,
) -> Result<MinresResult<T>, IterativeError> {
    let scaling = SystemScaling::for_rhs(a, b);
    let result = minres_scaled(
        &scaling.operator(a),
        &scaling.rhs(b),
        &scaling.initial_guess(x0),
        tol,
        max_iter,
    )?;
    Ok(unscale_minres(&scaling, result))
}

/// [`minres`] on the (rescaled) system `a x = b`.
fn minres_scaled<T: Scalar<Real = T> + Clone + Field + Real>(
    a: &Operator<'_, T>,
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
    a.apply(&x, &mut r);
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

    // Lanczos scalars. `beta` is the off-diagonal entry coupling the current
    // Lanczos vector to the previous one; there is none before the first
    // column (beta1 only enters through the right-hand side `eta`).
    let mut beta = T::zero();

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
        a.apply(&v, &mut v_new);

        let alpha = dot(&v, &v_new);

        // v_new = v_new - alpha*v - beta*v_old
        for i in 0..n {
            v_new[i] =
                v_new[i].clone() - alpha.clone() * v[i].clone() - beta.clone() * v_old[i].clone();
        }

        let beta_new = norm(&v_new);

        // ||A v|| = ||(beta, alpha, beta_new)|| in exact arithmetic: the scale
        // against which the entries of this Lanczos column are judged
        // (overflow-safe, summed as alpha² + beta² + beta_new²).
        let column_norm = norm(&[alpha.clone(), beta.clone(), beta_new.clone()]);

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
        // (rho = ||(r1, beta_new)||, overflow-safe)
        let rho = norm(&[r1.clone(), beta_new.clone()]);

        // rho negligible implies beta_new negligible as well: the projected
        // tridiagonal matrix is singular on an invariant Krylov subspace.
        if negligible_norm(rho.clone(), column_norm.clone()) {
            if let Some(residual_norm) = converged_true_residual(a, b, &x, tol_abs.clone()) {
                return Ok(MinresResult {
                    x,
                    iterations: k,
                    residual_norm,
                    converged: true,
                    residual_history,
                });
            }
            return Err(breakdown(
                k,
                "rotated diagonal negligible relative to ||A v|| (singular projected system)",
            ));
        }

        let c_new = r1.clone() / rho.clone();
        let s_new = beta_new.clone() / rho.clone();

        // Update direction vector: w_new = (v - r3*w_oold - r2*w_old) / rho
        // Here w is w_old from prev iteration, w_old is w_oold from prev iteration
        let mut w_new = vec![T::zero(); n];
        for i in 0..n {
            w_new[i] = (v[i].clone() - r3.clone() * w_old[i].clone() - r2.clone() * w[i].clone())
                / rho.clone();
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

        // Lanczos breakdown: the Krylov subspace is (numerically) invariant
        if negligible_norm(beta_new.clone(), column_norm) {
            if let Some(residual_norm) = converged_true_residual(a, b, &x, tol_abs.clone()) {
                return Ok(MinresResult {
                    x,
                    iterations: k + 1,
                    residual_norm,
                    converged: true,
                    residual_history,
                });
            }
            return Err(breakdown(
                k,
                "Lanczos breakdown: beta negligible relative to ||A v|| (invariant subspace)",
            ));
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
    a.apply(&x, &mut r_final);
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

/// Preconditioned MINRES; `precond` applies `M^{-1}` for a symmetric positive
/// definite `M`.
///
/// Runs the Lanczos process in the `M^{-1}` inner product (`v = M^{-1} p`,
/// `alpha = vᵀ A v`) and converges when the true residual satisfies
/// `‖b − A x‖ <= tol·‖b‖`. Norms are overflow-safe; for a system far from
/// unit scale `precond` receives vectors of the rescaled system and must be
/// linear (see the [module documentation](super)).
///
/// # Breakdown
///
/// Same scale-invariant rules as [`minres`]; additionally a negative
/// `M^{-1}`-norm reveals a preconditioner that is not positive definite.
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
    let scaling = SystemScaling::for_rhs(a, b);
    let precond_scaling = PrecondScaling::new();
    let result = pminres_scaled(
        &scaling.operator(a),
        &scaling.rhs(b),
        &scaling.initial_guess(x0),
        |r: &[T]| precond_scaling.finish(r, precond(r)),
        tol,
        max_iter,
    )?;
    Ok(unscale_minres(&scaling, result))
}

/// [`pminres`] on the (rescaled) system `a x = b`.
fn pminres_scaled<T, F>(
    a: &Operator<'_, T>,
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
    a.apply(&x, &mut r);
    for i in 0..n {
        r[i] = b[i].clone() - r[i].clone();
    }

    let b_norm = norm(b);

    // Immediate-convergence / zero-initial-residual check.
    //
    // If x0 is already the exact solution then the initial residual r = b - A*x0
    // is zero. In that degenerate-but-valid case the preconditioned quantity
    // r^T M^{-1} r is exactly zero, and reporting it as "preconditioner not
    // positive definite" would be a false positive. Return the initial guess as
    // converged BEFORE any preconditioner positive-definiteness test runs.
    let r_norm = norm(&r);
    residual_history.push(r_norm.clone());
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

    // y = M^{-1} * r (preconditioned residual)
    let mut y = precond(&r);

    // beta1 = sqrt(r^T * y) = sqrt(r^T * M^{-1} * r) (M-norm of r).
    // r is nonzero here (immediate convergence is handled above), so for a
    // genuine SPD preconditioner ry is strictly positive; ry <= 0 is a real
    // breakdown (the preconditioner is not positive definite w.r.t. r).
    let ry = dot(&r, &y);
    // Sign test (scale-invariant). The true residual of x0 is already known to
    // exceed the tolerance, so this is a genuine breakdown.
    if ry <= T::zero() {
        return Err(breakdown(0, "Preconditioner not positive definite"));
    }
    let beta1 = Real::sqrt(ry);

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

    // Lanczos scalars. `beta` couples the current Lanczos vector to the
    // previous one; there is none before the first column (beta1 only enters
    // through the right-hand side `eta`).
    let mut beta = T::zero();

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
        a.apply(&v, &mut av);

        // alpha = v^T * A * v, the Lanczos coefficient of M^{-1} A in the
        // M-inner product (v = M^{-1} p). Note <p, A v> = v^T M A v would be
        // wrong for any M other than the identity.
        let alpha = dot(&v, &av);

        // Update: av = A*v - alpha*p - beta*p_old
        // This maintains the relation that p holds M*v
        for i in 0..n {
            av[i] = av[i].clone() - alpha.clone() * p[i].clone() - beta.clone() * p_old[i].clone();
        }

        // Apply preconditioner: y = M^{-1} * av
        y = precond(&av);

        // beta_new = sqrt(av^T * y) = sqrt(av^T * M^{-1} * av)
        let avy = dot(&av, &y);
        // Sign test (scale-invariant); the true residual of the current x was
        // checked at the end of the previous iteration and exceeds the tolerance.
        if avy < T::zero() {
            return Err(breakdown(k, "Preconditioner not positive definite"));
        }
        let beta_new = Real::sqrt(avy);

        // ||A v||_{M^{-1}} = ||(beta, alpha, beta_new)|| in exact arithmetic:
        // the scale against which the entries of this Lanczos column are judged
        // (overflow-safe, summed as alpha² + beta² + beta_new²).
        let column_norm = norm(&[alpha.clone(), beta.clone(), beta_new.clone()]);

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

        // Compute new rotation (rho = ||(r1, beta_new)||, overflow-safe)
        let rho = norm(&[r1.clone(), beta_new.clone()]);

        // rho negligible implies beta_new negligible as well: the projected
        // tridiagonal matrix is singular on an invariant Krylov subspace.
        if negligible_norm(rho.clone(), column_norm.clone()) {
            if let Some(residual_norm) = converged_true_residual(a, b, &x, tol_abs.clone()) {
                return Ok(MinresResult {
                    x,
                    iterations: k,
                    residual_norm,
                    converged: true,
                    residual_history,
                });
            }
            return Err(breakdown(
                k,
                "rotated diagonal negligible relative to ||A v|| (singular projected system)",
            ));
        }

        let c_new = r1.clone() / rho.clone();
        let s_new = beta_new.clone() / rho.clone();

        // Update direction vector
        let mut w_new = vec![T::zero(); n];
        for i in 0..n {
            w_new[i] = (v[i].clone() - r3.clone() * w_old[i].clone() - r2.clone() * w[i].clone())
                / rho.clone();
        }

        // Update solution
        let eta_bar = c_new.clone() * eta.clone();
        let eta_new = T::zero() - s_new.clone() * eta.clone();

        for i in 0..n {
            x[i] = x[i].clone() + eta_bar.clone() * w_new[i].clone();
        }

        // Compute actual residual for convergence check
        let mut r_curr = vec![T::zero(); n];
        a.apply(&x, &mut r_curr);
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

        // Lanczos breakdown: the Krylov subspace is (numerically) invariant.
        // `res_norm` is the true residual and was just found to exceed the
        // tolerance, so the breakdown is genuine.
        if negligible_norm(beta_new.clone(), column_norm) {
            return Err(breakdown(
                k,
                "Lanczos breakdown: beta negligible relative to ||A v|| (invariant subspace)",
            ));
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
    a.apply(&x, &mut r_final);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr::CsrMatrix;

    fn spd_tridiag() -> CsrMatrix<f64> {
        // A = [4 1 0; 1 4 1; 0 1 4] (symmetric positive definite)
        let values = vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let row_ptrs = vec![0, 2, 5, 7];
        CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap()
    }

    /// Regression test: a zero initial residual (x0 already the exact solution)
    /// must NOT be reported as "preconditioner not positive definite". pminres
    /// should return the initial guess immediately as converged.
    #[test]
    fn test_pminres_zero_initial_residual_no_false_breakdown() {
        let a = spd_tridiag();
        // x_exact = [1, 1, 1]  =>  b = A * x_exact = [5, 6, 5]
        let b = vec![5.0, 6.0, 5.0];
        let x0 = vec![1.0, 1.0, 1.0]; // already the exact solution => r0 = 0
        let precond = |r: &[f64]| -> Vec<f64> { r.iter().map(|&v| v / 4.0).collect() };

        let result = pminres(&a, &b, &x0, precond, 1e-10, 100)
            .expect("zero initial residual must not trigger a spurious breakdown");

        assert!(result.converged, "should be reported converged");
        assert_eq!(result.iterations, 0, "no iterations needed");
        for i in 0..3 {
            assert!(
                (result.x[i] - 1.0).abs() < 1e-12,
                "initial guess should be returned unchanged at index {i}"
            );
        }
    }

    /// A genuinely non-positive-definite preconditioner on a nonzero residual
    /// must still be reported as a breakdown (the fix must not mask real errors).
    #[test]
    fn test_pminres_indefinite_precond_still_breaks_down() {
        let a = spd_tridiag();
        let b = vec![5.0, 6.0, 5.0];
        let x0 = vec![0.0, 0.0, 0.0]; // nonzero residual
        // Negative-definite "preconditioner": r^T M^{-1} r < 0.
        let precond = |r: &[f64]| -> Vec<f64> { r.iter().map(|&v| -v).collect() };

        let result = pminres(&a, &b, &x0, precond, 1e-10, 100);
        assert!(matches!(result, Err(IterativeError::Breakdown { .. })));
    }
}
