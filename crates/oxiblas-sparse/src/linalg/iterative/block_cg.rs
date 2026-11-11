//! Block Conjugate Gradient solvers for multiple right-hand sides.
//!
//! Each right-hand side is advanced by its own CG recurrence. A system is
//! *settled* once it has converged (`‖r_k‖ <= tol·‖b_k‖`) or once its recurrence
//! has broken down (`|p_kᵀ A p_k| <= eps·‖p_k‖·‖A p_k‖`; for [`block_pcg`]
//! instead the rule of [`pcg`](super::cg::pcg) for `r_kᵀ z_k`,
//! `z_k = M⁻¹ r_k`, and `p_kᵀ A p_k`: a negligible or NaN value, or a sign
//! that differs from its sign at the system's first iteration, which catches
//! singular or indefinite preconditioners and operators and NaN output; all
//! tests are scale-invariant). A broken-down system counts as converged only
//! when its true residual `‖b_k − A x_k‖` meets the tolerance; otherwise it
//! stops iterating, is excluded from [`BlockCgResult::num_converged`] and
//! makes [`BlockCgResult::converged`] false, while the remaining systems
//! continue.
//!
//! Norms are overflow-safe, and every system is rescaled independently when
//! it is far from unit scale (see the [module documentation](super)).

use super::helpers::{
    SumOfSquares, dot_and_sums_of_squares, negligible_inner, norm, sign_breakdown, sum_of_squares,
    true_residual_norm,
};
use super::scaling::{Operator, PrecondScaling, SystemScaling};
use super::types::{BlockCgResult, IterativeError};
use crate::csr::CsrMatrix;
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Per-system bookkeeping shared by [`block_cg`] and [`block_pcg`].
struct SystemStatus<T> {
    /// System met the tolerance (recursive residual, or true residual after a
    /// breakdown).
    converged: Vec<bool>,
    /// System broke down without meeting the tolerance; it no longer iterates.
    stopped: Vec<bool>,
    /// True residual norm recorded when a system was settled by a breakdown.
    breakdown_residual: Vec<Option<T>>,
}

impl<T: Scalar<Real = T> + Clone + Field + Real> SystemStatus<T> {
    fn new(num_rhs: usize) -> Self {
        Self {
            converged: vec![false; num_rhs],
            stopped: vec![false; num_rhs],
            breakdown_residual: vec![None; num_rhs],
        }
    }

    fn active(&self, k: usize) -> bool {
        !self.converged[k] && !self.stopped[k]
    }

    /// Settles system `k` after a breakdown, using its true residual.
    fn settle_breakdown(&mut self, k: usize, true_residual: T, tol_abs: &T) {
        if true_residual <= *tol_abs {
            self.converged[k] = true;
        } else {
            self.stopped[k] = true;
        }
        self.breakdown_residual[k] = Some(true_residual);
    }

    /// Builds the result; `recursive_norms[k]` is used unless system `k` was
    /// settled by a breakdown, in which case its true residual is reported.
    fn finish(
        self,
        x: Vec<Vec<T>>,
        iterations: usize,
        recursive_norms: Vec<T>,
    ) -> BlockCgResult<T> {
        let residual_norms: Vec<T> = recursive_norms
            .into_iter()
            .zip(self.breakdown_residual)
            .map(|(recursive, recorded)| recorded.unwrap_or(recursive))
            .collect();
        let num_converged = self.converged.iter().filter(|&&c| c).count();
        BlockCgResult {
            x,
            iterations,
            residual_norms,
            converged: num_converged == self.converged.len(),
            num_converged,
        }
    }
}

/// The rescaled systems of a block solve: one [`SystemScaling`] per
/// right-hand side, all with the factor of `A`.
struct BlockScaling<T> {
    base: SystemScaling<T>,
    systems: Vec<SystemScaling<T>>,
    b: Vec<Vec<T>>,
    x0: Vec<Vec<T>>,
}

impl<T: Scalar<Real = T> + Clone + Field + Real> BlockScaling<T> {
    fn new(a: &CsrMatrix<T>, b: &[Vec<T>], x0: &[Vec<T>]) -> Self {
        let base = SystemScaling::new(a, T::zero());
        let systems: Vec<SystemScaling<T>> = b.iter().map(|bk| base.with_rhs(bk)).collect();
        let b_hat = b
            .iter()
            .zip(&systems)
            .map(|(bk, s)| s.rhs(bk).into_owned())
            .collect();
        let x0_hat = x0
            .iter()
            .zip(&systems)
            .map(|(xk, s)| s.initial_guess(xk).into_owned())
            .collect();
        Self {
            base,
            systems,
            b: b_hat,
            x0: x0_hat,
        }
    }

    /// Maps the result of the rescaled systems back.
    fn finish(&self, mut result: BlockCgResult<T>) -> BlockCgResult<T> {
        for ((x, residual), s) in result
            .x
            .iter_mut()
            .zip(result.residual_norms.iter_mut())
            .zip(&self.systems)
        {
            s.solution(x);
            *residual = s.residual(*residual);
        }
        result
    }
}

/// Block Conjugate Gradient: solves `A x_k = b_k` for every right-hand side.
///
/// `x0` may be shorter than `b`; missing initial guesses are zero.
///
/// # Breakdown
///
/// See the module documentation: a broken-down system is reported as converged
/// only if its true residual meets `tol·‖b_k‖`.
pub fn block_cg<T: Scalar<Real = T> + Clone + Field + Real>(
    a: &CsrMatrix<T>,
    b: &[Vec<T>],
    x0: &[Vec<T>],
    tol: T,
    max_iter: usize,
) -> Result<BlockCgResult<T>, IterativeError> {
    let scaling = BlockScaling::new(a, b, x0);
    let result = block_cg_scaled(
        &scaling.base.operator(a),
        &scaling.b,
        &scaling.x0,
        tol,
        max_iter,
    )?;
    Ok(scaling.finish(result))
}

/// [`block_cg`] on the (rescaled) systems `a x_k = b_k`.
fn block_cg_scaled<T: Scalar<Real = T> + Clone + Field + Real>(
    a: &Operator<'_, T>,
    b: &[Vec<T>],
    x0: &[Vec<T>],
    tol: T,
    max_iter: usize,
) -> Result<BlockCgResult<T>, IterativeError> {
    let n = a.nrows();
    let num_rhs = b.len();

    if num_rhs == 0 {
        return Ok(BlockCgResult {
            x: vec![],
            iterations: 0,
            residual_norms: vec![],
            converged: true,
            num_converged: 0,
        });
    }

    // Validate dimensions
    for (i, bi) in b.iter().enumerate() {
        if bi.len() != n {
            return Err(IterativeError::DimensionMismatch {
                expected: n,
                actual: bi.len(),
            });
        }
        if i < x0.len() && x0[i].len() != n {
            return Err(IterativeError::DimensionMismatch {
                expected: n,
                actual: x0[i].len(),
            });
        }
    }

    // Initialize solution vectors
    let mut x: Vec<Vec<T>> = x0.to_vec();
    while x.len() < num_rhs {
        x.push(vec![T::zero(); n]);
    }

    // Initialize residual vectors: R = B - A*X
    let mut r: Vec<Vec<T>> = Vec::with_capacity(num_rhs);
    for k in 0..num_rhs {
        let mut rk = vec![T::zero(); n];
        a.residual(&b[k], &x[k], &mut rk);
        r.push(rk);
    }

    // Initialize search directions: P = R
    let mut p: Vec<Vec<T>> = r.clone();

    // Compute initial residual norms and tolerance
    let mut tol_abs: Vec<T> = Vec::with_capacity(num_rhs);
    for k in 0..num_rhs {
        let bn = norm(&b[k]);
        tol_abs.push(tol.clone() * bn);
    }

    // Initial r^T * r of each system, kept overflow-safe (its value is the
    // recurrence coefficient, its norm the convergence measure)
    let mut rr: Vec<SumOfSquares<T>> = r
        .iter()
        .map(|rk| sum_of_squares(rk.iter().copied()))
        .collect();

    let mut status = SystemStatus::new(num_rhs);

    // Working vector for A * p
    let mut ap = vec![T::zero(); n];

    for iter in 0..max_iter {
        // Check convergence of every active system
        let mut all_settled = true;
        for k in 0..num_rhs {
            if status.active(k) {
                let r_norm = rr[k].norm();
                if r_norm <= tol_abs[k] {
                    status.converged[k] = true;
                } else {
                    all_settled = false;
                }
            }
        }

        if all_settled {
            let norms: Vec<T> = rr.iter().map(|rr_k| rr_k.norm()).collect();
            return Ok(status.finish(x, iter, norms));
        }

        // Advance each active system
        for k in 0..num_rhs {
            if !status.active(k) {
                continue;
            }

            // ap = A * p[k]
            a.apply(&p[k], &mut ap);

            // alpha = rr[k] / (p[k]^T * ap)
            let (pap, pp, apap) = dot_and_sums_of_squares(&p[k], &ap);

            if negligible_inner(pap.clone(), pp.norm(), apap.norm()) {
                let true_residual = true_residual_norm(a, &b[k], &x[k]);
                status.settle_breakdown(k, true_residual, &tol_abs[k]);
                continue;
            }

            let rr_k = rr[k].value();
            let alpha = rr_k.clone() / pap;

            // x[k] = x[k] + alpha * p[k]
            for i in 0..n {
                x[k][i] = x[k][i].clone() + alpha.clone() * p[k][i].clone();
            }

            // r[k] = r[k] - alpha * ap
            for i in 0..n {
                r[k][i] = r[k][i].clone() - alpha.clone() * ap[i].clone();
            }

            // rr_new = r[k]^T * r[k]
            let rr_new = sum_of_squares(r[k].iter().copied());

            // beta = rr_new / rr[k]
            let beta = rr_new.value() / rr_k;

            // p[k] = r[k] + beta * p[k]
            for i in 0..n {
                p[k][i] = r[k][i].clone() + beta.clone() * p[k][i].clone();
            }

            rr[k] = rr_new;
        }
    }

    let norms: Vec<T> = rr.iter().map(|rr_k| rr_k.norm()).collect();
    Ok(status.finish(x, max_iter, norms))
}

/// Block Preconditioned Conjugate Gradient; `precond` applies `M^{-1}`.
///
/// `x0` may be shorter than `b`; missing initial guesses are zero.
///
/// # Breakdown
///
/// See the module documentation: a broken-down system is reported as converged
/// only if its true residual meets `tol·‖b_k‖`. Each system follows the rule
/// of [`pcg`](super::cg::pcg): its `r_kᵀ z_k` (`z_k = M⁻¹ r_k`) and
/// `p_kᵀ A p_k` must not be negligible (or NaN) and must keep the signs they
/// had at the system's first iteration, which would otherwise make its next
/// `β` infinite or NaN, or reveal an indefinite preconditioner or operator.
pub fn block_pcg<T, F>(
    a: &CsrMatrix<T>,
    b: &[Vec<T>],
    x0: &[Vec<T>],
    precond: F,
    tol: T,
    max_iter: usize,
) -> Result<BlockCgResult<T>, IterativeError>
where
    T: Scalar<Real = T> + Clone + Field + Real,
    F: Fn(&[T]) -> Vec<T>,
{
    let scaling = BlockScaling::new(a, b, x0);
    let precond_scaling = PrecondScaling::new();
    let result = block_pcg_scaled(
        &scaling.base.operator(a),
        &scaling.b,
        &scaling.x0,
        |r: &[T]| precond_scaling.finish(r, precond(r)),
        tol,
        max_iter,
    )?;
    Ok(scaling.finish(result))
}

/// [`block_pcg`] on the (rescaled) systems `a x_k = b_k`.
fn block_pcg_scaled<T, F>(
    a: &Operator<'_, T>,
    b: &[Vec<T>],
    x0: &[Vec<T>],
    precond: F,
    tol: T,
    max_iter: usize,
) -> Result<BlockCgResult<T>, IterativeError>
where
    T: Scalar<Real = T> + Clone + Field + Real,
    F: Fn(&[T]) -> Vec<T>,
{
    let n = a.nrows();
    let num_rhs = b.len();

    if num_rhs == 0 {
        return Ok(BlockCgResult {
            x: vec![],
            iterations: 0,
            residual_norms: vec![],
            converged: true,
            num_converged: 0,
        });
    }

    // Validate dimensions
    for (i, bi) in b.iter().enumerate() {
        if bi.len() != n {
            return Err(IterativeError::DimensionMismatch {
                expected: n,
                actual: bi.len(),
            });
        }
        if i < x0.len() && x0[i].len() != n {
            return Err(IterativeError::DimensionMismatch {
                expected: n,
                actual: x0[i].len(),
            });
        }
    }

    // Initialize solution vectors
    let mut x: Vec<Vec<T>> = x0.to_vec();
    while x.len() < num_rhs {
        x.push(vec![T::zero(); n]);
    }

    // Initialize residual vectors: R = B - A*X
    let mut r: Vec<Vec<T>> = Vec::with_capacity(num_rhs);
    for k in 0..num_rhs {
        let mut rk = vec![T::zero(); n];
        a.residual(&b[k], &x[k], &mut rk);
        r.push(rk);
    }

    // Initialize preconditioned residuals: Z = M^{-1} * R
    let mut z: Vec<Vec<T>> = Vec::with_capacity(num_rhs);
    for k in 0..num_rhs {
        z.push(precond(&r[k]));
    }

    // Initialize search directions: P = Z
    let mut p: Vec<Vec<T>> = z.clone();

    // Compute initial residual norms and tolerance
    let mut tol_abs: Vec<T> = Vec::with_capacity(num_rhs);
    for k in 0..num_rhs {
        let bn = norm(&b[k]);
        tol_abs.push(tol.clone() * bn);
    }

    // Initial r^T * z of each system, with ||r|| and ||z|| for the r^T z
    // breakdown test.
    let mut rz: Vec<T> = Vec::with_capacity(num_rhs);
    let mut rr: Vec<SumOfSquares<T>> = Vec::with_capacity(num_rhs);
    let mut zz: Vec<SumOfSquares<T>> = Vec::with_capacity(num_rhs);
    for k in 0..num_rhs {
        let (rz_k, rr_k, zz_k) = dot_and_sums_of_squares(&r[k], &z[k]);
        rz.push(rz_k);
        rr.push(rr_k);
        zz.push(zz_k);
    }

    // Signs of r_k^T z_k and p_k^T A p_k at each system's first checks.
    let mut rz_sign: Vec<Option<bool>> = vec![None; num_rhs];
    let mut pap_sign: Vec<Option<bool>> = vec![None; num_rhs];

    let mut status = SystemStatus::new(num_rhs);

    // Working vector for A * p
    let mut ap = vec![T::zero(); n];

    for iter in 0..max_iter {
        // Check convergence of every active system
        let mut all_settled = true;
        for k in 0..num_rhs {
            if status.active(k) {
                let r_norm = rr[k].norm();
                if r_norm <= tol_abs[k] {
                    status.converged[k] = true;
                } else {
                    all_settled = false;
                }
            }
        }

        if all_settled {
            let norms: Vec<T> = r.iter().map(|rk| norm(rk)).collect();
            return Ok(status.finish(x, iter, norms));
        }

        // Advance each active system
        for k in 0..num_rhs {
            if !status.active(k) {
                continue;
            }

            // r[k]^T z[k] = r[k]^T M^{-1} r[k] is the divisor of the next beta
            // (and the numerator of alpha): it must not be negligible and must
            // keep its sign.
            if sign_breakdown(rz[k].clone(), rr[k].norm(), zz[k].norm(), &mut rz_sign[k]) {
                let true_residual = true_residual_norm(a, &b[k], &x[k]);
                status.settle_breakdown(k, true_residual, &tol_abs[k]);
                continue;
            }

            // ap = A * p[k]
            a.apply(&p[k], &mut ap);

            // alpha = rz[k] / (p[k]^T * ap)
            let (pap, pp, apap) = dot_and_sums_of_squares(&p[k], &ap);

            if sign_breakdown(pap.clone(), pp.norm(), apap.norm(), &mut pap_sign[k]) {
                let true_residual = true_residual_norm(a, &b[k], &x[k]);
                status.settle_breakdown(k, true_residual, &tol_abs[k]);
                continue;
            }

            let alpha = rz[k].clone() / pap;

            // x[k] = x[k] + alpha * p[k]
            for i in 0..n {
                x[k][i] = x[k][i].clone() + alpha.clone() * p[k][i].clone();
            }

            // r[k] = r[k] - alpha * ap
            for i in 0..n {
                r[k][i] = r[k][i].clone() - alpha.clone() * ap[i].clone();
            }

            // z[k] = M^{-1} * r[k]
            z[k] = precond(&r[k]);

            // rz_new = r[k]^T * z[k] (with the norms of r[k] and z[k])
            let (rz_new, rr_new, zz_new) = dot_and_sums_of_squares(&r[k], &z[k]);

            // beta = rz_new / rz[k] (rz[k] passed the breakdown test above)
            let beta = rz_new.clone() / rz[k].clone();

            // p[k] = z[k] + beta * p[k]
            for i in 0..n {
                p[k][i] = z[k][i].clone() + beta.clone() * p[k][i].clone();
            }

            rz[k] = rz_new;
            rr[k] = rr_new;
            zz[k] = zz_new;
        }
    }

    let norms: Vec<T> = r.iter().map(|rk| norm(rk)).collect();
    Ok(status.finish(x, max_iter, norms))
}
