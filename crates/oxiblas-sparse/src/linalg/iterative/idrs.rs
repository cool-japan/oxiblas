//! IDR(s) (Induced Dimension Reduction) solvers.
//!
//! Both solvers follow the biorthogonal IDR(s) variant of van Gijzen and
//! Sonneveld (ACM TOMS 38(1), Algorithm 913). After the `k`-th step of a cycle
//! the residual is orthogonal to `p_0, ..., p_k`, so the right-hand side
//! `f = Pᵀ r` of the small lower-triangular system has `f_0 = ... = f_k = 0`;
//! the next step therefore only combines the directions `k + 1, ..., s - 1`
//! inherited from the previous cycle.
//!
//! Convergence of the recursively updated residual is always confirmed with
//! the true residual `‖b − A x‖` before it is reported. When the two have
//! drifted apart (a residual gap), the recursive residual is replaced by the
//! true one and the iteration continues with a fresh cycle.
//!
//! # Iteration count
//!
//! `iterations` counts the products with `A` that the iteration performs,
//! where they are performed: one per IDR step (also for a step whose pivot
//! breaks down and ends the cycle), one per stabilization step and one per
//! true-residual confirmation. `max_iter` bounds this count: a product is only
//! started while the count is below `max_iter`, and a confirmation that no
//! longer fits is left to the final report. Not counted are the initial
//! residual `b − A x0` and, when the iteration ends without a confirmed
//! convergence, the true residual `‖b − A x‖` computed for the report (which
//! then also decides `converged`).
//!
//! `residual_history` holds the initial residual norm and one entry per update
//! of the residual (IDR or stabilization step). When a step's residual was
//! confirmed (or replaced after a residual gap), the entry is the true
//! residual norm; the final report does the same for the last entry, so the
//! history always ends with the reported `residual_norm`.

use super::helpers::{
    dot, dot_and_sums_of_squares, negligible_inner, norm, solve_lower_triangular_s,
    true_residual_norm,
};
use super::scaling::{Operator, PrecondScaling, SystemScaling};
use super::types::{IdrSResult, IterativeError};
use crate::csr::CsrMatrix;
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Maps a result of the rescaled system back to the original one.
fn unscale_idrs<T: Scalar<Real = T> + Clone + Field + Real>(
    scaling: &SystemScaling<T>,
    mut result: IdrSResult<T>,
) -> IdrSResult<T> {
    scaling.solution(&mut result.x);
    result.residual_norm = scaling.residual(result.residual_norm);
    scaling.residuals(&mut result.residual_history);
    result
}

/// Stabilization parameter used during the first cycle. Every direction
/// generated in that cycle carries this factor and it cancels in the updates
/// of `x` and `r`, so (in exact arithmetic) its value does not matter.
const INITIAL_OMEGA: f64 = 0.7;

/// Angle `kappa` of the "maintaining the convergence" safeguard (Sleijpen and
/// van der Vorst), used when `omega = tᵀr / tᵀt` would vanish numerically: the
/// limit of that safeguard is `|omega| = kappa·‖r‖ / ‖t‖`.
const SAFEGUARD_ANGLE: f64 = 0.7;

/// Entry `(i, j)` of the raw shadow space: the SplitMix64 hash of a unique
/// counter, mapped to a uniform value in `[-1/2, 1/2)` (exact in binary64).
fn shadow_entry(i: usize, j: usize) -> f64 {
    let mut z = (((j as u64) << 40) ^ (i as u64)).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // The top 53 bits give an exact uniform value in [0, 1).
    (z >> 11) as f64 / 9_007_199_254_740_992.0 - 0.5
}

/// Shadow space `P` (`n x s`, columns orthonormalized by modified
/// Gram–Schmidt).
///
/// IDR(s) assumes a *generic* `P`. A structured choice (for instance a short
/// periodic pattern) can be nearly orthogonal to smooth residuals: a step
/// length then vanishes and the following direction becomes numerically
/// dependent on its predecessors, which silently corrupts `x`. The entries are
/// therefore deterministic but structure-free. `P` does not depend on `A` or
/// `b`, so it does not affect scale invariance.
fn shadow_space<T>(n: usize, s: usize) -> Vec<Vec<T>>
where
    T: Scalar<Real = T> + Clone + Field + Real,
{
    let mut p: Vec<Vec<T>> = Vec::with_capacity(s);
    for j in 0..s {
        let mut pj: Vec<T> = (0..n)
            .map(|i| T::from_f64(shadow_entry(i, j)).unwrap_or_else(T::zero))
            .collect();
        for pk in &p {
            let projection = dot(&pj, pk);
            for i in 0..n {
                pj[i] = pj[i].clone() - projection.clone() * pk[i].clone();
            }
        }
        let pj_norm = norm(&pj);
        if pj_norm > T::zero() {
            for value in pj.iter_mut() {
                *value = value.clone() / pj_norm.clone();
            }
        }
        p.push(pj);
    }
    p
}

/// Computes the stabilization parameter `omega` minimizing `‖r − omega·t‖`.
///
/// Returns `None` when `t = 0` (no step is possible; `omega` is kept). When `t`
/// is (numerically) orthogonal to `r`, i.e. `|tᵀr| <= eps·‖t‖·‖r‖`, the
/// minimizer would vanish and stall the dimension reduction, so the
/// scale-aware safeguard value `sign(tᵀr)·kappa·‖r‖ / ‖t‖` is used instead.
fn stabilization_omega<T>(t: &[T], r: &[T]) -> Option<T>
where
    T: Scalar<Real = T> + Clone + Field + Real,
{
    let (t_dot_r, t_squares, r_squares) = dot_and_sums_of_squares(t, r);
    if t_squares.is_zero() {
        return None;
    }
    let t_dot_t = t_squares.value();
    let t_norm = t_squares.norm();
    let r_norm = r_squares.norm();
    if negligible_inner(t_dot_r.clone(), t_norm.clone(), r_norm.clone()) {
        let kappa = T::from_f64(SAFEGUARD_ANGLE).unwrap_or_else(T::one);
        let magnitude = kappa * r_norm / t_norm;
        return Some(if t_dot_r >= T::zero() {
            magnitude
        } else {
            T::zero() - magnitude
        });
    }
    Some(t_dot_r / t_dot_t)
}

/// Confirms a converged recursive residual with the true residual.
///
/// Overwrites `r` with `b − A x` and `r_norm` with its norm, and counts the
/// product with `A` in `iterations`. Returns `Some(‖b − A x‖)` when the true
/// residual meets `tol_abs`; otherwise returns `None`, leaving the true
/// residual in `r` (which closes the residual gap).
fn confirm_or_replace<T>(
    a: &Operator<'_, T>,
    b: &[T],
    x: &[T],
    r: &mut [T],
    r_norm: &mut T,
    tol_abs: T,
    iterations: &mut usize,
) -> Option<T>
where
    T: Scalar<Real = T> + Clone + Field + Real,
{
    a.residual(b, x, r);
    *iterations += 1;
    *r_norm = norm(r);
    if *r_norm <= tol_abs {
        Some(r_norm.clone())
    } else {
        None
    }
}

fn check_dimensions<T: Scalar<Real = T> + Clone + Field + Real>(
    a: &Operator<'_, T>,
    b: &[T],
    x0: &[T],
) -> Result<(), IterativeError> {
    let n = a.nrows();
    if a.ncols() != n || b.len() != n || x0.len() != n {
        return Err(IterativeError::DimensionMismatch {
            expected: n,
            actual: if a.ncols() != n {
                a.ncols()
            } else if b.len() != n {
                b.len()
            } else {
                x0.len()
            },
        });
    }
    Ok(())
}

/// IDR(`s`) for general (non-symmetric) `A`; `s` is clamped to `[1, n]`.
///
/// Converges when the recursively updated residual satisfies
/// `‖r‖ <= tol·‖b‖` and the true residual confirms it (the reported
/// `residual_norm` is then the true residual). `iterations` counts the
/// products with `A`, at most `max_iter`; see the [module
/// documentation](self#iteration-count) for exactly what is counted.
///
/// Norms are overflow-safe; see the [module documentation](super) for
/// systems far from unit scale.
///
/// # Breakdown
///
/// A step whose pivot `μ_kk = p_kᵀ g_k` is negligible relative to
/// `‖p_k‖·‖g_k‖` (scale-invariant) ends the current cycle early; the
/// iteration then continues with the stabilization step. A vanishing
/// stabilization parameter is replaced by a scale-aware safeguard value.
pub fn idrs<T>(
    a: &CsrMatrix<T>,
    b: &[T],
    x0: &[T],
    s: usize,
    tol: T,
    max_iter: usize,
) -> Result<IdrSResult<T>, IterativeError>
where
    T: Scalar<Real = T> + Clone + Field + Real,
{
    let scaling = SystemScaling::for_rhs(a, b);
    let result = idrs_scaled(
        &scaling.operator(a),
        &scaling.rhs(b),
        &scaling.initial_guess(x0),
        s,
        tol,
        max_iter,
    )?;
    Ok(unscale_idrs(&scaling, result))
}

/// [`idrs`] on the (rescaled) system `a x = b`.
fn idrs_scaled<T>(
    a: &Operator<'_, T>,
    b: &[T],
    x0: &[T],
    s: usize,
    tol: T,
    max_iter: usize,
) -> Result<IdrSResult<T>, IterativeError>
where
    T: Scalar<Real = T> + Clone + Field + Real,
{
    check_dimensions(a, b, x0)?;
    let n = a.nrows();

    // Initialize x and r = b - A*x
    let mut x = x0.to_vec();
    let mut r = vec![T::zero(); n];
    a.apply(&x, &mut r);
    for i in 0..n {
        r[i] = b[i].clone() - r[i].clone();
    }

    let b_norm = norm(b);
    let tol_abs = tol.clone() * b_norm.clone();
    let mut r_norm = norm(&r);
    let mut residual_history = vec![r_norm.clone()];

    if r_norm <= tol_abs {
        return Ok(IdrSResult {
            x,
            iterations: 0,
            residual_norm: r_norm,
            converged: true,
            residual_history,
        });
    }

    // At most n shadow vectors can be linearly independent.
    let s = s.clamp(1, n.max(1));
    let p: Vec<Vec<T>> = shadow_space(n, s);
    let p_norms: Vec<T> = p.iter().map(|pj| norm(pj)).collect();

    // Initialize storage for IDR
    let mut g: Vec<Vec<T>> = (0..s).map(|_| vec![T::zero(); n]).collect();
    let mut u: Vec<Vec<T>> = (0..s).map(|_| vec![T::zero(); n]).collect();
    let mut m: Vec<Vec<T>> = (0..s)
        .map(|i| {
            let mut row = vec![T::zero(); s];
            row[i] = T::one();
            row
        })
        .collect();

    let mut omega = T::from_f64(INITIAL_OMEGA).unwrap_or_else(T::one);

    let mut iter = 0;

    while iter < max_iter {
        // f = P^T * r
        let mut f: Vec<T> = p.iter().map(|pj| dot(pj, &r)).collect();

        for k in 0..s {
            // Each step starts with a product with A; stop at the budget.
            if iter >= max_iter {
                break;
            }

            // Solve M * c = f for c
            let c = solve_lower_triangular_s(&m, &f);

            // v = r - sum_{j=0}^{s-1} c[j] * g[j]
            let mut v = r.clone();
            for j in 0..s {
                for i in 0..n {
                    v[i] = v[i].clone() - c[j].clone() * g[j][i].clone();
                }
            }

            // u_new = omega * v + sum_{j=0}^{s-1} c[j] * u[j]
            let mut u_new = vec![T::zero(); n];
            for i in 0..n {
                u_new[i] = omega.clone() * v[i].clone();
            }
            for j in 0..s {
                for i in 0..n {
                    u_new[i] = u_new[i].clone() + c[j].clone() * u[j][i].clone();
                }
            }

            // g_new = A * u_new (counted where it is performed, so that a
            // step whose pivot breaks down below is counted too)
            let mut g_new = vec![T::zero(); n];
            a.apply(&u_new, &mut g_new);
            iter += 1;

            // Bi-orthogonalize g_new and u_new against previous g and u
            for j in 0..k {
                let alpha = dot(&p[j], &g_new) / m[j][j].clone();
                for i in 0..n {
                    g_new[i] = g_new[i].clone() - alpha.clone() * g[j][i].clone();
                    u_new[i] = u_new[i].clone() - alpha.clone() * u[j][i].clone();
                }
            }

            // Store in slot k
            g[k] = g_new;
            u[k] = u_new;

            // m[i][k] = p[i]^T * g[k]
            for i in 0..s {
                m[i][k] = dot(&p[i], &g[k]);
            }

            // Breakdown: the pivot p_k^T g_k is negligible relative to
            // ||p_k|| ||g_k||. End the cycle early and recover through the
            // stabilization step.
            if negligible_inner(m[k][k].clone(), p_norms[k].clone(), norm(&g[k])) {
                break;
            }

            // Update x and r
            let beta = f[k].clone() / m[k][k].clone();
            for i in 0..n {
                x[i] = x[i].clone() + beta.clone() * u[k][i].clone();
                r[i] = r[i].clone() - beta.clone() * g[k][i].clone();
            }

            r_norm = norm(&r);

            if r_norm <= tol_abs {
                if iter < max_iter {
                    let confirmed = confirm_or_replace(
                        a,
                        b,
                        &x,
                        &mut r,
                        &mut r_norm,
                        tol_abs.clone(),
                        &mut iter,
                    );
                    // r_norm is now the true residual norm.
                    residual_history.push(r_norm.clone());
                    if let Some(residual_norm) = confirmed {
                        return Ok(IdrSResult {
                            x,
                            iterations: iter,
                            residual_norm,
                            converged: true,
                            residual_history,
                        });
                    }
                    // Residual gap: r now holds the true residual. The cycle's
                    // bookkeeping (f) no longer applies, so leave the cycle.
                } else {
                    // No budget left for the confirmation: the final report
                    // computes the true residual.
                    residual_history.push(r_norm.clone());
                }
                break;
            }
            residual_history.push(r_norm.clone());

            // The new residual is orthogonal to p_0, ..., p_k: f_0..f_k vanish
            // (in exact arithmetic), and the remaining entries are updated.
            for fi in f.iter_mut().take(k + 1) {
                *fi = T::zero();
            }
            if k < s - 1 {
                for i in (k + 1)..s {
                    f[i] = f[i].clone() - beta.clone() * m[i][k].clone();
                }
            }
        }

        if iter >= max_iter {
            break;
        }

        // Now do the "intermediate" step with omega update
        // v = r
        let v = r.clone();

        // t = A * v
        let mut t = vec![T::zero(); n];
        a.apply(&v, &mut t);
        iter += 1;

        // Update omega = (t^T * v) / (t^T * t)
        if let Some(new_omega) = stabilization_omega(&t, &v) {
            omega = new_omega;
        }

        // r_new = r - omega * t
        for i in 0..n {
            r[i] = v[i].clone() - omega.clone() * t[i].clone();
        }

        // x_new = x + omega * v (v here is the old r before t computation)
        for i in 0..n {
            x[i] = x[i].clone() + omega.clone() * v[i].clone();
        }

        r_norm = norm(&r);

        if r_norm <= tol_abs && iter < max_iter {
            let confirmed =
                confirm_or_replace(a, b, &x, &mut r, &mut r_norm, tol_abs.clone(), &mut iter);
            // r_norm is now the true residual norm.
            residual_history.push(r_norm.clone());
            if let Some(residual_norm) = confirmed {
                return Ok(IdrSResult {
                    x,
                    iterations: iter,
                    residual_norm,
                    converged: true,
                    residual_history,
                });
            }
        } else {
            residual_history.push(r_norm.clone());
        }
    }

    let final_residual = true_residual_norm(a, b, &x);
    // The report confirms or refutes the last residual update; record the
    // true norm for it, as a confirmation would have.
    if let Some(last) = residual_history.last_mut() {
        *last = final_residual.clone();
    }
    Ok(IdrSResult {
        x,
        iterations: iter,
        residual_norm: final_residual.clone(),
        converged: final_residual <= tol_abs,
        residual_history,
    })
}

/// Preconditioned IDR(`s`); `precond` applies `M^{-1}` (right preconditioning:
/// the residual stays the true, unpreconditioned residual). `s` is clamped to
/// `[1, n]`.
///
/// Convergence, breakdown handling and the iteration count are the same as
/// for [`idrs`]; the preconditioner is applied once before each product with
/// `A` of an IDR or stabilization step. For a system far from unit scale
/// `precond` receives vectors of the rescaled system and must be linear (see
/// the [module documentation](super)).
pub fn pidrs<T, M>(
    a: &CsrMatrix<T>,
    b: &[T],
    x0: &[T],
    precond: &M,
    s: usize,
    tol: T,
    max_iter: usize,
) -> Result<IdrSResult<T>, IterativeError>
where
    T: Scalar<Real = T> + Clone + Field + Real,
    M: Fn(&[T]) -> Vec<T>,
{
    let scaling = SystemScaling::for_rhs(a, b);
    let precond_scaling = PrecondScaling::new();
    let scaled_precond = |v: &[T]| precond_scaling.finish(v, precond(v));
    let result = pidrs_scaled(
        &scaling.operator(a),
        &scaling.rhs(b),
        &scaling.initial_guess(x0),
        &scaled_precond,
        s,
        tol,
        max_iter,
    )?;
    Ok(unscale_idrs(&scaling, result))
}

/// [`pidrs`] on the (rescaled) system `a x = b`.
fn pidrs_scaled<T, M>(
    a: &Operator<'_, T>,
    b: &[T],
    x0: &[T],
    precond: &M,
    s: usize,
    tol: T,
    max_iter: usize,
) -> Result<IdrSResult<T>, IterativeError>
where
    T: Scalar<Real = T> + Clone + Field + Real,
    M: Fn(&[T]) -> Vec<T>,
{
    check_dimensions(a, b, x0)?;
    let n = a.nrows();

    // Initialize x and r = b - A*x
    let mut x = x0.to_vec();
    let mut r = vec![T::zero(); n];
    a.apply(&x, &mut r);
    for i in 0..n {
        r[i] = b[i].clone() - r[i].clone();
    }

    let b_norm = norm(b);
    let tol_abs = tol.clone() * b_norm.clone();
    let mut r_norm = norm(&r);
    let mut residual_history = vec![r_norm.clone()];

    if r_norm <= tol_abs {
        return Ok(IdrSResult {
            x,
            iterations: 0,
            residual_norm: r_norm,
            converged: true,
            residual_history,
        });
    }

    // At most n shadow vectors can be linearly independent.
    let s = s.clamp(1, n.max(1));
    let p: Vec<Vec<T>> = shadow_space(n, s);
    let p_norms: Vec<T> = p.iter().map(|pj| norm(pj)).collect();

    let mut g: Vec<Vec<T>> = (0..s).map(|_| vec![T::zero(); n]).collect();
    let mut u: Vec<Vec<T>> = (0..s).map(|_| vec![T::zero(); n]).collect();
    let mut m: Vec<Vec<T>> = (0..s)
        .map(|i| {
            let mut row = vec![T::zero(); s];
            row[i] = T::one();
            row
        })
        .collect();

    let mut omega = T::from_f64(INITIAL_OMEGA).unwrap_or_else(T::one);

    let mut iter = 0;

    while iter < max_iter {
        let mut f: Vec<T> = p.iter().map(|pj| dot(pj, &r)).collect();

        for k in 0..s {
            // Each step starts with a product with A; stop at the budget.
            if iter >= max_iter {
                break;
            }

            let c = solve_lower_triangular_s(&m, &f);

            let mut v = r.clone();
            for j in 0..s {
                for i in 0..n {
                    v[i] = v[i].clone() - c[j].clone() * g[j][i].clone();
                }
            }

            // Apply preconditioner to v
            let mv = precond(&v);

            // u_new = omega * M^{-1}*v + sum c[j] * u[j]
            let mut u_new = vec![T::zero(); n];
            for i in 0..n {
                u_new[i] = omega.clone() * mv[i].clone();
            }
            for j in 0..s {
                for i in 0..n {
                    u_new[i] = u_new[i].clone() + c[j].clone() * u[j][i].clone();
                }
            }

            // Counted where it is performed (see `idrs`).
            let mut g_new = vec![T::zero(); n];
            a.apply(&u_new, &mut g_new);
            iter += 1;

            for j in 0..k {
                let alpha = dot(&p[j], &g_new) / m[j][j].clone();
                for i in 0..n {
                    g_new[i] = g_new[i].clone() - alpha.clone() * g[j][i].clone();
                    u_new[i] = u_new[i].clone() - alpha.clone() * u[j][i].clone();
                }
            }

            g[k] = g_new;
            u[k] = u_new;

            for i in 0..s {
                m[i][k] = dot(&p[i], &g[k]);
            }

            // Breakdown: negligible pivot (see `idrs`).
            if negligible_inner(m[k][k].clone(), p_norms[k].clone(), norm(&g[k])) {
                break;
            }

            let beta = f[k].clone() / m[k][k].clone();
            for i in 0..n {
                x[i] = x[i].clone() + beta.clone() * u[k][i].clone();
                r[i] = r[i].clone() - beta.clone() * g[k][i].clone();
            }

            r_norm = norm(&r);

            if r_norm <= tol_abs {
                if iter < max_iter {
                    let confirmed = confirm_or_replace(
                        a,
                        b,
                        &x,
                        &mut r,
                        &mut r_norm,
                        tol_abs.clone(),
                        &mut iter,
                    );
                    residual_history.push(r_norm.clone());
                    if let Some(residual_norm) = confirmed {
                        return Ok(IdrSResult {
                            x,
                            iterations: iter,
                            residual_norm,
                            converged: true,
                            residual_history,
                        });
                    }
                    // Residual gap: r now holds the true residual (see `idrs`).
                } else {
                    residual_history.push(r_norm.clone());
                }
                break;
            }
            residual_history.push(r_norm.clone());

            // f_0..f_k vanish; update the remaining entries (see `idrs`).
            for fi in f.iter_mut().take(k + 1) {
                *fi = T::zero();
            }
            if k < s - 1 {
                for i in (k + 1)..s {
                    f[i] = f[i].clone() - beta.clone() * m[i][k].clone();
                }
            }
        }

        if iter >= max_iter {
            break;
        }

        // Apply preconditioner
        let mr = precond(&r);

        // t = A * M^{-1} * r
        let mut t = vec![T::zero(); n];
        a.apply(&mr, &mut t);
        iter += 1;

        if let Some(new_omega) = stabilization_omega(&t, &r) {
            omega = new_omega;
        }

        for i in 0..n {
            x[i] = x[i].clone() + omega.clone() * mr[i].clone();
            r[i] = r[i].clone() - omega.clone() * t[i].clone();
        }

        r_norm = norm(&r);

        if r_norm <= tol_abs && iter < max_iter {
            let confirmed =
                confirm_or_replace(a, b, &x, &mut r, &mut r_norm, tol_abs.clone(), &mut iter);
            residual_history.push(r_norm.clone());
            if let Some(residual_norm) = confirmed {
                return Ok(IdrSResult {
                    x,
                    iterations: iter,
                    residual_norm,
                    converged: true,
                    residual_history,
                });
            }
        } else {
            residual_history.push(r_norm.clone());
        }
    }

    let final_residual = true_residual_norm(a, b, &x);
    // The report confirms or refutes the last residual update; record the
    // true norm for it, as a confirmation would have.
    if let Some(last) = residual_history.last_mut() {
        *last = final_residual.clone();
    }
    Ok(IdrSResult {
        x,
        iterations: iter,
        residual_norm: final_residual.clone(),
        converged: final_residual <= tol_abs,
        residual_history,
    })
}
