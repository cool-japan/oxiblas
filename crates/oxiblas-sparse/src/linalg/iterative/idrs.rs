//! IDR(s) (Induced Dimension Reduction) solvers.

use super::helpers::{dot, norm, solve_lower_triangular_s};
use super::types::{IdrSResult, IterativeError};
use crate::csr::CsrMatrix;
use crate::ops::spmv;
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Solver function. See module documentation for details.
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
    let n = a.nrows();

    // Validate dimensions
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

    let s = s.max(1); // s must be at least 1

    // Initialize x and r = b - A*x
    let mut x = x0.to_vec();
    let mut r = vec![T::zero(); n];
    spmv(T::one(), a, &x, T::zero(), &mut r);
    for i in 0..n {
        r[i] = b[i].clone() - r[i].clone();
    }

    let b_norm = norm(b);
    let mut r_norm = norm(&r);
    let mut residual_history = vec![r_norm.clone()];

    if r_norm <= tol.clone() * b_norm.clone() {
        return Ok(IdrSResult {
            x,
            iterations: 0,
            residual_norm: r_norm,
            converged: true,
            residual_history,
        });
    }

    // Build the P matrix: n x s random matrix
    // Using simple deterministic "random" vectors for reproducibility
    let mut p: Vec<Vec<T>> = Vec::with_capacity(s);
    for j in 0..s {
        let mut pj = vec![T::zero(); n];
        for i in 0..n {
            // Simple deterministic pseudo-random pattern
            let val = ((i + 1) * (j + 1) * 17 % 1000) as f64 / 1000.0 - 0.5;
            pj[i] = T::from_f64(val).unwrap_or_else(T::one);
        }
        // Orthogonalize against previous P vectors
        for k in 0..j {
            let dot_val = dot(&pj, &p[k]);
            let pk_norm_sq = dot(&p[k], &p[k]);
            if Scalar::abs(pk_norm_sq.clone()) > <T as Scalar>::epsilon() {
                let scale = dot_val / pk_norm_sq;
                for i in 0..n {
                    pj[i] = pj[i].clone() - scale.clone() * p[k][i].clone();
                }
            }
        }
        // Normalize
        let pj_norm = norm(&pj);
        if pj_norm > <T as Scalar>::epsilon() {
            for i in 0..n {
                pj[i] = pj[i].clone() / pj_norm.clone();
            }
        }
        p.push(pj);
    }

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

    let omega_init = T::from_f64(0.7).unwrap_or_else(T::one);
    let mut omega = omega_init;

    let mut iter = 0;

    while iter < max_iter {
        // f = P^T * r
        let mut f: Vec<T> = p.iter().map(|pj| dot(pj, &r)).collect();

        for k in 0..s {
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

            // g_new = A * u_new
            let mut g_new = vec![T::zero(); n];
            spmv(T::one(), a, &u_new, T::zero(), &mut g_new);

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

            // Check for breakdown
            if Scalar::abs(m[k][k].clone()) < <T as Scalar>::epsilon() {
                // Breakdown - try to recover by continuing
                break;
            }

            // Update x and r
            let beta = f[k].clone() / m[k][k].clone();
            for i in 0..n {
                x[i] = x[i].clone() + beta.clone() * u[k][i].clone();
                r[i] = r[i].clone() - beta.clone() * g[k][i].clone();
            }

            r_norm = norm(&r);
            residual_history.push(r_norm.clone());
            iter += 1;

            if r_norm <= tol.clone() * b_norm.clone() {
                return Ok(IdrSResult {
                    x,
                    iterations: iter,
                    residual_norm: r_norm,
                    converged: true,
                    residual_history,
                });
            }

            if iter >= max_iter {
                break;
            }

            // Update f for next iteration within this cycle
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
        spmv(T::one(), a, &v, T::zero(), &mut t);

        // Update omega = (t^T * v) / (t^T * t)
        let t_dot_v = dot(&t, &v);
        let t_dot_t = dot(&t, &t);
        if Scalar::abs(t_dot_t.clone()) > <T as Scalar>::epsilon() {
            omega = t_dot_v / t_dot_t;
            // Ensure omega has proper sign and magnitude
            if Scalar::abs(omega.clone()) < <T as Scalar>::epsilon() {
                omega = omega_init.clone();
            }
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
        residual_history.push(r_norm.clone());
        iter += 1;

        if r_norm <= tol.clone() * b_norm.clone() {
            return Ok(IdrSResult {
                x,
                iterations: iter,
                residual_norm: r_norm,
                converged: true,
                residual_history,
            });
        }
    }

    Ok(IdrSResult {
        x,
        iterations: iter,
        residual_norm: r_norm,
        converged: false,
        residual_history,
    })
}

/// Solver function. See module documentation for details.
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
    let n = a.nrows();

    // Validate dimensions
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

    let s = s.max(1);

    // Initialize x and r = b - A*x
    let mut x = x0.to_vec();
    let mut r = vec![T::zero(); n];
    spmv(T::one(), a, &x, T::zero(), &mut r);
    for i in 0..n {
        r[i] = b[i].clone() - r[i].clone();
    }

    let b_norm = norm(b);
    let mut r_norm = norm(&r);
    let mut residual_history = vec![r_norm.clone()];

    if r_norm <= tol.clone() * b_norm.clone() {
        return Ok(IdrSResult {
            x,
            iterations: 0,
            residual_norm: r_norm,
            converged: true,
            residual_history,
        });
    }

    // Build P matrix
    let mut p: Vec<Vec<T>> = Vec::with_capacity(s);
    for j in 0..s {
        let mut pj = vec![T::zero(); n];
        for i in 0..n {
            let val = ((i + 1) * (j + 1) * 17 % 1000) as f64 / 1000.0 - 0.5;
            pj[i] = T::from_f64(val).unwrap_or_else(T::one);
        }
        for k in 0..j {
            let dot_val = dot(&pj, &p[k]);
            let pk_norm_sq = dot(&p[k], &p[k]);
            if Scalar::abs(pk_norm_sq.clone()) > <T as Scalar>::epsilon() {
                let scale = dot_val / pk_norm_sq;
                for i in 0..n {
                    pj[i] = pj[i].clone() - scale.clone() * p[k][i].clone();
                }
            }
        }
        let pj_norm = norm(&pj);
        if pj_norm > <T as Scalar>::epsilon() {
            for i in 0..n {
                pj[i] = pj[i].clone() / pj_norm.clone();
            }
        }
        p.push(pj);
    }

    let mut g: Vec<Vec<T>> = (0..s).map(|_| vec![T::zero(); n]).collect();
    let mut u: Vec<Vec<T>> = (0..s).map(|_| vec![T::zero(); n]).collect();
    let mut m: Vec<Vec<T>> = (0..s)
        .map(|i| {
            let mut row = vec![T::zero(); s];
            row[i] = T::one();
            row
        })
        .collect();

    let omega_init = T::from_f64(0.7).unwrap_or_else(T::one);
    let mut omega = omega_init;

    let mut iter = 0;

    while iter < max_iter {
        let mut f: Vec<T> = p.iter().map(|pj| dot(pj, &r)).collect();

        for k in 0..s {
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

            let mut g_new = vec![T::zero(); n];
            spmv(T::one(), a, &u_new, T::zero(), &mut g_new);

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

            if Scalar::abs(m[k][k].clone()) < <T as Scalar>::epsilon() {
                break;
            }

            let beta = f[k].clone() / m[k][k].clone();
            for i in 0..n {
                x[i] = x[i].clone() + beta.clone() * u[k][i].clone();
                r[i] = r[i].clone() - beta.clone() * g[k][i].clone();
            }

            r_norm = norm(&r);
            residual_history.push(r_norm.clone());
            iter += 1;

            if r_norm <= tol.clone() * b_norm.clone() {
                return Ok(IdrSResult {
                    x,
                    iterations: iter,
                    residual_norm: r_norm,
                    converged: true,
                    residual_history,
                });
            }

            if iter >= max_iter {
                break;
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
        spmv(T::one(), a, &mr, T::zero(), &mut t);

        let t_dot_r = dot(&t, &r);
        let t_dot_t = dot(&t, &t);
        if Scalar::abs(t_dot_t.clone()) > <T as Scalar>::epsilon() {
            omega = t_dot_r / t_dot_t;
            if Scalar::abs(omega.clone()) < <T as Scalar>::epsilon() {
                omega = omega_init.clone();
            }
        }

        for i in 0..n {
            x[i] = x[i].clone() + omega.clone() * mr[i].clone();
            r[i] = r[i].clone() - omega.clone() * t[i].clone();
        }

        r_norm = norm(&r);
        residual_history.push(r_norm.clone());
        iter += 1;

        if r_norm <= tol.clone() * b_norm.clone() {
            return Ok(IdrSResult {
                x,
                iterations: iter,
                residual_norm: r_norm,
                converged: true,
                residual_history,
            });
        }
    }

    Ok(IdrSResult {
        x,
        iterations: iter,
        residual_norm: r_norm,
        converged: false,
        residual_history,
    })
}
