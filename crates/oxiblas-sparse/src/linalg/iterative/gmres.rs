//! GMRES (Generalized Minimal Residual) and Preconditioned GMRES solvers.

use super::helpers::{dot, givens_rotation_gmres, norm, solve_upper_triangular};
use super::types::{GmresResult, IterativeError};
use crate::csr::CsrMatrix;
use crate::ops::spmv;
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Solver function. See module documentation for details.
pub fn gmres<T: Scalar<Real = T> + Clone + Field + Real>(
    a: &CsrMatrix<T>,
    b: &[T],
    x0: &[T],
    restart: usize,
    tol: T,
    max_iter: usize,
) -> Result<GmresResult<T>, IterativeError> {
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
        spmv(T::one(), a, &x, T::zero(), &mut r);
        for i in 0..n {
            r[i] = b[i].clone() - r[i].clone();
        }

        let beta = norm(&r);
        residual_history.push(beta.clone());

        if beta <= tol_abs {
            return Ok(GmresResult {
                x,
                iterations: total_iter,
                restarts,
                residual_norm: beta,
                converged: true,
                residual_history,
            });
        }

        // Arnoldi process with Givens rotations
        // V: orthonormal basis vectors (n x (restart+1))
        // H: upper Hessenberg matrix ((restart+1) x restart)
        let mut v: Vec<Vec<T>> = Vec::with_capacity(restart + 1);

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

        let mut inner_converged = false;

        // Inner loop (Arnoldi iteration)
        for j in 0..restart {
            if total_iter >= max_iter {
                break;
            }

            // w = A * v[j]
            let mut w = vec![T::zero(); n];
            spmv(T::one(), a, &v[j], T::zero(), &mut w);

            // Modified Gram-Schmidt orthogonalization
            let mut h_col = vec![T::zero(); j + 2];
            for i in 0..=j {
                h_col[i] = dot(&v[i], &w);
                for k in 0..n {
                    w[k] = w[k].clone() - h_col[i].clone() * v[i][k].clone();
                }
            }
            h_col[j + 1] = norm(&w);

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
                // Converged - solve upper triangular system H*y = g
                let y = solve_upper_triangular(&h, &g, j + 1);

                // Update solution: x = x + V*y
                for i in 0..=j {
                    for k in 0..n {
                        x[k] = x[k].clone() + y[i].clone() * v[i][k].clone();
                    }
                }

                return Ok(GmresResult {
                    x,
                    iterations: total_iter,
                    restarts,
                    residual_norm: res_norm,
                    converged: true,
                    residual_history,
                });
            }

            // Check for breakdown
            if Scalar::abs(h[j][j + 1].clone()) <= <T as Scalar>::epsilon() {
                // Lucky breakdown - exact solution found in Krylov subspace
                let y = solve_upper_triangular(&h, &g, j + 1);
                for i in 0..=j {
                    for k in 0..n {
                        x[k] = x[k].clone() + y[i].clone() * v[i][k].clone();
                    }
                }
                inner_converged = true;
                break;
            }

            // Normalize w to get v[j+1]
            let mut v_new = vec![T::zero(); n];
            let h_norm = h[j][j + 1].clone();
            for k in 0..n {
                v_new[k] = w[k].clone() / h_norm.clone();
            }
            v.push(v_new);
        }

        if !inner_converged && !h.is_empty() {
            // Solve upper triangular system before restart
            let m = h.len();
            let y = solve_upper_triangular(&h, &g, m);

            // Update solution: x = x + V*y
            for i in 0..m {
                for k in 0..n {
                    x[k] = x[k].clone() + y[i].clone() * v[i][k].clone();
                }
            }
        }

        restarts += 1;
    }

    // Final residual computation
    let mut r = vec![T::zero(); n];
    spmv(T::one(), a, &x, T::zero(), &mut r);
    for i in 0..n {
        r[i] = b[i].clone() - r[i].clone();
    }
    let final_residual = norm(&r);

    Ok(GmresResult {
        x,
        iterations: total_iter,
        restarts,
        residual_norm: final_residual,
        converged: false,
        residual_history,
    })
}

/// Solver function. See module documentation for details.
pub fn pgmres<T, F>(
    a: &CsrMatrix<T>,
    b: &[T],
    x0: &[T],
    precond: F,
    restart: usize,
    tol: T,
    max_iter: usize,
) -> Result<GmresResult<T>, IterativeError>
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

    let restart = restart.min(n).max(1);
    let mut x = x0.to_vec();
    let mut residual_history = Vec::new();
    let mut total_iter = 0;
    let mut restarts = 0;

    let b_norm = norm(b);
    let tol_abs = tol.clone() * b_norm.clone();

    while total_iter < max_iter {
        // Compute initial residual r = b - A*x
        let mut r = vec![T::zero(); n];
        spmv(T::one(), a, &x, T::zero(), &mut r);
        for i in 0..n {
            r[i] = b[i].clone() - r[i].clone();
        }

        let beta = norm(&r);
        residual_history.push(beta.clone());

        if beta <= tol_abs {
            return Ok(GmresResult {
                x,
                iterations: total_iter,
                restarts,
                residual_norm: beta,
                converged: true,
                residual_history,
            });
        }

        let mut v: Vec<Vec<T>> = Vec::with_capacity(restart + 1);
        let mut z: Vec<Vec<T>> = Vec::with_capacity(restart); // Preconditioned vectors

        let mut v0 = vec![T::zero(); n];
        for i in 0..n {
            v0[i] = r[i].clone() / beta.clone();
        }
        v.push(v0);

        let mut h: Vec<Vec<T>> = Vec::with_capacity(restart);
        let mut cs: Vec<T> = Vec::with_capacity(restart);
        let mut sn: Vec<T> = Vec::with_capacity(restart);
        let mut g = vec![T::zero(); restart + 1];
        g[0] = beta.clone();

        let mut inner_converged = false;

        for j in 0..restart {
            if total_iter >= max_iter {
                break;
            }

            // z[j] = M^{-1} * v[j]
            let z_j = precond(&v[j]);
            z.push(z_j.clone());

            // w = A * z[j]
            let mut w = vec![T::zero(); n];
            spmv(T::one(), a, &z_j, T::zero(), &mut w);

            let mut h_col = vec![T::zero(); j + 2];
            for i in 0..=j {
                h_col[i] = dot(&v[i], &w);
                for k in 0..n {
                    w[k] = w[k].clone() - h_col[i].clone() * v[i][k].clone();
                }
            }
            h_col[j + 1] = norm(&w);

            for i in 0..j {
                let temp = cs[i].clone() * h_col[i].clone() + sn[i].clone() * h_col[i + 1].clone();
                h_col[i + 1] = T::zero() - sn[i].clone() * h_col[i].clone()
                    + cs[i].clone() * h_col[i + 1].clone();
                h_col[i] = temp;
            }

            let (c, s, r_val) = givens_rotation_gmres(h_col[j].clone(), h_col[j + 1].clone());
            cs.push(c.clone());
            sn.push(s.clone());

            h_col[j] = r_val;
            h_col[j + 1] = T::zero();

            let temp = c.clone() * g[j].clone() + s.clone() * g[j + 1].clone();
            g[j + 1] = T::zero() - s.clone() * g[j].clone() + c.clone() * g[j + 1].clone();
            g[j] = temp;

            h.push(h_col);

            let res_norm = Scalar::abs(g[j + 1].clone());
            residual_history.push(res_norm.clone());
            total_iter += 1;

            if res_norm <= tol_abs {
                // Converged - solve upper triangular system H*y = g
                let y = solve_upper_triangular(&h, &g, j + 1);

                // Update: x = x + Z*y (using preconditioned vectors)
                for i in 0..=j {
                    for k in 0..n {
                        x[k] = x[k].clone() + y[i].clone() * z[i][k].clone();
                    }
                }

                return Ok(GmresResult {
                    x,
                    iterations: total_iter,
                    restarts,
                    residual_norm: res_norm,
                    converged: true,
                    residual_history,
                });
            }

            if Scalar::abs(h[j][j + 1].clone()) <= <T as Scalar>::epsilon() {
                inner_converged = true;
                let y = solve_upper_triangular(&h, &g, j + 1);
                for i in 0..=j {
                    for k in 0..n {
                        x[k] = x[k].clone() + y[i].clone() * z[i][k].clone();
                    }
                }
                break;
            }

            let mut v_new = vec![T::zero(); n];
            let h_norm = h[j][j + 1].clone();
            for k in 0..n {
                v_new[k] = w[k].clone() / h_norm.clone();
            }
            v.push(v_new);
        }

        if !inner_converged && !h.is_empty() {
            let m = h.len();
            let y = solve_upper_triangular(&h, &g, m);

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

    let mut r = vec![T::zero(); n];
    spmv(T::one(), a, &x, T::zero(), &mut r);
    for i in 0..n {
        r[i] = b[i].clone() - r[i].clone();
    }
    let final_residual = norm(&r);

    Ok(GmresResult {
        x,
        iterations: total_iter,
        restarts,
        residual_norm: final_residual,
        converged: false,
        residual_history,
    })
}
