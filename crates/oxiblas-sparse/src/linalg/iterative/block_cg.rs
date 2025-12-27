//! Block Conjugate Gradient solvers for multiple right-hand sides.

use super::helpers::{dot, norm};
use super::types::{BlockCgResult, IterativeError};
use crate::csr::CsrMatrix;
use crate::ops::spmv;
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Solver function. See module documentation for details.
pub fn block_cg<T: Scalar<Real = T> + Clone + Field + Real>(
    a: &CsrMatrix<T>,
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
        spmv(T::one(), a, &x[k], T::zero(), &mut rk);
        for i in 0..n {
            rk[i] = b[k][i].clone() - rk[i].clone();
        }
        r.push(rk);
    }

    // Initialize search directions: P = R
    let mut p: Vec<Vec<T>> = r.clone();

    // Compute initial residual norms and tolerance
    let mut b_norms: Vec<T> = Vec::with_capacity(num_rhs);
    let mut tol_abs: Vec<T> = Vec::with_capacity(num_rhs);
    for k in 0..num_rhs {
        let bn = norm(&b[k]);
        let ta = tol.clone() * bn.clone();
        b_norms.push(bn);
        tol_abs.push(ta);
    }

    // Compute initial r^T * r for each system
    let mut rr: Vec<T> = Vec::with_capacity(num_rhs);
    for k in 0..num_rhs {
        rr.push(dot(&r[k], &r[k]));
    }

    // Track which systems have converged
    let mut converged_flags: Vec<bool> = vec![false; num_rhs];

    // Working vector for A * p
    let mut ap = vec![T::zero(); n];

    for iter in 0..max_iter {
        // Check convergence for all systems
        let mut all_converged = true;
        for k in 0..num_rhs {
            if !converged_flags[k] {
                let r_norm = Real::sqrt(rr[k].clone());
                if r_norm <= tol_abs[k] {
                    converged_flags[k] = true;
                } else {
                    all_converged = false;
                }
            }
        }

        if all_converged {
            let residual_norms: Vec<T> = rr.iter().map(|rr_k| Real::sqrt(rr_k.clone())).collect();
            let num_converged = converged_flags.iter().filter(|&&c| c).count();
            return Ok(BlockCgResult {
                x,
                iterations: iter,
                residual_norms,
                converged: true,
                num_converged,
            });
        }

        // Process each non-converged system
        for k in 0..num_rhs {
            if converged_flags[k] {
                continue;
            }

            // ap = A * p[k]
            spmv(T::one(), a, &p[k], T::zero(), &mut ap);

            // alpha = rr[k] / (p[k]^T * ap)
            let pap = dot(&p[k], &ap);

            if Scalar::abs(pap.clone()) <= <T as Scalar>::epsilon() {
                // System k has breakdown - mark as converged at current state
                converged_flags[k] = true;
                continue;
            }

            let alpha = rr[k].clone() / pap;

            // x[k] = x[k] + alpha * p[k]
            for i in 0..n {
                x[k][i] = x[k][i].clone() + alpha.clone() * p[k][i].clone();
            }

            // r[k] = r[k] - alpha * ap
            for i in 0..n {
                r[k][i] = r[k][i].clone() - alpha.clone() * ap[i].clone();
            }

            // rr_new = r[k]^T * r[k]
            let rr_new = dot(&r[k], &r[k]);

            // beta = rr_new / rr[k]
            let beta = rr_new.clone() / rr[k].clone();

            // p[k] = r[k] + beta * p[k]
            for i in 0..n {
                p[k][i] = r[k][i].clone() + beta.clone() * p[k][i].clone();
            }

            rr[k] = rr_new;
        }
    }

    let residual_norms: Vec<T> = rr.iter().map(|rr_k| Real::sqrt(rr_k.clone())).collect();
    let num_converged = converged_flags.iter().filter(|&&c| c).count();

    Ok(BlockCgResult {
        x,
        iterations: max_iter,
        residual_norms,
        converged: num_converged == num_rhs,
        num_converged,
    })
}

/// Solver function. See module documentation for details.
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
        spmv(T::one(), a, &x[k], T::zero(), &mut rk);
        for i in 0..n {
            rk[i] = b[k][i].clone() - rk[i].clone();
        }
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

    // Compute initial r^T * z for each system
    let mut rz: Vec<T> = Vec::with_capacity(num_rhs);
    for k in 0..num_rhs {
        rz.push(dot(&r[k], &z[k]));
    }

    // Track which systems have converged
    let mut converged_flags: Vec<bool> = vec![false; num_rhs];

    // Working vector for A * p
    let mut ap = vec![T::zero(); n];

    for iter in 0..max_iter {
        // Check convergence for all systems
        let mut all_converged = true;
        for k in 0..num_rhs {
            if !converged_flags[k] {
                let r_norm = norm(&r[k]);
                if r_norm <= tol_abs[k] {
                    converged_flags[k] = true;
                } else {
                    all_converged = false;
                }
            }
        }

        if all_converged {
            let residual_norms: Vec<T> = r.iter().map(|rk| norm(rk)).collect();
            let num_converged = converged_flags.iter().filter(|&&c| c).count();
            return Ok(BlockCgResult {
                x,
                iterations: iter,
                residual_norms,
                converged: true,
                num_converged,
            });
        }

        // Process each non-converged system
        for k in 0..num_rhs {
            if converged_flags[k] {
                continue;
            }

            // ap = A * p[k]
            spmv(T::one(), a, &p[k], T::zero(), &mut ap);

            // alpha = rz[k] / (p[k]^T * ap)
            let pap = dot(&p[k], &ap);

            if Scalar::abs(pap.clone()) <= <T as Scalar>::epsilon() {
                converged_flags[k] = true;
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

            // rz_new = r[k]^T * z[k]
            let rz_new = dot(&r[k], &z[k]);

            // beta = rz_new / rz[k]
            let beta = rz_new.clone() / rz[k].clone();

            // p[k] = z[k] + beta * p[k]
            for i in 0..n {
                p[k][i] = z[k][i].clone() + beta.clone() * p[k][i].clone();
            }

            rz[k] = rz_new;
        }
    }

    let residual_norms: Vec<T> = r.iter().map(|rk| norm(rk)).collect();
    let num_converged = converged_flags.iter().filter(|&&c| c).count();

    Ok(BlockCgResult {
        x,
        iterations: max_iter,
        residual_norms,
        converged: num_converged == num_rhs,
        num_converged,
    })
}
