//! BiCGStab (Bi-Conjugate Gradient Stabilized) solver.

use super::helpers::{dot, norm};
use super::types::{CgResult, IterativeError};
use crate::csr::CsrMatrix;
use crate::ops::spmv;
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Solver function. See module documentation for details.
pub fn bicgstab<T: Scalar<Real = T> + Clone + Field + Real>(
    a: &CsrMatrix<T>,
    b: &[T],
    x0: &[T],
    tol: T,
    max_iter: usize,
) -> Result<CgResult<T>, IterativeError> {
    let n = a.nrows();

    if b.len() != n || x0.len() != n {
        return Err(IterativeError::DimensionMismatch {
            expected: n,
            actual: b.len().min(x0.len()),
        });
    }

    let mut x = x0.to_vec();
    let mut r = vec![T::zero(); n];
    let mut p = vec![T::zero(); n];
    let mut v = vec![T::zero(); n];
    let mut s = vec![T::zero(); n];
    let mut t = vec![T::zero(); n];

    // r = b - A*x
    spmv(T::one(), a, &x, T::zero(), &mut r);
    for i in 0..n {
        r[i] = b[i].clone() - r[i].clone();
    }

    // r_hat = r (shadow residual, kept constant)
    let r_hat = r.clone();

    // p = r
    p.clone_from_slice(&r);

    let mut rho = dot(&r_hat, &r);

    let b_norm = norm(b);
    let tol_abs = tol.clone() * b_norm.clone();

    for iter in 0..max_iter {
        let r_norm = norm(&r);

        if r_norm <= tol_abs {
            return Ok(CgResult {
                x,
                iterations: iter,
                residual_norm: r_norm,
                converged: true,
            });
        }

        // v = A * p
        spmv(T::one(), a, &p, T::zero(), &mut v);

        // alpha = rho / (r_hat^T * v)
        let rv = dot(&r_hat, &v);

        if Scalar::abs(rv.clone()) <= <T as Scalar>::epsilon() {
            return Err(IterativeError::Breakdown {
                iteration: iter,
                description: "r_hat^T * v near zero".to_string(),
            });
        }

        let alpha = rho.clone() / rv;

        // s = r - alpha * v
        for i in 0..n {
            s[i] = r[i].clone() - alpha.clone() * v[i].clone();
        }

        // Check if s is small enough
        let s_norm = norm(&s);
        if s_norm <= tol_abs {
            // x = x + alpha * p
            for i in 0..n {
                x[i] = x[i].clone() + alpha.clone() * p[i].clone();
            }
            return Ok(CgResult {
                x,
                iterations: iter,
                residual_norm: s_norm,
                converged: true,
            });
        }

        // t = A * s
        spmv(T::one(), a, &s, T::zero(), &mut t);

        // omega = (t^T * s) / (t^T * t)
        let tt = dot(&t, &t);

        if Scalar::abs(tt.clone()) <= <T as Scalar>::epsilon() {
            return Err(IterativeError::Breakdown {
                iteration: iter,
                description: "t^T * t near zero".to_string(),
            });
        }

        let omega = dot(&t, &s) / tt;

        // x = x + alpha * p + omega * s
        for i in 0..n {
            x[i] = x[i].clone() + alpha.clone() * p[i].clone() + omega.clone() * s[i].clone();
        }

        // r = s - omega * t
        for i in 0..n {
            r[i] = s[i].clone() - omega.clone() * t[i].clone();
        }

        // rho_new = r_hat^T * r
        let rho_new = dot(&r_hat, &r);

        if Scalar::abs(rho.clone()) <= <T as Scalar>::epsilon()
            || Scalar::abs(omega.clone()) <= <T as Scalar>::epsilon()
        {
            return Err(IterativeError::Breakdown {
                iteration: iter,
                description: "rho or omega near zero".to_string(),
            });
        }

        // beta = (rho_new / rho) * (alpha / omega)
        let beta = (rho_new.clone() / rho.clone()) * (alpha / omega.clone());

        // p = r + beta * (p - omega * v)
        for i in 0..n {
            p[i] = r[i].clone() + beta.clone() * (p[i].clone() - omega.clone() * v[i].clone());
        }

        rho = rho_new;
    }

    let r_norm = norm(&r);
    Ok(CgResult {
        x,
        iterations: max_iter,
        residual_norm: r_norm,
        converged: false,
    })
}
