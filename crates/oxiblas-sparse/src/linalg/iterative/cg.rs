//! Conjugate Gradient (CG) and Preconditioned CG solvers.

use super::helpers::{dot, norm};
use super::types::{CgResult, IterativeError};
use crate::csr::CsrMatrix;
use crate::ops::spmv;
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Solver function. See module documentation for details.
pub fn cg<T: Scalar<Real = T> + Clone + Field + Real>(
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
    let mut ap = vec![T::zero(); n];

    // r = b - A*x
    spmv(T::one(), a, &x, T::zero(), &mut r);
    for i in 0..n {
        r[i] = b[i].clone() - r[i].clone();
    }

    // p = r
    p.clone_from_slice(&r);

    // rr = r^T * r
    let mut rr = dot(&r, &r);

    let b_norm = norm(b);
    let tol_abs = tol.clone() * b_norm.clone();

    for iter in 0..max_iter {
        let r_norm = Real::sqrt(rr.clone());

        if r_norm <= tol_abs {
            return Ok(CgResult {
                x,
                iterations: iter,
                residual_norm: r_norm,
                converged: true,
            });
        }

        // ap = A * p
        spmv(T::one(), a, &p, T::zero(), &mut ap);

        // alpha = rr / (p^T * ap)
        let pap = dot(&p, &ap);

        if Scalar::abs(pap.clone()) <= <T as Scalar>::epsilon() {
            return Err(IterativeError::Breakdown {
                iteration: iter,
                description: "p^T * A * p near zero".to_string(),
            });
        }

        let alpha = rr.clone() / pap;

        // x = x + alpha * p
        for i in 0..n {
            x[i] = x[i].clone() + alpha.clone() * p[i].clone();
        }

        // r = r - alpha * ap
        for i in 0..n {
            r[i] = r[i].clone() - alpha.clone() * ap[i].clone();
        }

        // rr_new = r^T * r
        let rr_new = dot(&r, &r);

        // beta = rr_new / rr
        let beta = rr_new.clone() / rr.clone();

        // p = r + beta * p
        for i in 0..n {
            p[i] = r[i].clone() + beta.clone() * p[i].clone();
        }

        rr = rr_new;
    }

    let r_norm = Real::sqrt(rr);
    Ok(CgResult {
        x,
        iterations: max_iter,
        residual_norm: r_norm,
        converged: false,
    })
}

/// Solver function. See module documentation for details.
pub fn pcg<T, F>(
    a: &CsrMatrix<T>,
    b: &[T],
    x0: &[T],
    precond: F,
    tol: T,
    max_iter: usize,
) -> Result<CgResult<T>, IterativeError>
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
    let mut r = vec![T::zero(); n];
    let mut ap = vec![T::zero(); n];

    // r = b - A*x
    spmv(T::one(), a, &x, T::zero(), &mut r);
    for i in 0..n {
        r[i] = b[i].clone() - r[i].clone();
    }

    // z = M^{-1} * r
    let mut z = precond(&r);

    // p = z
    let mut p = z.clone();

    // rz = r^T * z
    let mut rz = dot(&r, &z);

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

        // ap = A * p
        spmv(T::one(), a, &p, T::zero(), &mut ap);

        // alpha = rz / (p^T * ap)
        let pap = dot(&p, &ap);

        if Scalar::abs(pap.clone()) <= <T as Scalar>::epsilon() {
            return Err(IterativeError::Breakdown {
                iteration: iter,
                description: "p^T * A * p near zero".to_string(),
            });
        }

        let alpha = rz.clone() / pap;

        // x = x + alpha * p
        for i in 0..n {
            x[i] = x[i].clone() + alpha.clone() * p[i].clone();
        }

        // r = r - alpha * ap
        for i in 0..n {
            r[i] = r[i].clone() - alpha.clone() * ap[i].clone();
        }

        // z = M^{-1} * r
        z = precond(&r);

        // rz_new = r^T * z
        let rz_new = dot(&r, &z);

        // beta = rz_new / rz
        let beta = rz_new.clone() / rz.clone();

        // p = z + beta * p
        for i in 0..n {
            p[i] = z[i].clone() + beta.clone() * p[i].clone();
        }

        rz = rz_new;
    }

    let r_norm = norm(&r);
    Ok(CgResult {
        x,
        iterations: max_iter,
        residual_norm: r_norm,
        converged: false,
    })
}
