//! QMR (Quasi-Minimal Residual) solvers.
//!
//! Both solvers use the two-sided (nonsymmetric) Lanczos / BiCG process, which
//! genuinely requires products with BOTH `A` and `A^T` (and, for the
//! preconditioned variant, with `M^{-1}` and its transpose `M^{-T}`), combined
//! with Freund's quasi-minimal residual smoothing of the BiCG residuals.

use super::helpers::{dot, norm};
use super::types::{IterativeError, QmrResult};
use crate::csr::CsrMatrix;
use crate::ops::{spmv, spmv_transpose};
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Solver function. See module documentation for details.
pub fn qmr<T: Scalar<Real = T> + Clone + Field + Real>(
    a: &CsrMatrix<T>,
    b: &[T],
    x0: &[T],
    tol: T,
    max_iter: usize,
) -> Result<QmrResult<T>, IterativeError> {
    let n = a.nrows();

    if b.len() != n || x0.len() != n {
        return Err(IterativeError::DimensionMismatch {
            expected: n,
            actual: b.len().min(x0.len()),
        });
    }

    if a.nrows() != a.ncols() {
        return Err(IterativeError::DimensionMismatch {
            expected: a.nrows(),
            actual: a.ncols(),
        });
    }

    let mut x = x0.to_vec();
    let mut residual_history = Vec::new();

    // Initial residual: r0 = b - A*x0
    let mut r = vec![T::zero(); n];
    spmv(T::one(), a, &x, T::zero(), &mut r);
    for i in 0..n {
        r[i] = b[i].clone() - r[i].clone();
    }

    let b_norm = norm(b);
    if b_norm <= T::zero() {
        return Ok(QmrResult {
            x: vec![T::zero(); n],
            iterations: 0,
            residual_norm: T::zero(),
            converged: true,
            residual_history: vec![T::zero()],
        });
    }
    let tol_abs = tol.clone() * b_norm.clone();

    let r_norm = norm(&r);
    residual_history.push(r_norm.clone());

    if r_norm <= tol_abs {
        return Ok(QmrResult {
            x,
            iterations: 0,
            residual_norm: r_norm,
            converged: true,
            residual_history,
        });
    }

    // ================================================================
    // Two-sided (nonsymmetric) Lanczos / BiCG core with QMR smoothing.
    //
    // Alongside the primary residual r_j and search direction p_j we propagate
    // the DUAL (shadow) residual r_tilde_j = r_j^* and dual search direction
    // p_tilde_j = p_j^*, both advanced with A^T. The biorthogonality
    // <r_i^*, r_j> = 0 (i != j) of the two-sided Lanczos process supplies the
    // BiCG scalars:
    //     rho_j   = <r_j^*, r_j>
    //     sigma_j = <p_j^*, A p_j>
    // The dual sequence is what makes this genuine BiCG (rather than a
    // fixed-shadow surrogate), so the A^T products below are load-bearing, not
    // dead code.
    // ================================================================
    let mut r_tilde = r.clone(); // dual residual r_0^*, updated each step via A^T
    let mut p = r.clone();
    let mut p_tilde = r.clone(); // dual search direction p_0^*

    // QMR smoothing state.
    let mut d = vec![T::zero(); n];
    let mut tau = r_norm.clone();
    let mut theta = T::zero();
    let mut eta = T::zero();

    let eps = <T as Scalar>::epsilon();

    let mut rho = dot(&r_tilde, &r);
    // rho = <r_0^*, r_0> = <r_0, r_0> = ||r_0||^2 > 0 here (r_0 != 0), but guard
    // for robustness against a caller-supplied degenerate start.
    if Scalar::abs(rho.clone()) <= eps * norm(&r_tilde) * norm(&r) {
        return Err(IterativeError::Breakdown {
            iteration: 0,
            description: "QMR breakdown: <r_tilde, r> = 0 at start".to_string(),
        });
    }

    for iter in 0..max_iter {
        // Primary and dual matrix-vector products (two-sided Lanczos):
        //   q       = A   * p
        //   q_tilde = A^T * p_tilde
        let mut q = vec![T::zero(); n];
        spmv(T::one(), a, &p, T::zero(), &mut q);
        let mut q_tilde = vec![T::zero(); n];
        spmv_transpose(T::one(), a, &p_tilde, T::zero(), &mut q_tilde);

        // sigma = <p_tilde, A p> = <p_tilde, q>
        let sigma = dot(&p_tilde, &q);

        // BiCG breakdown of the first kind: <p_tilde, A p> = 0. Without
        // look-ahead this cannot be continued; report an honest breakdown after
        // checking for a lucky breakdown (the true residual is already tiny).
        if Scalar::abs(sigma.clone()) <= eps * norm(&p_tilde) * norm(&q) {
            let actual_residual = true_residual_norm(a, b, &x);
            if actual_residual <= tol_abs {
                return Ok(QmrResult {
                    x,
                    iterations: iter,
                    residual_norm: actual_residual,
                    converged: true,
                    residual_history,
                });
            }
            return Err(IterativeError::Breakdown {
                iteration: iter,
                description: "QMR breakdown: <p_tilde, A p> = 0".to_string(),
            });
        }

        let alpha = rho.clone() / sigma.clone();

        // Advance primary and dual residuals:
        //   r       = r       - alpha * A   p
        //   r_tilde = r_tilde - alpha * A^T p_tilde
        for i in 0..n {
            r[i] = r[i].clone() - alpha.clone() * q[i].clone();
        }
        for i in 0..n {
            r_tilde[i] = r_tilde[i].clone() - alpha.clone() * q_tilde[i].clone();
        }

        // QMR residual smoothing (Freund): quasi-minimize over the BiCG
        // residuals via a sequence of Givens-like rotations.
        let r_norm_new = norm(&r);
        let theta_new = if tau > T::zero() {
            r_norm_new.clone() / tau.clone()
        } else {
            T::one()
        };
        let c_sq = T::one() / (T::one() + theta_new.clone() * theta_new.clone());
        let c = Real::sqrt(c_sq.clone());
        let tau_new = tau.clone() * theta_new.clone() * c.clone();
        let eta_new = c_sq.clone() * alpha.clone();

        // d_j = p_{j-1} + (theta_{j-1}^2 * eta_{j-1} / alpha_{j-1}) * d_{j-1}
        let coeff = theta.clone() * theta.clone() * eta.clone() / alpha.clone();
        for i in 0..n {
            d[i] = p[i].clone() + coeff.clone() * d[i].clone();
        }
        // x_j = x_{j-1} + eta_j * d_j
        for i in 0..n {
            x[i] = x[i].clone() + eta_new.clone() * d[i].clone();
        }

        theta = theta_new;
        tau = tau_new.clone();
        eta = eta_new;

        residual_history.push(tau.clone());

        // Convergence: tau upper-bounds the true residual; confirm with the
        // actual residual before declaring success.
        if tau <= tol_abs {
            let actual_residual = true_residual_norm(a, b, &x);
            if actual_residual <= tol_abs {
                return Ok(QmrResult {
                    x,
                    iterations: iter + 1,
                    residual_norm: actual_residual,
                    converged: true,
                    residual_history,
                });
            }
        }

        // Next BiCG coefficient rho and honest breakdown detection.
        let rho_new = dot(&r_tilde, &r);

        // BiCG breakdown of the second kind (Lanczos breakdown): rho = 0. This
        // is a real breakdown; do NOT fabricate a beta to "continue". Check for
        // a lucky breakdown (converged), otherwise return an honest error.
        if Scalar::abs(rho_new.clone()) <= eps * norm(&r_tilde) * norm(&r) {
            let actual_residual = true_residual_norm(a, b, &x);
            if actual_residual <= tol_abs {
                return Ok(QmrResult {
                    x,
                    iterations: iter + 1,
                    residual_norm: actual_residual,
                    converged: true,
                    residual_history,
                });
            }
            return Err(IterativeError::Breakdown {
                iteration: iter,
                description: "QMR breakdown: <r_tilde, r> = 0 (Lanczos breakdown)".to_string(),
            });
        }

        let beta = rho_new.clone() / rho.clone();

        // Advance primary and dual search directions:
        //   p       = r       + beta * p
        //   p_tilde = r_tilde + beta * p_tilde
        for i in 0..n {
            p[i] = r[i].clone() + beta.clone() * p[i].clone();
        }
        for i in 0..n {
            p_tilde[i] = r_tilde[i].clone() + beta.clone() * p_tilde[i].clone();
        }

        rho = rho_new;
    }

    let final_residual = true_residual_norm(a, b, &x);
    Ok(QmrResult {
        x,
        iterations: max_iter,
        residual_norm: final_residual.clone(),
        converged: final_residual <= tol_abs,
        residual_history,
    })
}

/// Preconditioned QMR with left preconditioner `M1` and right preconditioner
/// `M2`, solving the split-preconditioned operator `B = M1^{-1} A M2^{-1}`.
///
/// The two-sided Lanczos process requires products with `B` and with its
/// transpose `B^T = M2^{-T} A^T M1^{-T}`. The dual (bi-orthogonal) sequence
/// therefore needs the TRANSPOSE preconditioner solves `M1^{-T}` and `M2^{-T}`
/// — applying `M1^{-1}`/`M2^{-1}` there is only correct for symmetric
/// preconditioners and is wrong in general. The caller supplies each solve and
/// its transpose explicitly.
///
/// # Arguments
///
/// * `left_precond` - applies `M1^{-1}`
/// * `left_precond_transpose` - applies `M1^{-T}`
/// * `right_precond` - applies `M2^{-1}`
/// * `right_precond_transpose` - applies `M2^{-T}`
pub fn pqmr<T, FL, FLT, FR, FRT>(
    a: &CsrMatrix<T>,
    b: &[T],
    x0: &[T],
    left_precond: FL,
    left_precond_transpose: FLT,
    right_precond: FR,
    right_precond_transpose: FRT,
    tol: T,
    max_iter: usize,
) -> Result<QmrResult<T>, IterativeError>
where
    T: Scalar<Real = T> + Clone + Field + Real,
    FL: Fn(&[T]) -> Vec<T>,
    FLT: Fn(&[T]) -> Vec<T>,
    FR: Fn(&[T]) -> Vec<T>,
    FRT: Fn(&[T]) -> Vec<T>,
{
    let n = a.nrows();

    if b.len() != n || x0.len() != n {
        return Err(IterativeError::DimensionMismatch {
            expected: n,
            actual: b.len().min(x0.len()),
        });
    }

    if a.nrows() != a.ncols() {
        return Err(IterativeError::DimensionMismatch {
            expected: a.nrows(),
            actual: a.ncols(),
        });
    }

    let mut x = x0.to_vec();
    let mut residual_history = Vec::new();

    // Initial residual: r0 = b - A*x0
    let mut r = vec![T::zero(); n];
    spmv(T::one(), a, &x, T::zero(), &mut r);
    for i in 0..n {
        r[i] = b[i].clone() - r[i].clone();
    }

    let b_norm = norm(b);
    if b_norm <= T::zero() {
        return Ok(QmrResult {
            x: vec![T::zero(); n],
            iterations: 0,
            residual_norm: T::zero(),
            converged: true,
            residual_history: vec![T::zero()],
        });
    }
    // True-residual convergence tolerance (original / unpreconditioned space).
    let tol_abs = tol.clone() * b_norm.clone();

    // Left-preconditioned residual s0 = M1^{-1} r0. This is the residual of the
    // split-preconditioned system B u = M1^{-1} b (with x = M2^{-1} u), and the
    // quantity QMR quasi-minimizes.
    let mut s = left_precond(&r);

    // Scale for the (preconditioned) quasi-residual tolerance.
    let c_precond = left_precond(b);
    let c_norm = norm(&c_precond);
    let tol_pre = tol.clone() * c_norm.clone();

    let s_norm = norm(&s);
    residual_history.push(s_norm.clone());

    // Immediate convergence (also covers a zero initial residual).
    if s_norm <= tol_pre {
        let actual_residual = true_residual_norm(a, b, &x);
        return Ok(QmrResult {
            x,
            iterations: 0,
            residual_norm: actual_residual,
            converged: true,
            residual_history,
        });
    }

    // ================================================================
    // Two-sided Lanczos / BiCG on the split-preconditioned operator
    // B = M1^{-1} A M2^{-1}, with QMR smoothing. The dual sequence uses
    // B^T = M2^{-T} A^T M1^{-T}: the TRANSPOSE preconditioner solves are applied
    // on the dual side. The u-space (preconditioned) smoothing direction d is
    // mapped back to x-space through M2^{-1} for the solution update.
    // ================================================================
    let mut s_tilde = s.clone();
    let mut p = s.clone();
    let mut p_tilde = s.clone();
    let mut d = vec![T::zero(); n];

    let eps = <T as Scalar>::epsilon();

    let mut rho = dot(&s_tilde, &s);
    if Scalar::abs(rho.clone()) <= eps * norm(&s_tilde) * norm(&s) {
        return Err(IterativeError::Breakdown {
            iteration: 0,
            description: "QMR breakdown: <s_tilde, s> = 0 at start".to_string(),
        });
    }

    let mut tau = s_norm.clone();
    let mut theta = T::zero();
    let mut eta = T::zero();

    for iter in 0..max_iter {
        // q = B p = M1^{-1} A M2^{-1} p
        let p_right = right_precond(&p);
        let mut ap = vec![T::zero(); n];
        spmv(T::one(), a, &p_right, T::zero(), &mut ap);
        let q = left_precond(&ap);

        // q_tilde = B^T p_tilde = M2^{-T} A^T M1^{-T} p_tilde
        // (correct transpose preconditioner solves on the dual side).
        let pt_left_t = left_precond_transpose(&p_tilde);
        let mut atpt = vec![T::zero(); n];
        spmv_transpose(T::one(), a, &pt_left_t, T::zero(), &mut atpt);
        let q_tilde = right_precond_transpose(&atpt);

        let sigma = dot(&p_tilde, &q);
        if Scalar::abs(sigma.clone()) <= eps * norm(&p_tilde) * norm(&q) {
            let actual_residual = true_residual_norm(a, b, &x);
            if actual_residual <= tol_abs {
                return Ok(QmrResult {
                    x,
                    iterations: iter,
                    residual_norm: actual_residual,
                    converged: true,
                    residual_history,
                });
            }
            return Err(IterativeError::Breakdown {
                iteration: iter,
                description: "QMR breakdown: <p_tilde, B p> = 0".to_string(),
            });
        }

        let alpha = rho.clone() / sigma.clone();

        // Advance preconditioned primary and dual residuals.
        for i in 0..n {
            s[i] = s[i].clone() - alpha.clone() * q[i].clone();
        }
        for i in 0..n {
            s_tilde[i] = s_tilde[i].clone() - alpha.clone() * q_tilde[i].clone();
        }

        // QMR residual smoothing (in preconditioned / u-space).
        let s_norm_new = norm(&s);
        let theta_new = if tau > T::zero() {
            s_norm_new.clone() / tau.clone()
        } else {
            T::one()
        };
        let c_sq = T::one() / (T::one() + theta_new.clone() * theta_new.clone());
        let c = Real::sqrt(c_sq.clone());
        let tau_new = tau.clone() * theta_new.clone() * c.clone();
        let eta_new = c_sq.clone() * alpha.clone();

        let coeff = theta.clone() * theta.clone() * eta.clone() / alpha.clone();
        for i in 0..n {
            d[i] = p[i].clone() + coeff.clone() * d[i].clone();
        }
        // Map the u-space smoothing direction back to x-space via M2^{-1}:
        //   x = x + eta * M2^{-1} d
        let d_x = right_precond(&d);
        for i in 0..n {
            x[i] = x[i].clone() + eta_new.clone() * d_x[i].clone();
        }

        theta = theta_new;
        tau = tau_new.clone();
        eta = eta_new;

        residual_history.push(tau.clone());

        if tau <= tol_pre {
            let actual_residual = true_residual_norm(a, b, &x);
            if actual_residual <= tol_abs {
                return Ok(QmrResult {
                    x,
                    iterations: iter + 1,
                    residual_norm: actual_residual,
                    converged: true,
                    residual_history,
                });
            }
        }

        let rho_new = dot(&s_tilde, &s);
        if Scalar::abs(rho_new.clone()) <= eps * norm(&s_tilde) * norm(&s) {
            let actual_residual = true_residual_norm(a, b, &x);
            if actual_residual <= tol_abs {
                return Ok(QmrResult {
                    x,
                    iterations: iter + 1,
                    residual_norm: actual_residual,
                    converged: true,
                    residual_history,
                });
            }
            return Err(IterativeError::Breakdown {
                iteration: iter,
                description: "QMR breakdown: <s_tilde, s> = 0 (Lanczos breakdown)".to_string(),
            });
        }

        let beta = rho_new.clone() / rho.clone();
        for i in 0..n {
            p[i] = s[i].clone() + beta.clone() * p[i].clone();
        }
        for i in 0..n {
            p_tilde[i] = s_tilde[i].clone() + beta.clone() * p_tilde[i].clone();
        }
        rho = rho_new;
    }

    let final_residual = true_residual_norm(a, b, &x);
    Ok(QmrResult {
        x,
        iterations: max_iter,
        residual_norm: final_residual.clone(),
        converged: final_residual <= tol_abs,
        residual_history,
    })
}

/// Compute the true residual norm ||b - A x||.
fn true_residual_norm<T: Scalar<Real = T> + Clone + Field + Real>(
    a: &CsrMatrix<T>,
    b: &[T],
    x: &[T],
) -> T {
    let n = a.nrows();
    let mut r_final = vec![T::zero(); n];
    spmv(T::one(), a, x, T::zero(), &mut r_final);
    for i in 0..n {
        r_final[i] = b[i].clone() - r_final[i].clone();
    }
    norm(&r_final)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr::CsrMatrix;

    /// Strongly diagonally-dominant non-symmetric matrix:
    /// A = [10 1 0; 2 10 1; 0 2 10]
    fn nonsymmetric() -> CsrMatrix<f64> {
        let values = vec![10.0, 1.0, 2.0, 10.0, 1.0, 2.0, 10.0];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2];
        let row_ptrs = vec![0, 2, 5, 7];
        CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap()
    }

    #[test]
    fn test_qmr_identity() {
        let a: CsrMatrix<f64> = CsrMatrix::eye(3);
        let b = vec![1.0, 2.0, 3.0];
        let x0 = vec![0.0, 0.0, 0.0];

        let result = qmr(&a, &b, &x0, 1e-12, 100).unwrap();
        assert!(result.converged);
        for i in 0..3 {
            assert!((result.x[i] - b[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn test_qmr_nonsymmetric() {
        let a = nonsymmetric();
        let b = vec![1.0, 2.0, 3.0];
        let x0 = vec![0.0, 0.0, 0.0];

        let result = qmr(&a, &b, &x0, 1e-10, 200).unwrap();
        assert!(result.converged, "QMR should converge on nonsymmetric A");

        let mut ax = vec![0.0; 3];
        spmv(1.0, &a, &result.x, 0.0, &mut ax);
        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-7,
                "QMR solution incorrect at index {i}"
            );
        }
    }

    /// Zero initial residual: x0 already solves the system.
    #[test]
    fn test_qmr_zero_initial_residual() {
        let a = nonsymmetric();
        let x_exact = [1.0, 2.0, 3.0];
        let mut b = vec![0.0; 3];
        spmv(1.0, &a, &x_exact, 0.0, &mut b);

        let result = qmr(&a, &b, &x_exact, 1e-10, 100).unwrap();
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_pqmr_jacobi_symmetric_precond() {
        let a = nonsymmetric();
        let b = vec![1.0, 2.0, 3.0];
        let x0 = vec![0.0, 0.0, 0.0];

        // Jacobi (diagonal, hence symmetric): M1 = diag(10), M2 = I.
        let left = |v: &[f64]| -> Vec<f64> { v.iter().map(|&e| e / 10.0).collect() };
        let left_t = left; // symmetric => M1^{-T} = M1^{-1}
        let right = |v: &[f64]| -> Vec<f64> { v.to_vec() };
        let right_t = right;

        let result = pqmr(&a, &b, &x0, left, left_t, right, right_t, 1e-10, 200).unwrap();
        assert!(result.converged, "preconditioned QMR should converge");

        let mut ax = vec![0.0; 3];
        spmv(1.0, &a, &result.x, 0.0, &mut ax);
        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-7,
                "PQMR solution incorrect at {i}"
            );
        }
    }

    /// Non-symmetric preconditioner: exercises the M1^{-T} transpose solve on the
    /// dual Lanczos sequence. M1 is lower-triangular (M1 != M1^T), M2 = I.
    ///   M1 = [10 0 0; 2 10 0; 0 2 10]
    #[test]
    fn test_pqmr_nonsymmetric_precond_uses_transpose() {
        let a = nonsymmetric();
        let b = vec![1.0, 2.0, 3.0];
        let x0 = vec![0.0, 0.0, 0.0];

        // M1^{-1} y via forward substitution on the lower-triangular M1.
        let left = |y: &[f64]| -> Vec<f64> {
            let x0 = y[0] / 10.0;
            let x1 = (y[1] - 2.0 * x0) / 10.0;
            let x2 = (y[2] - 2.0 * x1) / 10.0;
            vec![x0, x1, x2]
        };
        // M1^{-T} y via back substitution on the upper-triangular M1^T.
        let left_t = |y: &[f64]| -> Vec<f64> {
            let x2 = y[2] / 10.0;
            let x1 = (y[1] - 2.0 * x2) / 10.0;
            let x0 = (y[0] - 2.0 * x1) / 10.0;
            vec![x0, x1, x2]
        };
        let right = |v: &[f64]| -> Vec<f64> { v.to_vec() };
        let right_t = right;

        let result = pqmr(&a, &b, &x0, left, left_t, right, right_t, 1e-10, 200).unwrap();
        assert!(
            result.converged,
            "PQMR with non-symmetric preconditioner should converge"
        );

        let mut ax = vec![0.0; 3];
        spmv(1.0, &a, &result.x, 0.0, &mut ax);
        for i in 0..3 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-7,
                "PQMR (nonsymmetric precond) solution incorrect at {i}"
            );
        }
    }

    #[test]
    fn test_qmr_dimension_mismatch() {
        let a = nonsymmetric();
        let b = vec![1.0, 2.0]; // wrong dimension
        let x0 = vec![0.0, 0.0, 0.0];

        let result = qmr(&a, &b, &x0, 1e-10, 100);
        assert!(matches!(
            result,
            Err(IterativeError::DimensionMismatch { .. })
        ));
    }
}
