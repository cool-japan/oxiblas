//! Block GMRES solver for multiple right-hand sides.

use super::helpers::{
    block_inner_prod_gmres, block_qr_gmres, negligible_norm, solve_block_ls_gmres,
};
use super::scaling::{Operator, SystemScaling};
use super::types::{BlockGmresResult, IterativeError};
use crate::csr::CsrMatrix;
use crate::linalg::precond::tolerance::norm2;
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Overflow-safe Frobenius norm of a block of vectors (or of a small dense
/// block), summed column by column.
fn frobenius_norm<T: Scalar<Real = T> + Clone + Field + Real>(block: &[Vec<T>]) -> T {
    norm2(block.iter().flat_map(|column| column.iter().copied()))
}

/// Restarted block GMRES(`restart`) for `A X = B` with `p = b.len()`
/// right-hand sides (`x0` must have one initial guess per right-hand side).
///
/// Convergence is checked on the true residual at the end of every cycle:
/// `‖B − A X‖_F <= tol·‖B‖_F` (overflow-safe norms; for a system far from unit
/// scale all right-hand sides are rescaled by one common factor, see the
/// [module documentation](super)). `iterations` counts block Arnoldi steps.
///
/// # Breakdown
///
/// The block Arnoldi process stops early when the new sub-diagonal block is
/// negligible relative to `‖A V_k‖_F`, the norm it was derived from
/// (scale-invariant). The cycle is then folded into `X` and the true residual
/// decides between convergence and a restart.
pub fn block_gmres<T>(
    a: &CsrMatrix<T>,
    b: &[Vec<T>],
    x0: &[Vec<T>],
    restart: usize,
    tol: T,
    max_iter: usize,
) -> Result<BlockGmresResult<T>, IterativeError>
where
    T: Scalar<Real = T> + Clone + Field + Real,
{
    let scaling = SystemScaling::for_rhs_block(a, b);
    let b_hat: Vec<Vec<T>> = b.iter().map(|bk| scaling.rhs(bk).into_owned()).collect();
    let x0_hat: Vec<Vec<T>> = x0
        .iter()
        .map(|xk| scaling.initial_guess(xk).into_owned())
        .collect();
    let mut result = block_gmres_scaled(
        &scaling.operator(a),
        &b_hat,
        &x0_hat,
        restart,
        tol,
        max_iter,
    )?;
    for x in result.x.iter_mut() {
        scaling.solution(x);
    }
    result.residual_norm = scaling.residual(result.residual_norm);
    scaling.residuals(&mut result.residual_history);
    Ok(result)
}

/// [`block_gmres`] on the (rescaled) systems `a X = B`.
fn block_gmres_scaled<T>(
    a: &Operator<'_, T>,
    b: &[Vec<T>],
    x0: &[Vec<T>],
    restart: usize,
    tol: T,
    max_iter: usize,
) -> Result<BlockGmresResult<T>, IterativeError>
where
    T: Scalar<Real = T> + Clone + Field + Real,
{
    let n = a.nrows();
    let p = b.len();

    // Validate dimensions
    if a.ncols() != n {
        return Err(IterativeError::DimensionMismatch {
            expected: n,
            actual: a.ncols(),
        });
    }
    if p == 0 || x0.len() != p {
        return Err(IterativeError::DimensionMismatch {
            expected: p,
            actual: x0.len(),
        });
    }
    for (i, bi) in b.iter().enumerate() {
        if bi.len() != n {
            return Err(IterativeError::DimensionMismatch {
                expected: n,
                actual: bi.len(),
            });
        }
        if x0[i].len() != n {
            return Err(IterativeError::DimensionMismatch {
                expected: n,
                actual: x0[i].len(),
            });
        }
    }

    let restart = restart.max(1).min(n);

    // Initialize X and R = B - A*X
    let mut x: Vec<Vec<T>> = x0.to_vec();
    let mut r: Vec<Vec<T>> = (0..p).map(|_| vec![T::zero(); n]).collect();

    for j in 0..p {
        a.apply(&x[j], &mut r[j]);
        for i in 0..n {
            r[j][i] = b[j][i].clone() - r[j][i].clone();
        }
    }

    // Compute B norm (Frobenius, overflow-safe)
    let b_norm = frobenius_norm(b);

    // Compute initial residual norm
    let mut r_norm = frobenius_norm(&r);
    let mut residual_history = vec![r_norm.clone()];

    if r_norm <= tol.clone() * b_norm.clone() {
        return Ok(BlockGmresResult {
            x,
            iterations: 0,
            restarts: 0,
            residual_norm: r_norm,
            converged: true,
            residual_history,
        });
    }

    let mut total_iter = 0;
    let mut restarts = 0;

    // Outer restart loop
    while total_iter < max_iter {
        // QR factorization of R to get initial block: R = V_1 * R_0
        let (v1, r0) = block_qr_gmres(&r, n, p);

        // V is a vector of block columns (each is p vectors of length n)
        let mut v: Vec<Vec<Vec<T>>> = vec![v1];

        // Block upper Hessenberg matrix stored column by column:
        // h_cols[k][i] is the p x p block H_{i,k} for i = 0..=k+1.
        let mut h_cols: Vec<Vec<Vec<Vec<T>>>> = Vec::with_capacity(restart);

        // Block Arnoldi iteration
        let mut k = 0;
        while k < restart && total_iter < max_iter {
            // W = A * V_k (p matrix-vector products)
            let mut w: Vec<Vec<T>> = (0..p).map(|_| vec![T::zero(); n]).collect();
            for j in 0..p {
                a.apply(&v[k][j], &mut w[j]);
            }
            // ||A V_k||_F: the norm the new sub-diagonal block is derived from
            let w_norm = frobenius_norm(&w);

            // Block MGS orthogonalization
            let mut h_col: Vec<Vec<Vec<T>>> = Vec::with_capacity(k + 2);
            for i in 0..=k {
                // H[i][k] = V_i^T * W (p x p block)
                let h_ik = block_inner_prod_gmres(&v[i], &w, p);

                // W = W - V_i * H[i][k]
                for jw in 0..p {
                    for jv in 0..p {
                        for row in 0..n {
                            w[jw][row] =
                                w[jw][row].clone() - v[i][jv][row].clone() * h_ik[jv][jw].clone();
                        }
                    }
                }
                h_col.push(h_ik);
            }

            // QR factorization of W to get V_{k+1} and H[k+1][k]
            let (v_new, h_new) = block_qr_gmres(&w, n, p);
            let h_norm = frobenius_norm(&h_new);

            h_col.push(h_new);
            h_cols.push(h_col);
            v.push(v_new);

            k += 1;
            total_iter += 1;

            // Block breakdown: the Krylov block space is (numerically) invariant
            if negligible_norm(h_norm, w_norm) {
                break;
            }
        }

        // Solve the block least-squares problem
        let m_rows = (k + 1) * p;
        let m_cols = k * p;

        if m_cols == 0 {
            break;
        }

        let mut h_full: Vec<Vec<T>> = vec![vec![T::zero(); m_cols]; m_rows];

        for (col_block, h_col) in h_cols.iter().enumerate() {
            for (row_block, block) in h_col.iter().enumerate() {
                for (bi, block_row) in block.iter().enumerate() {
                    for (bj, value) in block_row.iter().enumerate() {
                        h_full[row_block * p + bi][col_block * p + bj] = value.clone();
                    }
                }
            }
        }

        // RHS: e_1 * R_0
        let mut rhs: Vec<Vec<T>> = vec![vec![T::zero(); p]; m_rows];
        for i in 0..p {
            for j in 0..p {
                rhs[i][j] = r0[i][j].clone();
            }
        }

        // Solve H * Y = RHS
        let y = solve_block_ls_gmres(&h_full, &rhs, m_rows, m_cols, p);

        // Update X: X = X + V * Y
        for col_block in 0..k {
            for bi in 0..p {
                for bj in 0..p {
                    let y_idx = col_block * p + bi;
                    if y_idx < y.len() && bj < y[y_idx].len() {
                        let y_val = y[y_idx][bj].clone();
                        for row in 0..n {
                            x[bj][row] =
                                x[bj][row].clone() + v[col_block][bi][row].clone() * y_val.clone();
                        }
                    }
                }
            }
        }

        // Recompute residual
        for j in 0..p {
            a.apply(&x[j], &mut r[j]);
            for i in 0..n {
                r[j][i] = b[j][i].clone() - r[j][i].clone();
            }
        }

        r_norm = frobenius_norm(&r);
        residual_history.push(r_norm.clone());

        if r_norm <= tol.clone() * b_norm.clone() {
            return Ok(BlockGmresResult {
                x,
                iterations: total_iter,
                restarts,
                residual_norm: r_norm,
                converged: true,
                residual_history,
            });
        }

        restarts += 1;
    }

    Ok(BlockGmresResult {
        x,
        iterations: total_iter,
        restarts,
        residual_norm: r_norm,
        converged: false,
        residual_history,
    })
}
