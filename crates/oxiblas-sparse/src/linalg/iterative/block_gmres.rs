//! Block GMRES solver for multiple right-hand sides.

use super::helpers::{block_inner_prod_gmres, block_qr_gmres, solve_block_ls_gmres};
use super::types::{BlockGmresResult, IterativeError};
use crate::csr::CsrMatrix;
use crate::ops::spmv;
use oxiblas_core::scalar::{Field, Real, Scalar};

/// Solver function. See module documentation for details.
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
        spmv(T::one(), a, &x[j], T::zero(), &mut r[j]);
        for i in 0..n {
            r[j][i] = b[j][i].clone() - r[j][i].clone();
        }
    }

    // Compute B norm (Frobenius)
    let mut b_norm_sq = T::zero();
    for bj in b {
        for bi in bj {
            b_norm_sq = b_norm_sq + bi.clone() * bi.clone();
        }
    }
    let b_norm = Real::sqrt(b_norm_sq);

    // Compute initial residual norm
    let mut r_norm_sq = T::zero();
    for rj in &r {
        for ri in rj {
            r_norm_sq = r_norm_sq + ri.clone() * ri.clone();
        }
    }
    let mut r_norm = Real::sqrt(r_norm_sq);
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

        // H is block upper Hessenberg: (restart+1) x restart blocks, each p x p
        let mut h: Vec<Vec<Vec<Vec<T>>>> = Vec::with_capacity(restart + 1);
        for _ in 0..=restart {
            h.push(Vec::with_capacity(restart));
        }

        // Block Arnoldi iteration
        let mut k = 0;
        while k < restart && total_iter < max_iter {
            // W = A * V_k (p matrix-vector products)
            let mut w: Vec<Vec<T>> = (0..p).map(|_| vec![T::zero(); n]).collect();
            for j in 0..p {
                spmv(T::one(), a, &v[k][j], T::zero(), &mut w[j]);
            }

            // Block MGS orthogonalization
            for i in 0..=k {
                // H[i][k] = V_i^T * W (p x p block)
                let h_ik = block_inner_prod_gmres(&v[i], &w, p);
                h[i].push(h_ik.clone());

                // W = W - V_i * H[i][k]
                for jw in 0..p {
                    for jv in 0..p {
                        for row in 0..n {
                            w[jw][row] =
                                w[jw][row].clone() - v[i][jv][row].clone() * h_ik[jv][jw].clone();
                        }
                    }
                }
            }

            // QR factorization of W to get V_{k+1} and H[k+1][k]
            let (v_new, h_new) = block_qr_gmres(&w, n, p);

            // Check for breakdown
            let mut h_norm_sq = T::zero();
            for hi in &h_new {
                for hv in hi {
                    h_norm_sq = h_norm_sq + hv.clone() * hv.clone();
                }
            }
            let h_norm = Real::sqrt(h_norm_sq);

            h[k + 1].push(h_new);
            v.push(v_new);

            k += 1;
            total_iter += 1;

            if h_norm < <T as Scalar>::epsilon() {
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

        for col_block in 0..k {
            for row_block in 0..=k.min(col_block + 1) {
                if row_block < h.len() && col_block < h[row_block].len() {
                    let block = &h[row_block][col_block];
                    for bi in 0..p {
                        for bj in 0..p {
                            let row = row_block * p + bi;
                            let col = col_block * p + bj;
                            if row < m_rows
                                && col < m_cols
                                && bi < block.len()
                                && bj < block[bi].len()
                            {
                                h_full[row][col] = block[bi][bj].clone();
                            }
                        }
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
            spmv(T::one(), a, &x[j], T::zero(), &mut r[j]);
            for i in 0..n {
                r[j][i] = b[j][i].clone() - r[j][i].clone();
            }
        }

        r_norm_sq = T::zero();
        for rj in &r {
            for ri in rj {
                r_norm_sq = r_norm_sq + ri.clone() * ri.clone();
            }
        }
        r_norm = Real::sqrt(r_norm_sq);
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
