//! LOBPCG (Locally Optimal Block Preconditioned Conjugate Gradient) eigensolver.
//!
//! Implements the Knyazev (2001) algorithm for computing a block of extreme eigenvalues
//! of a symmetric positive (semi-)definite sparse matrix A x = λ x.
//!
//! # Algorithm Overview
//!
//! LOBPCG maintains three blocks of k vectors each:
//! - **X**: current approximate eigenvectors
//! - **W**: preconditioned residuals (W = P⁻¹ R, R = AX - XΛ)
//! - **P**: previous search directions (for conjugacy / "locally optimal" update)
//!
//! At each step, the method forms S = [X, W, P] (or [X, W] on the first step),
//! projects A onto the column space of S, solves the small dense EVP via Jacobi
//! iteration, and updates X and P from the Ritz pairs.
//!
//! # Convergence
//!
//! Convergence is declared for eigenpair j when:
//! `‖AX[:,j] - λ_j X[:,j]‖ / max(1, |λ_j|) < tol`
//!
//! # References
//!
//! - Knyazev, A. V. (2001). "Toward the Optimal Preconditioned Eigensolver: Locally
//!   Optimal Block Preconditioned Conjugate Gradient Method." SIAM J. Sci. Comput.,
//!   23(2), 517–541.

use core::fmt;

use crate::csr::CsrMatrix;
use crate::ops::spmv;

// =============================================================================
// Public types
// =============================================================================

/// Which extreme eigenvalues to target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LobpcgTarget {
    /// Algebraically smallest eigenvalues.
    Smallest,
    /// Algebraically largest eigenvalues.
    Largest,
}

/// Configuration for the LOBPCG eigensolver.
#[derive(Debug, Clone)]
pub struct LobpcgConfig {
    /// Number of eigenpairs to compute (k).
    pub num_eigenvalues: usize,
    /// Maximum number of iterations. Default: 300.
    pub max_iter: usize,
    /// Convergence tolerance on the relative residual. Default: 1e-10.
    pub tol: f64,
    /// Whether to seek smallest or largest eigenvalues. Default: Smallest.
    pub which: LobpcgTarget,
    /// Block size. `None` means use `num_eigenvalues`. Must be >= `num_eigenvalues`.
    pub block_size: Option<usize>,
}

impl Default for LobpcgConfig {
    fn default() -> Self {
        Self {
            num_eigenvalues: 1,
            max_iter: 300,
            tol: 1e-10,
            which: LobpcgTarget::Smallest,
            block_size: None,
        }
    }
}

/// Result of a successful (or partially successful) LOBPCG computation.
#[derive(Debug, Clone)]
pub struct LobpcgResult {
    /// Computed eigenvalues sorted in ascending order (smallest first)
    /// regardless of the `which` setting — caller interprets direction.
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors; `eigenvectors[j]` is the j-th eigenvector (length n).
    pub eigenvectors: Vec<Vec<f64>>,
    /// Relative residual norms `‖AX[:,j] - λ_j X[:,j]‖ / max(1, |λ_j|)`.
    pub residuals: Vec<f64>,
    /// Number of iterations performed.
    pub n_iter: usize,
    /// Number of eigenpairs that satisfied the convergence criterion.
    pub converged: usize,
}

/// Error type for the LOBPCG solver.
#[derive(Debug, Clone)]
pub enum LobpcgError {
    /// Solver did not converge to all requested eigenpairs.
    NotConverged {
        /// Number of converged eigenpairs.
        found: usize,
        /// Number of requested eigenpairs.
        requested: usize,
    },
    /// Configuration is invalid.
    InvalidConfig(String),
    /// A numerical failure occurred during the computation.
    NumericalFailure(String),
    /// The user-supplied preconditioner produced an invalid result.
    PreconditionerError(String),
}

impl fmt::Display for LobpcgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotConverged { found, requested } => write!(
                f,
                "LOBPCG did not converge: {found}/{requested} eigenpairs converged"
            ),
            Self::InvalidConfig(msg) => write!(f, "Invalid LOBPCG configuration: {msg}"),
            Self::NumericalFailure(msg) => write!(f, "LOBPCG numerical failure: {msg}"),
            Self::PreconditionerError(msg) => write!(f, "LOBPCG preconditioner error: {msg}"),
        }
    }
}

impl std::error::Error for LobpcgError {}

// =============================================================================
// Solver
// =============================================================================

/// LOBPCG eigensolver for sparse symmetric matrices.
///
/// # Example
///
/// ```ignore
/// use oxiblas_sparse::linalg::eigenvalue::{Lobpcg, LobpcgConfig, LobpcgTarget};
///
/// let config = LobpcgConfig {
///     num_eigenvalues: 3,
///     tol: 1e-10,
///     which: LobpcgTarget::Smallest,
///     ..Default::default()
/// };
/// let solver = Lobpcg::new(config).unwrap();
/// let result = solver.compute(&csr).unwrap();
/// ```
pub struct Lobpcg {
    config: LobpcgConfig,
}

impl Lobpcg {
    /// Create a new solver, validating the configuration.
    pub fn new(config: LobpcgConfig) -> Result<Self, LobpcgError> {
        if config.num_eigenvalues == 0 {
            return Err(LobpcgError::InvalidConfig(
                "num_eigenvalues must be >= 1".to_string(),
            ));
        }
        if config.max_iter == 0 {
            return Err(LobpcgError::InvalidConfig(
                "max_iter must be >= 1".to_string(),
            ));
        }
        if config.tol <= 0.0 {
            return Err(LobpcgError::InvalidConfig("tol must be > 0".to_string()));
        }
        Ok(Self { config })
    }

    /// Solve the standard eigenvalue problem A x = λ x (no preconditioner).
    pub fn compute(&self, csr: &CsrMatrix<f64>) -> Result<LobpcgResult, LobpcgError> {
        self.compute_preconditioned(csr, |r, w| w.copy_from_slice(r))
    }

    /// Solve A x = λ x with a user-supplied preconditioner.
    ///
    /// `precond(r, w)` should compute `w = P⁻¹ r` in place.
    pub fn compute_preconditioned<F>(
        &self,
        csr: &CsrMatrix<f64>,
        precond: F,
    ) -> Result<LobpcgResult, LobpcgError>
    where
        F: Fn(&[f64], &mut [f64]),
    {
        let n = csr.nrows();
        if csr.ncols() != n {
            return Err(LobpcgError::InvalidConfig(format!(
                "Matrix must be square, got {}×{}",
                n,
                csr.ncols()
            )));
        }
        let k = self.config.num_eigenvalues;
        if k > n {
            return Err(LobpcgError::InvalidConfig(format!(
                "num_eigenvalues ({k}) must be <= matrix dimension ({n})"
            )));
        }

        // Block size: at least k, at most n
        let block_size = self.config.block_size.unwrap_or(k).max(k).min(n);

        // Initialize X: block_size orthonormal vectors
        let mut x = init_eigenvector_block(n, block_size)?;

        // P (previous search directions): empty on first iteration
        let mut p: Vec<Vec<f64>> = Vec::new();

        // Current eigenvalue estimates
        let mut lambda = vec![0.0_f64; block_size];

        let mut n_iter = 0usize;
        let mut converged_count;

        for iter in 0..self.config.max_iter {
            n_iter = iter + 1;

            let done = self.lobpcg_step(csr, &mut x, &mut p, &mut lambda, &precond, iter)?;
            // Count converged
            let ax = spmv_block(csr, &x);
            converged_count = 0usize;
            for j in 0..block_size {
                let res = vec_axpy(-lambda[j], &x[j], &ax[j]);
                let res_norm = vec_norm(&res);
                let rel = res_norm / lambda[j].abs().max(1.0);
                if rel < self.config.tol {
                    converged_count += 1;
                }
            }

            if done || converged_count >= k {
                break;
            }
        }

        // Compute final residuals and sort output by eigenvalue
        let ax_final = spmv_block(csr, &x);
        let mut pairs: Vec<(f64, Vec<f64>, f64)> = (0..block_size)
            .map(|j| {
                let lam = lambda[j];
                let res = vec_axpy(-lam, &x[j], &ax_final[j]);
                let rel = vec_norm(&res) / lam.abs().max(1.0);
                (lam, x[j].clone(), rel)
            })
            .collect();

        // Sort ascending; for Largest we will return the k largest (last k after sort)
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let output_pairs: Vec<(f64, Vec<f64>, f64)> = match self.config.which {
            LobpcgTarget::Smallest => pairs.into_iter().take(k).collect(),
            LobpcgTarget::Largest => pairs.into_iter().rev().take(k).collect(),
        };

        let eigenvalues: Vec<f64> = output_pairs.iter().map(|(l, _, _)| *l).collect();
        let eigenvectors: Vec<Vec<f64>> = output_pairs.iter().map(|(_, v, _)| v.clone()).collect();
        let residuals: Vec<f64> = output_pairs.iter().map(|(_, _, r)| *r).collect();

        converged_count = residuals.iter().filter(|&&r| r < self.config.tol).count();

        if converged_count < k {
            return Err(LobpcgError::NotConverged {
                found: converged_count,
                requested: k,
            });
        }

        Ok(LobpcgResult {
            eigenvalues,
            eigenvectors,
            residuals,
            n_iter,
            converged: converged_count,
        })
    }

    // -------------------------------------------------------------------------
    // Core LOBPCG step
    // -------------------------------------------------------------------------

    /// Perform one LOBPCG iteration.
    ///
    /// Returns `true` if all block_size vectors have converged.
    fn lobpcg_step(
        &self,
        csr: &CsrMatrix<f64>,
        x: &mut Vec<Vec<f64>>,
        p: &mut Vec<Vec<f64>>,
        lambda: &mut Vec<f64>,
        precond: &dyn Fn(&[f64], &mut [f64]),
        iter: usize,
    ) -> Result<bool, LobpcgError> {
        let n = csr.nrows();
        let bs = x.len();

        // --- Step 1: Compute A*X ---
        let ax = spmv_block(csr, x);

        // --- Step 2: Rayleigh quotients and residuals ---
        let mut residuals: Vec<Vec<f64>> = Vec::with_capacity(bs);
        for j in 0..bs {
            lambda[j] = vec_dot(&x[j], &ax[j]);
            let r = vec_axpy(-lambda[j], &x[j], &ax[j]);
            residuals.push(r);
        }

        // --- Step 3: Check convergence ---
        let all_converged = residuals.iter().enumerate().all(|(j, r)| {
            let rel = vec_norm(r) / lambda[j].abs().max(1.0);
            rel < self.config.tol
        });
        if all_converged {
            return Ok(true);
        }

        // --- Step 4: Apply preconditioner to residuals -> W ---
        let mut w: Vec<Vec<f64>> = Vec::with_capacity(bs);
        for r in &residuals {
            let mut wj = vec![0.0_f64; n];
            precond(r, &mut wj);
            // Validate preconditioner output
            if wj.iter().any(|v| v.is_nan() || v.is_infinite()) {
                return Err(LobpcgError::PreconditionerError(
                    "Preconditioner produced NaN or Inf".to_string(),
                ));
            }
            w.push(wj);
        }

        // --- Step 5: Orthogonalize W against X ---
        for wj in w.iter_mut() {
            orthogonalize_against(wj, x);
        }

        // --- Step 6: Build subspace S = [X | W | P] or [X | W] ---
        let mut s: Vec<Vec<f64>> = Vec::new();
        for xj in x.iter() {
            s.push(xj.clone());
        }
        for wj in &w {
            s.push(wj.clone());
        }
        let has_p = iter > 0 && !p.is_empty();
        if has_p {
            for pj in p.iter() {
                s.push(pj.clone());
            }
        }

        // --- Step 7: Orthonormalize S ---
        let s_cols = mgs_orthonormalize(&mut s)?;
        s.truncate(s_cols);

        // --- Step 8: Compute A*S ---
        let a_s = spmv_block(csr, &s);

        // --- Step 9: Projected matrix A_proj = S^T A S ---
        let dim = s.len();
        let mut a_proj: Vec<Vec<f64>> = vec![vec![0.0; dim]; dim];
        for i in 0..dim {
            for jj in 0..dim {
                a_proj[i][jj] = vec_dot(&s[i], &a_s[jj]);
            }
        }

        // --- Step 10: Symmetrize (numerical noise) ---
        for i in 0..dim {
            for jj in (i + 1)..dim {
                let avg = 0.5 * (a_proj[i][jj] + a_proj[jj][i]);
                a_proj[i][jj] = avg;
                a_proj[jj][i] = avg;
            }
        }

        // --- Step 11: Solve dense symmetric EVP via Jacobi ---
        let (evals, evecs) = jacobi_evd(&mut a_proj, 100)?;

        // evals come out ascending; for Largest we want the LAST bs cols
        let col_offset = match self.config.which {
            LobpcgTarget::Smallest => 0,
            LobpcgTarget::Largest => dim.saturating_sub(bs),
        };

        // --- Step 12: Update lambda ---
        for j in 0..bs {
            let col = col_offset + j;
            if col < evals.len() {
                lambda[j] = evals[col];
            }
        }

        // --- Step 13: Compute new X = S * evecs[:, col_offset..col_offset+bs] ---
        let mut x_new: Vec<Vec<f64>> = vec![vec![0.0; n]; bs];
        for j in 0..bs {
            let col = col_offset + j;
            if col >= evecs[0].len() {
                continue;
            }
            for (si, sv) in s.iter().enumerate() {
                let coeff = evecs[si][col];
                for i in 0..n {
                    x_new[j][i] += coeff * sv[i];
                }
            }
        }

        // --- Step 14: Update P = S * evecs[:, (non-X columns)] ---
        // P columns come from the W (and old P) block of Ritz vectors
        let p_col_start = match self.config.which {
            LobpcgTarget::Smallest => bs,        // skip first bs (X) cols
            LobpcgTarget::Largest => col_offset, // skip last bs (X) cols (i.e. take first col_offset)
        };
        let p_col_end = match self.config.which {
            LobpcgTarget::Smallest => dim,
            LobpcgTarget::Largest => col_offset,
        };

        let mut p_new: Vec<Vec<f64>> = Vec::new();
        if p_col_end > p_col_start {
            for col in p_col_start..p_col_end {
                if col >= evecs[0].len() {
                    break;
                }
                let mut pj = vec![0.0_f64; n];
                for (si, sv) in s.iter().enumerate() {
                    let coeff = evecs[si][col];
                    for i in 0..n {
                        pj[i] += coeff * sv[i];
                    }
                }
                p_new.push(pj);
            }
        } else if match self.config.which {
            LobpcgTarget::Largest => true,
            LobpcgTarget::Smallest => false,
        } {
            // For Largest: P is everything after col_offset (i.e. 0..col_offset already used for X)
            // Actually re-derive: for Largest, X uses last bs cols, P uses first col_offset cols
            for col in 0..col_offset {
                if col >= evecs[0].len() {
                    break;
                }
                let mut pj = vec![0.0_f64; n];
                for (si, sv) in s.iter().enumerate() {
                    let coeff = evecs[si][col];
                    for i in 0..n {
                        pj[i] += coeff * sv[i];
                    }
                }
                p_new.push(pj);
            }
        }

        *x = x_new;
        *p = p_new;

        // Orthonormalize X for numerical stability
        let x_cols = mgs_orthonormalize(x)?;
        x.truncate(x_cols.max(1));

        Ok(false)
    }
}

// =============================================================================
// Dense symmetric EVD via Jacobi sweeps
// =============================================================================

/// Compute eigenpairs of a symmetric dense matrix via classical Jacobi iteration.
///
/// The matrix `a` is modified in place (diagonal → eigenvalues after convergence).
/// Returns `(eigenvalues, eigenvectors)` with eigenvalues sorted ascending and
/// eigenvectors stored column-major (evecs[row][col]).
fn jacobi_evd(
    a: &mut Vec<Vec<f64>>,
    max_sweeps: usize,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), LobpcgError> {
    let n = a.len();
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    // Initialize eigenvector matrix V = I
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            row
        })
        .collect();

    for _sweep in 0..max_sweeps {
        // Find max off-diagonal
        let mut max_off = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let v_abs = a[i][j].abs();
                if v_abs > max_off {
                    max_off = v_abs;
                }
            }
        }

        // Convergence check: max off-diag < eps * max diag
        let max_diag = (0..n).map(|i| a[i][i].abs()).fold(0.0_f64, f64::max);
        if max_off < 1e-14 * max_diag.max(1e-300) {
            break;
        }

        // Sweep all off-diagonal pairs
        for p in 0..n {
            for q in (p + 1)..n {
                let a_pq = a[p][q];
                if a_pq.abs() < 1e-15 * max_diag.max(1e-300) {
                    continue;
                }
                let a_pp = a[p][p];
                let a_qq = a[q][q];

                // Compute rotation angle
                let tau = (a_qq - a_pp) / (2.0 * a_pq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Update diagonal
                let a_pp_new = a_pp - t * a_pq;
                let a_qq_new = a_qq + t * a_pq;
                a[p][p] = a_pp_new;
                a[q][q] = a_qq_new;
                a[p][q] = 0.0;
                a[q][p] = 0.0;

                // Update off-diagonal rows/cols
                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let a_rp = a[r][p];
                    let a_rq = a[r][q];
                    let new_rp = c * a_rp - s * a_rq;
                    let new_rq = s * a_rp + c * a_rq;
                    a[r][p] = new_rp;
                    a[p][r] = new_rp;
                    a[r][q] = new_rq;
                    a[q][r] = new_rq;
                }

                // Update eigenvector matrix V (columns p and q)
                for r in 0..n {
                    let v_rp = v[r][p];
                    let v_rq = v[r][q];
                    v[r][p] = c * v_rp - s * v_rq;
                    v[r][q] = s * v_rp + c * v_rq;
                }
            }
        }
    }

    // Extract eigenvalues from diagonal
    let mut pairs: Vec<(f64, usize)> = (0..n).map(|i| (a[i][i], i)).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let eigenvalues: Vec<f64> = pairs.iter().map(|(val, _)| *val).collect();
    // evecs[row][col] — col in sorted order
    let mut evecs: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for (new_col, (_, old_col)) in pairs.iter().enumerate() {
        for row in 0..n {
            evecs[row][new_col] = v[row][*old_col];
        }
    }

    Ok((eigenvalues, evecs))
}

// =============================================================================
// Block SpMV
// =============================================================================

/// Apply the CSR matrix A to each column of `x_block`, returning A * x_block.
fn spmv_block(csr: &CsrMatrix<f64>, x_block: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = csr.nrows();
    x_block
        .iter()
        .map(|xj| {
            let mut y = vec![0.0_f64; n];
            spmv(1.0_f64, csr, xj, 0.0_f64, &mut y);
            y
        })
        .collect()
}

// =============================================================================
// Modified Gram-Schmidt orthonormalization
// =============================================================================

/// Orthonormalize a set of vectors in-place using Modified Gram-Schmidt.
///
/// Vectors that become near-zero after projection (norm < threshold) are dropped.
/// Returns the number of surviving linearly-independent vectors.
///
/// # Errors
///
/// Returns `LobpcgError::NumericalFailure` if no vectors survive.
fn mgs_orthonormalize(vecs: &mut Vec<Vec<f64>>) -> Result<usize, LobpcgError> {
    let m = vecs.len();
    if m == 0 {
        return Ok(0);
    }
    let threshold = 1e-13;
    let mut keep = 0usize;

    for i in 0..m {
        // Project out all previously accepted vectors
        for j in 0..keep {
            // Compute dot product of accepted[j] and current[i]
            let dot_ji = {
                let mut d = 0.0_f64;
                let n = vecs[i].len();
                for idx in 0..n {
                    d += vecs[j][idx] * vecs[i][idx];
                }
                d
            };
            let n = vecs[i].len();
            // vecs[i] -= dot_ji * vecs[j] — clone to avoid borrow conflict
            let accepted: Vec<f64> = vecs[j].clone();
            for idx in 0..n {
                vecs[i][idx] -= dot_ji * accepted[idx];
            }
        }

        let nrm = vec_norm(&vecs[i]);
        if nrm < threshold {
            // Linearly dependent — skip this vector
            continue;
        }

        // Normalize
        let inv = 1.0 / nrm;
        for val in vecs[i].iter_mut() {
            *val *= inv;
        }

        // Move to position `keep` if needed
        if keep != i {
            // swap so accepted vectors stay at front
            vecs.swap(keep, i);
        }
        keep += 1;
    }

    if keep == 0 {
        return Err(LobpcgError::NumericalFailure(
            "All vectors became linearly dependent during MGS".to_string(),
        ));
    }

    Ok(keep)
}

// =============================================================================
// Orthogonalize a single vector against a basis
// =============================================================================

/// Project out of `w` all components in span(`basis`), using two-pass CGS for stability.
fn orthogonalize_against(w: &mut Vec<f64>, basis: &[Vec<f64>]) {
    for _pass in 0..2 {
        for bv in basis.iter() {
            let d = vec_dot(w, bv);
            let n = w.len();
            for i in 0..n {
                w[i] -= d * bv[i];
            }
        }
    }
}

// =============================================================================
// Initial eigenvector block
// =============================================================================

/// Create k orthonormal starting vectors of length n.
///
/// Uses the first k standard basis vectors e_0 … e_{k-1} and orthonormalizes.
fn init_eigenvector_block(n: usize, k: usize) -> Result<Vec<Vec<f64>>, LobpcgError> {
    if k == 0 || n == 0 {
        return Err(LobpcgError::InvalidConfig(
            "n and k must both be >= 1".to_string(),
        ));
    }
    let k_actual = k.min(n);
    let mut vecs: Vec<Vec<f64>> = (0..k_actual)
        .map(|j| {
            let mut v = vec![0.0_f64; n];
            v[j] = 1.0;
            v
        })
        .collect();

    // If k > n, pad with random-ish (deterministic) vectors
    // In practice k <= n is enforced by the caller.
    if k > k_actual {
        return Err(LobpcgError::InvalidConfig(format!(
            "k ({k}) must be <= n ({n})"
        )));
    }

    let count = mgs_orthonormalize(&mut vecs)?;
    vecs.truncate(count);
    Ok(vecs)
}

// =============================================================================
// Dense vector helpers
// =============================================================================

#[inline]
fn vec_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn vec_norm(v: &[f64]) -> f64 {
    vec_dot(v, v).sqrt()
}

/// Compute `y - alpha * x` and return as a new Vec (does not modify y).
#[inline]
fn vec_axpy(alpha: f64, x: &[f64], y: &[f64]) -> Vec<f64> {
    y.iter()
        .zip(x.iter())
        .map(|(yi, xi)| yi + alpha * xi)
        .collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csr::CsrMatrix;

    /// Build a 2×2 symmetric matrix [[2, 1], [1, 2]].
    /// Eigenvalues: 1.0 and 3.0.
    fn make_2x2() -> CsrMatrix<f64> {
        let values = vec![2.0, 1.0, 1.0, 2.0];
        let col_indices = vec![0usize, 1, 0, 1];
        let row_ptrs = vec![0usize, 2, 4];
        CsrMatrix::new(2, 2, row_ptrs, col_indices, values).unwrap()
    }

    /// Build an n×n symmetric tridiagonal matrix with diag=2, off-diag=-1.
    fn make_tridiagonal(n: usize) -> CsrMatrix<f64> {
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptrs = vec![0usize];
        for i in 0..n {
            if i > 0 {
                values.push(-1.0_f64);
                col_indices.push(i - 1);
            }
            values.push(2.0_f64);
            col_indices.push(i);
            if i < n - 1 {
                values.push(-1.0_f64);
                col_indices.push(i + 1);
            }
            row_ptrs.push(values.len());
        }
        CsrMatrix::new(n, n, row_ptrs, col_indices, values).unwrap()
    }

    /// Build a 1D Laplacian (same as tridiagonal).
    fn make_laplacian_1d(n: usize) -> CsrMatrix<f64> {
        make_tridiagonal(n)
    }

    #[test]
    fn test_lobpcg_config_default() {
        let config = LobpcgConfig::default();
        assert_eq!(config.num_eigenvalues, 1);
        assert_eq!(config.max_iter, 300);
        assert!((config.tol - 1e-10).abs() < 1e-20);
        assert_eq!(config.which, LobpcgTarget::Smallest);
        assert!(config.block_size.is_none());
    }

    #[test]
    fn test_lobpcg_2x2() {
        let a = make_2x2();
        let config = LobpcgConfig {
            num_eigenvalues: 1,
            which: LobpcgTarget::Smallest,
            tol: 1e-8,
            max_iter: 500,
            ..Default::default()
        };
        let solver = Lobpcg::new(config).unwrap();
        let result = solver.compute(&a).unwrap();
        assert!(!result.eigenvalues.is_empty());
        let lambda = result.eigenvalues[0];
        assert!(
            (lambda - 1.0).abs() < 1e-6,
            "Expected smallest eigenvalue 1.0, got {lambda}"
        );
    }

    #[test]
    fn test_lobpcg_tridiagonal_5() {
        let a = make_tridiagonal(5);
        let config = LobpcgConfig {
            num_eigenvalues: 2,
            which: LobpcgTarget::Smallest,
            tol: 1e-7,
            max_iter: 600,
            ..Default::default()
        };
        let solver = Lobpcg::new(config).unwrap();
        let result = solver.compute(&a).unwrap();
        assert!(
            result.eigenvalues.len() >= 2,
            "Expected 2 eigenvalues, got {}",
            result.eigenvalues.len()
        );
        let lambda0 = result.eigenvalues[0];
        // Smallest eigenvalue of 5×5 tridiagonal ≈ 2*(1-cos(π/6)) ≈ 0.2679
        assert!(
            lambda0 > 0.0 && lambda0 < 1.0,
            "Smallest eigenvalue ~0.268, got {lambda0}"
        );
    }

    #[test]
    fn test_lobpcg_laplacian_1d() {
        let a = make_laplacian_1d(20);
        let config = LobpcgConfig {
            num_eigenvalues: 3,
            which: LobpcgTarget::Smallest,
            tol: 1e-7,
            max_iter: 600,
            ..Default::default()
        };
        let solver = Lobpcg::new(config).unwrap();
        let result = solver.compute(&a).unwrap();
        assert!(
            result.eigenvalues.len() >= 3,
            "Expected 3 eigenvalues, got {}",
            result.eigenvalues.len()
        );
        for &lam in &result.eigenvalues {
            assert!(lam > 0.0, "All eigenvalues should be positive, got {lam}");
        }
        let lambda0 = result.eigenvalues[0];
        // Smallest ≈ 2*(1-cos(π/21)) ≈ 0.02227
        assert!(
            lambda0 < 0.1,
            "Smallest eigenvalue should be < 0.1, got {lambda0}"
        );
    }

    #[test]
    fn test_lobpcg_largest() {
        let a = make_tridiagonal(10);
        let config = LobpcgConfig {
            num_eigenvalues: 2,
            which: LobpcgTarget::Largest,
            tol: 1e-7,
            max_iter: 600,
            ..Default::default()
        };
        let solver = Lobpcg::new(config).unwrap();
        let result = solver.compute(&a).unwrap();
        assert!(
            result.eigenvalues.len() >= 2,
            "Expected 2 eigenvalues, got {}",
            result.eigenvalues.len()
        );
        let lambda_max = result.eigenvalues[0];
        // Largest eigenvalue of 10×10 tridiagonal ≈ 2*(1+cos(π/11)) ≈ 3.919
        assert!(
            lambda_max > 3.0,
            "Largest eigenvalue should be > 3.0, got {lambda_max}"
        );
    }

    #[test]
    fn test_lobpcg_preconditioned() {
        // Diagonal preconditioner: w[i] = r[i] / 2.0 (diagonal of tridiagonal is 2.0)
        let a = make_tridiagonal(10);
        let config = LobpcgConfig {
            num_eigenvalues: 2,
            which: LobpcgTarget::Smallest,
            tol: 1e-7,
            max_iter: 600,
            ..Default::default()
        };
        let solver = Lobpcg::new(config).unwrap();
        let result = solver
            .compute_preconditioned(&a, |r, w| {
                for (wi, ri) in w.iter_mut().zip(r.iter()) {
                    *wi = ri / 2.0;
                }
            })
            .unwrap();
        assert!(
            result.eigenvalues.len() >= 2,
            "Expected 2 eigenvalues, got {}",
            result.eigenvalues.len()
        );
        let lambda0 = result.eigenvalues[0];
        // Smallest eigenvalue of 10×10 tridiagonal ≈ 0.0955
        assert!(
            lambda0 > 0.0 && lambda0 < 1.0,
            "Smallest eigenvalue ~ 0.0955, got {lambda0}"
        );
    }
}
