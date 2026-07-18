//! Real bidiagonal SVD via divide-and-conquer.
//!
//! This module implements the singular value decomposition of an upper
//! bidiagonal matrix `B` (diagonal `d`, super-diagonal `e`).  It is shared by
//! both the real ([`crate::svd::SvdDc`]) and complex ([`crate::svd::ComplexSvdDc`])
//! divide-and-conquer front-ends: complex bidiagonalization produces a *real*
//! bidiagonal matrix, so exactly the same real kernel serves both.
//!
//! # Algorithm
//!
//! For small problems (`n <= DIRECT_THRESHOLD`) the SVD is computed directly by
//! the implicit-shift (Golub–Kahan) bidiagonal QR sweep with a Wilkinson shift.
//!
//! For larger problems the SVD is obtained by a genuine divide-and-conquer
//! solve of the symmetric tridiagonal eigenproblem `T = Bᵀ·B`:
//!
//! 1. **Divide** – `T` is split at its midpoint by removing the coupling
//!    off-diagonal `β`, writing `T = T̃₁ ⊕ T̃₂ + |β|·u·uᵀ` (Cuppen's rank-one
//!    modification, with the two boundary diagonal entries shifted by `|β|`).
//! 2. **Conquer** – the two halves are solved recursively, yielding the merged
//!    diagonal `D` (their eigenvalues) and block-diagonal eigenvector matrix
//!    `Q = diag(Q₁, Q₂)`.
//! 3. **Deflate** – eigenpairs whose rank-one coupling component `uᵢ` is
//!    negligible, and numerically-coincident eigenvalues, are deflated: they
//!    pass through unchanged, contributing their (trivial) eigenvector directly.
//! 4. **Secular equation** – the remaining eigenvalues solve
//!    `f(λ) = 1 + |β|·Σᵢ uᵢ²/(dᵢ − λ) = 0`, found by safeguarded
//!    Newton–Raphson with bisection fall-back.  The associated eigenvectors are
//!    `(D − λI)⁻¹u`, normalised, then rotated back through `Q`.
//!
//! The right singular vectors of `B` are the eigenvectors `V` of `T`; the
//! corresponding singular values are `σᵢ = ‖B·vᵢ‖`, and the left singular
//! vectors are `uᵢ = B·vᵢ / σᵢ`.  Because `V` is orthonormal this reconstructs
//! `B = U·Σ·Vᵀ` to working precision.
//!
//! # References
//!
//! - Gu, M. & Eisenstat, S. C. (1995). *A Divide-and-Conquer Algorithm for the
//!   Bidiagonal SVD*. SIAM J. Matrix Anal. Appl., 16(1), 79–92.
//! - Cuppen, J. J. M. (1981). *A divide and conquer method for the symmetric
//!   tridiagonal eigenproblem*. Numer. Math., 36(2), 177–195.

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::Mat;

/// Error returned by the real bidiagonal divide-and-conquer SVD kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BidiagDcError {
    /// An iterative sub-solver (bidiagonal QR or Jacobi base case) failed to
    /// converge within its iteration budget.
    NotConverged,
    /// The secular-equation solver produced an inconsistent result.
    SecularEquationFailed,
}

/// Below this size the bidiagonal SVD is computed directly by shifted QR.
const DIRECT_THRESHOLD: usize = 25;
/// Below this size a tridiagonal block is solved directly by Jacobi rotations.
const CUPPEN_BASE: usize = 8;
/// Maximum bidiagonal QR sweeps allowed (per diagonal entry).
const MAX_QR_SWEEPS: usize = 40;
/// Maximum Newton iterations for a single secular-equation root.
const MAX_SECULAR_ITER: usize = 80;
/// Maximum Jacobi sweeps for a base-case tridiagonal block.
const MAX_JACOBI_SWEEPS: usize = 100;

#[inline]
fn fabs<R: Real>(x: R) -> R {
    if x < R::zero() {
        -x
    } else {
        x
    }
}

#[inline]
fn from_f64<R: Real>(v: f64) -> R {
    R::from_f64(v).unwrap_or_else(R::zero)
}

#[inline]
fn rsqrt<R: Real>(x: R) -> R {
    <R as Real>::sqrt(x)
}

/// Computes the SVD of the upper bidiagonal matrix with diagonal `d` and
/// super-diagonal `e` (`e.len() == d.len() - 1`).
///
/// Returns `(U, sigma, Vt)` where `U` (`n×n`) and `Vt` (`n×n`) are orthogonal
/// and `sigma` holds the `n` non-negative singular values in descending order,
/// so that `B = U · diag(sigma) · Vt`.
pub(crate) fn bidiagonal_svd_dc<R>(
    d: &[R],
    e: &[R],
) -> Result<(Mat<R>, Vec<R>, Mat<R>), BidiagDcError>
where
    R: Field + Real + bytemuck::Zeroable,
{
    let n = d.len();

    if n == 0 {
        return Ok((Mat::zeros(0, 0), Vec::new(), Mat::zeros(0, 0)));
    }

    if n == 1 {
        let sigma = vec![fabs(d[0])];
        let mut u = Mat::zeros(1, 1);
        let mut vt = Mat::zeros(1, 1);
        u[(0, 0)] = if d[0] >= R::zero() {
            R::one()
        } else {
            -R::one()
        };
        vt[(0, 0)] = R::one();
        return Ok((u, sigma, vt));
    }

    if n <= DIRECT_THRESHOLD {
        return bidiagonal_svd_qr(d, e);
    }

    // Divide-and-conquer on the symmetric tridiagonal T = Bᵀ·B.
    //   T[i,i]   = d[i]² + e[i-1]²   (e[-1] ≡ 0)
    //   T[i,i+1] = d[i] · e[i]
    let mut t_diag = vec![R::zero(); n];
    let mut t_off = vec![R::zero(); n - 1];
    for i in 0..n {
        let mut diag = d[i] * d[i];
        if i > 0 {
            diag = diag + e[i - 1] * e[i - 1];
        }
        t_diag[i] = diag;
    }
    for i in 0..n - 1 {
        t_off[i] = d[i] * e[i];
    }

    let (evals, vmat) = cuppen_tridiag(&t_diag, &t_off)?;
    assemble_svd(d, e, &evals, vmat)
}

/// Builds the bidiagonal SVD from the eigen-decomposition `T = V·diag(evals)·Vᵀ`.
///
/// `evals` are the eigenvalues of `T = BᵀB` (i.e. `σ²`) and `vmat` holds the
/// corresponding orthonormal eigenvectors as columns.  The singular values are
/// recovered as `σⱼ = ‖B·vⱼ‖` (a Rayleigh quotient, more accurate than
/// `√evalⱼ` for reconstruction) and the left vectors as `uⱼ = B·vⱼ/σⱼ`.
fn assemble_svd<R>(
    d: &[R],
    e: &[R],
    evals: &[R],
    mut vmat: Mat<R>,
) -> Result<(Mat<R>, Vec<R>, Mat<R>), BidiagDcError>
where
    R: Field + Real + bytemuck::Zeroable,
{
    let n = d.len();

    // The eigen-solver must return exactly one eigenpair per column.
    if evals.len() != n || vmat.nrows() != n || vmat.ncols() != n {
        return Err(BidiagDcError::SecularEquationFailed);
    }

    // Guarantee V is orthonormal to working precision before deriving U from it.
    mgs_orthonormalize(&mut vmat);

    // w_j = B · v_j and σ_j = ‖w_j‖.
    let mut w = Mat::zeros(n, n);
    let mut sigma_col = vec![R::zero(); n];
    for j in 0..n {
        let mut norm_sq = R::zero();
        for i in 0..n {
            let mut bv = d[i] * vmat[(i, j)];
            if i + 1 < n {
                bv = bv + e[i] * vmat[(i + 1, j)];
            }
            w[(i, j)] = bv;
            norm_sq = norm_sq + bv * bv;
        }
        sigma_col[j] = rsqrt(norm_sq);
    }

    // Order columns by descending singular value.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        sigma_col[b]
            .partial_cmp(&sigma_col[a])
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    let scale = sigma_col.iter().copied().fold(R::zero(), |m, s| m.max(s));
    let eps = <R as Scalar>::epsilon();
    let tiny = eps * scale.max(R::one()) * from_f64::<R>(8.0);

    let mut sigma = vec![R::zero(); n];
    let mut u = Mat::zeros(n, n);
    let mut vt = Mat::zeros(n, n);
    let mut needs_fill = Vec::new();

    for (new_j, &old_j) in order.iter().enumerate() {
        sigma[new_j] = sigma_col[old_j];
        for i in 0..n {
            vt[(new_j, i)] = vmat[(i, old_j)];
        }
        if sigma_col[old_j] > tiny {
            let inv = R::one() / sigma_col[old_j];
            for i in 0..n {
                u[(i, new_j)] = w[(i, old_j)] * inv;
            }
        } else {
            needs_fill.push(new_j);
        }
    }

    // For (near-)zero singular values, B·v carries no reliable direction, so
    // complete U with an orthonormal basis of the remaining left null-space.
    if !needs_fill.is_empty() {
        fill_orthonormal_columns(&mut u, &needs_fill);
    }

    Ok((u, sigma, vt))
}

// ---------------------------------------------------------------------------
// Direct bidiagonal SVD by implicit-shift (Golub–Kahan) QR
// ---------------------------------------------------------------------------

/// Computes the SVD of a small upper bidiagonal matrix by the implicit-shift
/// Golub–Kahan QR algorithm (LAPACK `dbdsqr` style) with a Wilkinson shift.
///
/// Returns `Err(BidiagDcError::NotConverged)` if the sweep budget is exhausted
/// before every super-diagonal entry is negligible.
fn bidiagonal_svd_qr<R>(d: &[R], e: &[R]) -> Result<(Mat<R>, Vec<R>, Mat<R>), BidiagDcError>
where
    R: Field + Real + bytemuck::Zeroable,
{
    let n = d.len();
    let mut d_work: Vec<R> = d.to_vec();
    let mut e_work: Vec<R> = e.to_vec();

    let mut u = Mat::zeros(n, n);
    let mut vt = Mat::zeros(n, n);
    for i in 0..n {
        u[(i, i)] = R::one();
        vt[(i, i)] = R::one();
    }

    let eps = <R as Scalar>::epsilon();
    let tol = eps * from_f64::<R>(4.0);

    let max_iter = MAX_QR_SWEEPS * n + 20;
    let mut converged = false;

    for _iter in 0..max_iter {
        // Deflate: zero-out negligible super-diagonals.
        for i in 0..e_work.len() {
            let thresh = tol * (fabs(d_work[i]) + fabs(d_work[i + 1]));
            if fabs(e_work[i]) <= thresh {
                e_work[i] = R::zero();
            }
        }

        // Find the bottom `p` of the lowest unreduced block: the largest index
        // with e[p-1] != 0.
        let mut p = e_work.len();
        while p > 0 && e_work[p - 1] == R::zero() {
            p -= 1;
        }
        if p == 0 {
            converged = true;
            break;
        }

        // Find the top `q` of that block: the smallest index such that all
        // e[q..p] are non-zero.
        let mut q = p - 1;
        while q > 0 && e_work[q - 1] != R::zero() {
            q -= 1;
        }

        golub_kahan_step(&mut d_work, &mut e_work, &mut u, &mut vt, q, p + 1);
    }

    if !converged {
        return Err(BidiagDcError::NotConverged);
    }

    // Make singular values non-negative (flip the matching left vector).
    for i in 0..n {
        if d_work[i] < R::zero() {
            d_work[i] = -d_work[i];
            for r in 0..n {
                u[(r, i)] = -u[(r, i)];
            }
        }
    }

    // Sort descending.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        d_work[b]
            .partial_cmp(&d_work[a])
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    let mut sigma = vec![R::zero(); n];
    let mut u_sorted = Mat::zeros(n, n);
    let mut vt_sorted = Mat::zeros(n, n);
    for (new_idx, &old_idx) in order.iter().enumerate() {
        sigma[new_idx] = d_work[old_idx];
        for r in 0..n {
            u_sorted[(r, new_idx)] = u[(r, old_idx)];
            vt_sorted[(new_idx, r)] = vt[(old_idx, r)];
        }
    }

    Ok((u_sorted, sigma, vt_sorted))
}

/// One implicit-shift Golub–Kahan sweep over the unreduced block `[start, end)`
/// of the bidiagonal matrix, accumulating the rotations into `u` and `vt`.
fn golub_kahan_step<R>(
    d: &mut [R],
    e: &mut [R],
    u: &mut Mat<R>,
    vt: &mut Mat<R>,
    start: usize,
    end: usize,
) where
    R: Field + Real + bytemuck::Zeroable,
{
    let n = u.nrows();
    let last = end - 1;

    // Wilkinson shift: eigenvalue of the trailing 2×2 of Bᵀ·B nearest t22.
    let d_last = d[last];
    let d_prev = d[last - 1];
    let e_prev = e[last - 1];
    let e_prev2 = if last >= start + 2 {
        e[last - 2]
    } else {
        R::zero()
    };

    let t11 = d_prev * d_prev + e_prev2 * e_prev2;
    let t22 = d_last * d_last + e_prev * e_prev;
    let t12 = d_prev * e_prev;

    let shift = wilkinson_shift(t11, t12, t22);

    // Chase the bulge.
    let mut f = d[start] * d[start] - shift;
    let mut g = d[start] * e[start];

    for k in start..last {
        let (c, s, r) = givens(f, g);
        if k > start {
            e[k - 1] = r;
        }

        f = c * d[k] + s * e[k];
        e[k] = c * e[k] - s * d[k];
        g = s * d[k + 1];
        d[k + 1] = c * d[k + 1];

        // Right rotation acts on rows k, k+1 of Vᵀ.
        for j in 0..n {
            let vk = vt[(k, j)];
            let vk1 = vt[(k + 1, j)];
            vt[(k, j)] = c * vk + s * vk1;
            vt[(k + 1, j)] = c * vk1 - s * vk;
        }

        let (c, s, r) = givens(f, g);
        d[k] = r;
        f = c * e[k] + s * d[k + 1];
        d[k + 1] = c * d[k + 1] - s * e[k];
        if k < last - 1 {
            g = s * e[k + 1];
            e[k + 1] = c * e[k + 1];
        }

        // Left rotation acts on columns k, k+1 of U.
        for i in 0..n {
            let uk = u[(i, k)];
            let uk1 = u[(i, k + 1)];
            u[(i, k)] = c * uk + s * uk1;
            u[(i, k + 1)] = c * uk1 - s * uk;
        }
    }

    e[last - 1] = f;
}

/// Wilkinson shift for the symmetric 2×2 `[[t11, t12], [t12, t22]]`: the
/// eigenvalue nearest `t22`.
#[inline]
fn wilkinson_shift<R: Real>(t11: R, t12: R, t22: R) -> R {
    if t12 == R::zero() {
        return t22;
    }
    let two = from_f64::<R>(2.0);
    let delta = (t11 - t22) / two;
    let sign = if delta < R::zero() { -R::one() } else { R::one() };
    // λ = t22 − t12² / (δ + sign(δ)·√(δ² + t12²))
    let denom = delta + sign * rsqrt(delta * delta + t12 * t12);
    if denom == R::zero() {
        t22
    } else {
        t22 - (t12 * t12) / denom
    }
}

/// Givens rotation zeroing `g`: returns `(c, s, r)` with
/// `[c s; -s c]·[f; g] = [r; 0]`.
#[inline]
fn givens<R: Field + Real>(f: R, g: R) -> (R, R, R) {
    if g == R::zero() {
        (R::one(), R::zero(), f)
    } else if f == R::zero() {
        let sign = if g < R::zero() { -R::one() } else { R::one() };
        (R::zero(), sign, fabs(g))
    } else {
        let r = <R as Real>::hypot(f, g);
        (f / r, g / r, r)
    }
}

// ---------------------------------------------------------------------------
// Cuppen divide-and-conquer for the symmetric tridiagonal eigenproblem
// ---------------------------------------------------------------------------

/// Eigen-decomposition of a symmetric tridiagonal matrix by Cuppen's D&C.
///
/// Returns `(evals, V)` with `evals` sorted ascending and `V` holding the
/// orthonormal eigenvectors as columns (`V[(i, k)]` is component `i` of the
/// `k`-th eigenvector).
fn cuppen_tridiag<R>(diag: &[R], off: &[R]) -> Result<(Vec<R>, Mat<R>), BidiagDcError>
where
    R: Field + Real + bytemuck::Zeroable,
{
    let (evals, evecs) = cuppen_impl(diag, off)?;
    let n = evals.len();
    let mut v = Mat::zeros(n, n);
    for (k, col) in evecs.iter().enumerate() {
        for (i, &val) in col.iter().enumerate() {
            v[(i, k)] = val;
        }
    }
    Ok((evals, v))
}

/// Recursive core.  Eigenvectors are returned as `evecs[k]` = `k`-th eigenvector.
fn cuppen_impl<R>(diag: &[R], off: &[R]) -> Result<(Vec<R>, Vec<Vec<R>>), BidiagDcError>
where
    R: Field + Real + bytemuck::Zeroable,
{
    let n = diag.len();

    if n == 1 {
        return Ok((vec![diag[0]], vec![vec![R::one()]]));
    }
    if n <= CUPPEN_BASE {
        return jacobi_tridiag(diag, off);
    }

    let mid = n / 2;
    let beta = off[mid - 1];
    let abs_beta = fabs(beta);

    let scale = tridiag_scale(diag, off);
    let eps = <R as Scalar>::epsilon();

    // Decoupled case: negligible coupling ⇒ solve the two halves independently.
    if abs_beta <= eps * scale {
        let (ev1, evec1) = cuppen_impl(&diag[..mid], &off[..mid - 1])?;
        let (ev2, evec2) = cuppen_impl(&diag[mid..], &off[mid..])?;
        return Ok(combine_block_diag(ev1, evec1, ev2, evec2, mid, n));
    }

    // Rank-one modification: T = T̃₁ ⊕ T̃₂ + |β|·u·uᵀ.
    let mut diag1 = diag[..mid].to_vec();
    let mut diag2 = diag[mid..].to_vec();
    diag1[mid - 1] = diag1[mid - 1] - abs_beta;
    diag2[0] = diag2[0] - abs_beta;

    let off1 = &off[..mid - 1];
    let off2 = &off[mid..];

    let (evals1, evecs1) = cuppen_impl(&diag1, off1)?;
    let (evals2, evecs2) = cuppen_impl(&diag2, off2)?;

    // Merged diagonal d = [d₁; d₂].
    let d: Vec<R> = evals1.iter().chain(evals2.iter()).copied().collect();

    // Rank-one vector u = Qᵀ·z, z = sign(β)·e_{mid-1} + e_mid.
    let sign_beta = if beta < R::zero() {
        -R::one()
    } else {
        R::one()
    };
    let mut u = Vec::with_capacity(n);
    for ev1 in evecs1.iter().take(mid) {
        u.push(sign_beta * ev1[mid - 1]);
    }
    for ev2 in evecs2.iter().take(n - mid) {
        u.push(ev2[0]);
    }

    let deflation_tol = eps * from_f64::<R>(n as f64) * scale.max(R::one());
    let (d_defl, u_defl, active_idx, trivial_evals) = deflate(&d, &u, deflation_tol);

    // Solve the secular equation for the active part and build its eigenvectors
    // with the Gu–Eisenstat stable (Löwner) formula, which stays orthogonal even
    // for tightly-clustered eigenvalues.
    let (secular_evals, secular_evecs) = if d_defl.is_empty() {
        (Vec::new(), Vec::new())
    } else {
        secular_eigenpairs(&d_defl, &u_defl, abs_beta)?
    };

    let total = trivial_evals.len() + secular_evals.len();
    if total != n {
        // Deflation bookkeeping disagreed with the split; fall back to a direct
        // (always-correct) Jacobi solve of the original block.
        return jacobi_tridiag(diag, off);
    }

    // Assemble (eigenvalue, D-space eigenvector) pairs.
    let mut pairs: Vec<(R, Vec<R>)> = Vec::with_capacity(n);

    let trivial_positions: Vec<usize> = (0..n).filter(|i| !active_idx.contains(i)).collect();
    for (&eval, &pos) in trivial_evals.iter().zip(trivial_positions.iter()) {
        let mut ev = vec![R::zero(); n];
        ev[pos] = R::one();
        pairs.push((eval, ev));
    }

    for (lam, evec_active) in secular_evals.into_iter().zip(secular_evecs) {
        let mut ev = vec![R::zero(); n];
        for (k, &ai) in active_idx.iter().enumerate() {
            ev[ai] = evec_active[k];
        }
        pairs.push((lam, ev));
    }

    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));

    // Back-transform through Q = diag(Q₁, Q₂).
    let n2 = n - mid;
    let mut evals_final = Vec::with_capacity(n);
    let mut evecs_final = Vec::with_capacity(n);
    for (lam, v_d) in pairs {
        evals_final.push(lam);
        let mut out = vec![R::zero(); n];
        for (k, evec1) in evecs1.iter().enumerate().take(mid) {
            let x = v_d[k];
            if x != R::zero() {
                for (i, &c) in evec1.iter().enumerate().take(mid) {
                    out[i] = out[i] + x * c;
                }
            }
        }
        for (k, evec2) in evecs2.iter().enumerate().take(n2) {
            let x = v_d[mid + k];
            if x != R::zero() {
                for (i, &c) in evec2.iter().enumerate().take(n2) {
                    out[mid + i] = out[mid + i] + x * c;
                }
            }
        }
        evecs_final.push(out);
    }

    Ok((evals_final, evecs_final))
}

/// Combines two independent (decoupled) sub-solutions block-diagonally.
#[allow(clippy::type_complexity)]
fn combine_block_diag<R: Real>(
    ev1: Vec<R>,
    evec1: Vec<Vec<R>>,
    ev2: Vec<R>,
    evec2: Vec<Vec<R>>,
    mid: usize,
    n: usize,
) -> (Vec<R>, Vec<Vec<R>>) {
    let n2 = n - mid;
    let mut pairs: Vec<(R, Vec<R>)> = Vec::with_capacity(n);
    for (k, lam) in ev1.into_iter().enumerate() {
        let mut ev = vec![R::zero(); n];
        for i in 0..mid {
            ev[i] = evec1[k][i];
        }
        pairs.push((lam, ev));
    }
    for (k, lam) in ev2.into_iter().enumerate() {
        let mut ev = vec![R::zero(); n];
        for i in 0..n2 {
            ev[mid + i] = evec2[k][i];
        }
        pairs.push((lam, ev));
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
    let evals = pairs.iter().map(|p| p.0).collect();
    let evecs = pairs.into_iter().map(|p| p.1).collect();
    (evals, evecs)
}

/// Estimate of the magnitude scale of a symmetric tridiagonal matrix.
fn tridiag_scale<R: Real>(diag: &[R], off: &[R]) -> R {
    let mut s = R::zero();
    for &x in diag {
        s = s.max(fabs(x));
    }
    for &x in off {
        s = s.max(fabs(x));
    }
    if s > R::zero() {
        s
    } else {
        R::one()
    }
}

// ---------------------------------------------------------------------------
// Deflation and secular equation
// ---------------------------------------------------------------------------

/// Deflates the rank-one modification: drops eigenpairs with negligible
/// coupling `|uᵢ|` and merges numerically-coincident `dᵢ`.
///
/// Returns `(deflated_d, deflated_u, active_indices, trivial_evals)`.
#[allow(clippy::type_complexity)]
fn deflate<R: Real>(d: &[R], u: &[R], tol: R) -> (Vec<R>, Vec<R>, Vec<usize>, Vec<R>) {
    let n = d.len();
    let mut u_norm_sq = R::zero();
    for &x in u {
        u_norm_sq = u_norm_sq + x * x;
    }
    let u_norm = rsqrt(u_norm_sq).max(R::min_positive());

    let mut defl_d = Vec::new();
    let mut defl_u = Vec::new();
    let mut active_idx = Vec::new();
    let mut trivial = Vec::new();

    for i in 0..n {
        if fabs(u[i]) < tol * u_norm {
            trivial.push(d[i]);
        } else {
            defl_d.push(d[i]);
            defl_u.push(u[i]);
            active_idx.push(i);
        }
    }

    // Merge nearly-equal d among the active components (Givens rotation in the
    // exact algorithm; here the coincident eigenvalue deflates out directly).
    let mut i = 0;
    while i < defl_d.len() {
        let mut j = i + 1;
        while j < defl_d.len() {
            if fabs(defl_d[j] - defl_d[i]) < tol * (fabs(defl_d[i]) + R::one()) {
                let merged = rsqrt(defl_u[i] * defl_u[i] + defl_u[j] * defl_u[j]);
                defl_u[i] = merged;
                trivial.push(defl_d[j]);
                defl_d.remove(j);
                defl_u.remove(j);
                active_idx.remove(j);
            } else {
                j += 1;
            }
        }
        i += 1;
    }

    (defl_d, defl_u, active_idx, trivial)
}

/// Solves `1 + β·Σᵢ uᵢ²/(dᵢ − λ) = 0` (`β = |β| > 0`) for all `m` eigenvalues of
/// `D + β·u·uᵀ` and their eigenvectors.
///
/// Returns `(evals, evecs)` where `evals` is ascending and `evecs[k]` is the
/// `k`-th eigenvector in the *input* ordering of `d`/`u`.  Eigenvectors are
/// formed from Gu–Eisenstat's stable weights
/// `ŵᵢ² = ∏ₖ(λₖ − dᵢ) / ∏_{k≠i}(dₖ − dᵢ)`, so that `vₖ[i] = ŵᵢ/(dᵢ − λₖ)`
/// (normalised) is numerically orthogonal even for clustered eigenvalues.
#[allow(clippy::type_complexity)]
fn secular_eigenpairs<R>(
    d: &[R],
    u: &[R],
    beta: R,
) -> Result<(Vec<R>, Vec<Vec<R>>), BidiagDcError>
where
    R: Field + Real,
{
    let m = d.len();
    if m == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    // Sort d ascending, permuting u alongside; remember the inverse map.
    let mut idx: Vec<usize> = (0..m).collect();
    idx.sort_by(|&a, &b| d[a].partial_cmp(&d[b]).unwrap_or(core::cmp::Ordering::Equal));
    let d_s: Vec<R> = idx.iter().map(|&i| d[i]).collect();
    let u_s: Vec<R> = idx.iter().map(|&i| u[i]).collect();

    // Eigenvalues (ascending): root i lies in (d_i, d_{i+1}); the top one above.
    let mut weight_sum = R::zero();
    for &ui in &u_s {
        weight_sum = weight_sum + ui * ui;
    }
    let mut lam = vec![R::zero(); m];
    for i in 0..m {
        let (lo, hi) = if i < m - 1 {
            (d_s[i], d_s[i + 1])
        } else {
            (d_s[m - 1], d_s[m - 1] + beta * weight_sum + R::one())
        };
        lam[i] = find_secular_root(&d_s, &u_s, beta, lo, hi)?;
    }
    lam.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

    // Stable Löwner weights ŵ (up to a global scale, which cancels on
    // normalisation).  The interleaved product keeps magnitudes bounded.
    let mut w_hat = vec![R::zero(); m];
    for i in 0..m {
        let mut prod = R::one();
        for k in 0..m {
            prod = prod * (lam[k] - d_s[i]);
            if k != i {
                prod = prod / (d_s[k] - d_s[i]);
            }
        }
        let mag = rsqrt(prod.max(R::zero()));
        let sign = if u_s[i] < R::zero() {
            -R::one()
        } else {
            R::one()
        };
        w_hat[i] = sign * mag;
    }

    // Eigenvectors in sorted order, then permuted back to the input ordering.
    let mut evecs = vec![vec![R::zero(); m]; m];
    for k in 0..m {
        let mut col = vec![R::zero(); m];
        let mut norm_sq = R::zero();
        for i in 0..m {
            let denom = d_s[i] - lam[k];
            let val = if fabs(denom) < R::min_positive() {
                let s = if w_hat[i] < R::zero() {
                    -R::one()
                } else {
                    R::one()
                };
                s / R::min_positive()
            } else {
                w_hat[i] / denom
            };
            col[i] = val;
            norm_sq = norm_sq + val * val;
        }
        let norm = rsqrt(norm_sq);
        let inv = if norm > R::min_positive() {
            R::one() / norm
        } else {
            R::one()
        };
        for i in 0..m {
            evecs[k][idx[i]] = col[i] * inv;
        }
    }

    Ok((lam, evecs))
}

/// Evaluates `f(λ) = 1 + β·Σ uᵢ²/(dᵢ − λ)` and `f'(λ) = β·Σ uᵢ²/(dᵢ − λ)²`.
#[inline]
fn secular_f_df<R: Field + Real>(d: &[R], u: &[R], beta: R, lam: R) -> (R, R) {
    let mut f = R::one();
    let mut df = R::zero();
    for (&di, &ui) in d.iter().zip(u.iter()) {
        let t = di - lam;
        let w = ui * ui / t;
        f = f + beta * w;
        df = df + beta * w / t;
    }
    (f, df)
}

/// Finds the single root of the secular equation in `(lo, hi)` by safeguarded
/// Newton–Raphson with bisection fall-back.
///
/// On each interval `(dᵢ, dᵢ₊₁)` (with `β > 0`) `f` increases monotonically from
/// `−∞` to `+∞`, so `f(lo) < 0 < f(hi)`; the bracket is maintained purely from
/// the sign of `f(x)`.
fn find_secular_root<R>(d: &[R], u: &[R], beta: R, lo: R, hi: R) -> Result<R, BidiagDcError>
where
    R: Field + Real,
{
    let eps = <R as Scalar>::epsilon();
    let two = from_f64::<R>(2.0);
    let margin = eps * (fabs(lo) + fabs(hi) + R::one()) * from_f64::<R>(4.0);

    let mut lo_m = lo + margin;
    let mut hi_m = hi - margin;
    if hi_m <= lo_m {
        return Ok((lo + hi) / two);
    }

    let conv = eps * from_f64::<R>(8.0);
    let mut x = (lo_m + hi_m) / two;

    for _ in 0..MAX_SECULAR_ITER {
        let (fx, dfx) = secular_f_df(d, u, beta, x);
        if !fx.is_finite() {
            x = (lo_m + hi_m) / two;
            continue;
        }

        // f is increasing: f < 0 ⇒ root is to the right, f > 0 ⇒ to the left.
        if fx < R::zero() {
            lo_m = x;
        } else if fx > R::zero() {
            hi_m = x;
        } else {
            return Ok(x);
        }

        let newton_ok = dfx > R::min_positive();
        let x_new = if newton_ok { x - fx / dfx } else { x };

        let next = if newton_ok && x_new > lo_m && x_new < hi_m {
            x_new
        } else {
            (lo_m + hi_m) / two
        };

        let step = fabs(next - x);
        x = next;
        if step < conv * (fabs(x) + R::one()) || hi_m - lo_m < conv * (fabs(x) + R::one()) {
            return Ok(x);
        }
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Base-case tridiagonal eigensolver (Jacobi)
// ---------------------------------------------------------------------------

/// Eigen-decomposition of a small symmetric tridiagonal matrix by cyclic
/// Jacobi rotations on its dense symmetric form.  Returns eigenvalues ascending
/// and eigenvectors as columns (`evecs[k]` = `k`-th eigenvector).
fn jacobi_tridiag<R>(diag: &[R], off: &[R]) -> Result<(Vec<R>, Vec<Vec<R>>), BidiagDcError>
where
    R: Field + Real + bytemuck::Zeroable,
{
    let n = diag.len();
    if n == 1 {
        return Ok((vec![diag[0]], vec![vec![R::one()]]));
    }

    let mut a = Mat::zeros(n, n);
    for i in 0..n {
        a[(i, i)] = diag[i];
    }
    for i in 0..n - 1 {
        a[(i, i + 1)] = off[i];
        a[(i + 1, i)] = off[i];
    }

    // q[(i, k)] accumulates the k-th eigenvector.
    let mut q = Mat::zeros(n, n);
    for i in 0..n {
        q[(i, i)] = R::one();
    }

    let eps = <R as Scalar>::epsilon();
    let scale = tridiag_scale(diag, off);
    let tol = eps * scale * from_f64::<R>(n as f64);
    let two = from_f64::<R>(2.0);

    let mut converged = false;
    for _sweep in 0..MAX_JACOBI_SWEEPS {
        let mut off_norm = R::zero();
        for i in 0..n {
            for j in (i + 1)..n {
                off_norm = off_norm + a[(i, j)] * a[(i, j)];
            }
        }
        if rsqrt(off_norm) <= tol {
            converged = true;
            break;
        }

        for p in 0..n {
            for qcol in (p + 1)..n {
                let apq = a[(p, qcol)];
                if apq == R::zero() {
                    continue;
                }
                let app = a[(p, p)];
                let aqq = a[(qcol, qcol)];
                let theta = (aqq - app) / (two * apq);
                let sign_t = if theta < R::zero() {
                    -R::one()
                } else {
                    R::one()
                };
                let t = sign_t / (fabs(theta) + rsqrt(theta * theta + R::one()));
                let c = R::one() / rsqrt(t * t + R::one());
                let s = t * c;

                // Rotate rows/cols p, qcol.
                for k in 0..n {
                    let akp = a[(k, p)];
                    let akq = a[(k, qcol)];
                    a[(k, p)] = c * akp - s * akq;
                    a[(k, qcol)] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[(p, k)];
                    let aqk = a[(qcol, k)];
                    a[(p, k)] = c * apk - s * aqk;
                    a[(qcol, k)] = s * apk + c * aqk;
                }
                a[(p, qcol)] = R::zero();
                a[(qcol, p)] = R::zero();

                for k in 0..n {
                    let qkp = q[(k, p)];
                    let qkq = q[(k, qcol)];
                    q[(k, p)] = c * qkp - s * qkq;
                    q[(k, qcol)] = s * qkp + c * qkq;
                }
            }
        }
    }

    if !converged {
        return Err(BidiagDcError::NotConverged);
    }

    let mut pairs: Vec<(R, Vec<R>)> = (0..n)
        .map(|k| {
            let col: Vec<R> = (0..n).map(|i| q[(i, k)]).collect();
            (a[(k, k)], col)
        })
        .collect();
    pairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(core::cmp::Ordering::Equal));

    let evals = pairs.iter().map(|p| p.0).collect();
    let evecs = pairs.into_iter().map(|p| p.1).collect();
    Ok((evals, evecs))
}

// ---------------------------------------------------------------------------
// Orthonormalisation helpers
// ---------------------------------------------------------------------------

/// Modified Gram–Schmidt orthonormalisation of the columns of `mat`, replacing
/// any (numerically) dependent column by an orthonormal complement direction.
fn mgs_orthonormalize<R>(mat: &mut Mat<R>)
where
    R: Field + Real + bytemuck::Zeroable,
{
    let m = mat.nrows();
    let n = mat.ncols();
    let eps = <R as Scalar>::epsilon();
    let tol = eps * from_f64::<R>(16.0);

    for j in 0..n {
        for k in 0..j {
            let mut dot = R::zero();
            for i in 0..m {
                dot = dot + mat[(i, j)] * mat[(i, k)];
            }
            for i in 0..m {
                mat[(i, j)] = mat[(i, j)] - dot * mat[(i, k)];
            }
        }
        let mut norm_sq = R::zero();
        for i in 0..m {
            norm_sq = norm_sq + mat[(i, j)] * mat[(i, j)];
        }
        let norm = rsqrt(norm_sq);
        if norm > tol {
            let inv = R::one() / norm;
            for i in 0..m {
                mat[(i, j)] = mat[(i, j)] * inv;
            }
        } else {
            fill_orthonormal_columns(mat, &[j]);
        }
    }
}

/// Fills the listed columns of `mat` with unit vectors orthonormal to every
/// other (already-orthonormal) column, completing an orthonormal basis.
fn fill_orthonormal_columns<R>(mat: &mut Mat<R>, cols: &[usize])
where
    R: Field + Real + bytemuck::Zeroable,
{
    let m = mat.nrows();
    let n = mat.ncols();
    let eps = <R as Scalar>::epsilon();
    let tol = eps * from_f64::<R>(16.0);

    for &j in cols {
        for i in 0..m {
            mat[(i, j)] = R::zero();
        }
        let mut placed = false;
        for basis in 0..m {
            for i in 0..m {
                mat[(i, j)] = if i == basis { R::one() } else { R::zero() };
            }
            for k in 0..n {
                if k == j {
                    continue;
                }
                let mut dot = R::zero();
                for i in 0..m {
                    dot = dot + mat[(i, j)] * mat[(i, k)];
                }
                for i in 0..m {
                    mat[(i, j)] = mat[(i, j)] - dot * mat[(i, k)];
                }
            }
            let mut norm_sq = R::zero();
            for i in 0..m {
                norm_sq = norm_sq + mat[(i, j)] * mat[(i, j)];
            }
            let norm = rsqrt(norm_sq);
            if norm > tol {
                let inv = R::one() / norm;
                for i in 0..m {
                    mat[(i, j)] = mat[(i, j)] * inv;
                }
                placed = true;
                break;
            }
        }
        if !placed {
            // Should not happen for a well-formed basis; leave the column zero.
            for i in 0..m {
                mat[(i, j)] = R::zero();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_bidiag(d: &[f64], e: &[f64]) -> Vec<Vec<f64>> {
        let n = d.len();
        let mut b = vec![vec![0.0; n]; n];
        for i in 0..n {
            b[i][i] = d[i];
            if i + 1 < n {
                b[i][i + 1] = e[i];
            }
        }
        b
    }

    /// Returns `(max |UᵀU−I|, max |VᵀV−I|, max |UΣVᵀ − B|, sigma)`.
    fn svd_errors(d: &[f64], e: &[f64]) -> (f64, f64, f64, Vec<f64>) {
        let n = d.len();
        let (u, sigma, vt) = bidiagonal_svd_dc(d, e).expect("svd");

        let mut uu_err = 0.0f64;
        let mut vv_err = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let mut uu = 0.0;
                let mut vv = 0.0;
                for k in 0..n {
                    uu += u[(k, i)] * u[(k, j)];
                    vv += vt[(i, k)] * vt[(j, k)];
                }
                let expect = if i == j { 1.0 } else { 0.0 };
                uu_err = uu_err.max((uu - expect).abs());
                vv_err = vv_err.max((vv - expect).abs());
            }
        }

        let b = build_bidiag(d, e);
        let mut rec_err = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += u[(i, k)] * sigma[k] * vt[(k, j)];
                }
                rec_err = rec_err.max((acc - b[i][j]).abs());
            }
        }

        for i in 1..n {
            assert!(sigma[i] <= sigma[i - 1] + 1e-12, "not descending at {i}");
        }
        (uu_err, vv_err, rec_err, sigma)
    }

    #[test]
    fn svd_small_qr_path() {
        let d = vec![2.0, 3.0, 1.5, 4.0, 0.7];
        let e = vec![0.5, -0.3, 0.8, 0.2];
        let (uu, vv, rec, _) = svd_errors(&d, &e);
        assert!(uu < 1e-12 && vv < 1e-12 && rec < 1e-12, "{uu} {vv} {rec}");
    }

    #[test]
    fn svd_accuracy_across_sizes() {
        for &n in &[26usize, 50, 100, 200] {
            let d: Vec<f64> = (0..n)
                .map(|i| 2.0 + (i as f64 * 0.37).sin() + 0.5 * (i as f64 * 1.7).cos())
                .collect();
            let e: Vec<f64> = (0..n - 1)
                .map(|i| 0.6 + 0.4 * (i as f64 * 0.9).cos() + 0.2 * (i as f64 * 0.3).sin())
                .collect();
            let (uu, vv, rec, _) = svd_errors(&d, &e);
            println!("n={n}: UtU={uu:.2e} VtV={vv:.2e} recon={rec:.2e}");
            assert!(uu < 1e-6, "n={n} UtU={uu}");
            assert!(vv < 1e-9, "n={n} VtV={vv}");
            assert!(rec < 1e-8, "n={n} recon={rec}");
        }
    }

    #[test]
    fn svd_dc_path_with_zero_coupling() {
        let mut d = vec![0.0; 30];
        let mut e = vec![0.6; 29];
        for (i, di) in d.iter_mut().enumerate() {
            *di = 3.0 + (i as f64 * 0.5).sin();
        }
        e[15] = 0.0;
        let (uu, vv, rec, _) = svd_errors(&d, &e);
        assert!(uu < 1e-6 && vv < 1e-9 && rec < 1e-8, "{uu} {vv} {rec}");
    }
}
