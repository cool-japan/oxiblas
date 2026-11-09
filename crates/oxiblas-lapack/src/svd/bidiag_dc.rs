//! Real bidiagonal SVD by divide and conquer (LAPACK `dbdsdc` / `dlasd0`).
//!
//! This module computes the SVD `B = U·diag(σ)·Vᵀ` of an upper bidiagonal
//! matrix `B` (diagonal `d`, super-diagonal `e`). It is shared by the real
//! ([`crate::svd::SvdDc`]) and complex ([`crate::svd::ComplexSvdDc`])
//! divide-and-conquer front-ends: complex bidiagonalisation produces a *real*
//! bidiagonal matrix, so the same kernel serves both.
//!
//! # Algorithm
//!
//! The divide and conquer works on `B` itself (Gu & Eisenstat), never on
//! `BᵀB`, so it is backward stable and `U` and `V` come out numerically
//! orthogonal without any re-orthogonalisation:
//!
//! 1. **Scaling.** `d` and `e` are scaled to unit max-norm by a power of two
//!    (exactly; `dbdsdc` scales with `dlascl`) and the singular values are
//!    unscaled on exit, so the absolute tolerances below are relative to
//!    `‖B‖_max` whatever the magnitude of the input.
//! 2. **Small matrices** (`n ≤ 25`) are solved by the implicit QR iteration
//!    of [`crate::svd::bidiag_qr`] (`dbdsqr` in relative-accuracy mode with
//!    explicit removal of zero diagonal entries).
//! 3. **Splitting.** Larger matrices split at every `|eᵢ| < 0.9·ε` (scaled);
//!    each subproblem is solved independently.
//! 4. **Divide and conquer** (`dlasd0`): a subproblem of more than 25 rows
//!    is cut at its middle row into an upper `nl × (nl + 1)` and a lower
//!    `nr × (nr + sqre)` bidiagonal, solved recursively (leaves by the QR
//!    iteration), and merged by [`crate::svd::bidiag_merge`] (`dlasd1`–
//!    `dlasd3`: deflation relative to the scaled node, the `dlasd4` secular
//!    equation, Gu–Eisenstat vectors and GEMM back-transforms over the
//!    diagonal blocks).
//! 5. The singular values are sorted descending with their vectors.
//!
//! Singular values carry absolute errors of order `ε·‖B‖` (as for
//! `dbdsdc`, whose deflation and splitting tolerances are absolute within
//! each scaled subproblem). Singular values well above `ε·‖B‖` come out with
//! high relative accuracy in practice (graded bidiagonals spanning twelve
//! decades agree with the QR iteration to a few ulps, as reference LAPACK
//! does); below `ε·‖B‖` only the absolute bound holds, and in exactly
//! singular cases the smallest ones come out as `O(ε·‖B‖)` rather than
//! exactly zero once the divide and conquer is used. The QR path (`n ≤ 25`)
//! keeps high relative accuracy for every singular value. Unlike `dbdsdc`,
//! diagonal entries below `ε·‖B‖` are not raised to `±ε·‖B‖` before the
//! divide and conquer, which keeps tiny singular values far more accurate.
//!
//! Non-finite input is reported as [`BidiagDcError::NotConverged`].
//!
//! # References
//!
//! - Gu, M. & Eisenstat, S. C. (1995). *A Divide-and-Conquer Algorithm for the
//!   Bidiagonal SVD*. SIAM J. Matrix Anal. Appl., 16(1), 79–92.
//! - Demmel, J. & Kahan, W. (1990). *Accurate Singular Values of Bidiagonal
//!   Matrices*. SIAM J. Sci. Stat. Comput., 11(5), 873–912.
//! - LAPACK 3.12: `dbdsdc`, `dlasd0`–`dlasd5`, `dlasdq`, `dbdsqr`.

use oxiblas_core::scalar::{Field, Real};
use oxiblas_matrix::Mat;

use super::bidiag_merge::{Node, merge};
use super::bidiag_qr::lasdq;
use super::dc_kernels::{
    BlasOps, DenseOps, PortableOps, cmp_desc_nan_last, fabs, lapack_eps, max_abs, pow2_exponent,
    real_const, same_mut, same_ref, scale_pow2, sign_of, transpose_into,
};

/// Error returned by the real bidiagonal SVD kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BidiagDcError {
    /// The QR iteration exhausted its budget, or the input is not finite.
    NotConverged,
    /// The secular-equation solver produced no finite root or vector.
    SecularEquationFailed,
}

/// Subproblems of at most this many rows are solved by the QR iteration
/// (`SMLSIZ`, LAPACK `ilaenv(9)`).
pub(crate) const SMLSIZ: usize = 25;

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
    let d_vec = d.to_vec();
    let e_vec = e.to_vec();
    let mut u = Mat::zeros(n, n);
    let mut sigma = vec![R::zero(); n];
    let mut v = Mat::zeros(n, n);

    if let Some(res) = typed_bidiag::<R, f64>(&d_vec, &e_vec, &mut u, &mut sigma, &mut v) {
        res?;
    } else if let Some(res) = typed_bidiag::<R, f32>(&d_vec, &e_vec, &mut u, &mut sigma, &mut v) {
        res?;
    } else {
        bidiag_svd_into::<R, PortableOps>(PortableOps, &d_vec, &e_vec, &mut u, &mut sigma, &mut v)?;
    }

    let mut vt = Mat::zeros(n, n);
    transpose_into(v.as_ref(), vt.as_mut());
    Ok((u, sigma, vt))
}

/// Runs the GEMM-backed kernel when `R` is the BLAS scalar `G`.
fn typed_bidiag<R, G>(
    d: &Vec<R>,
    e: &Vec<R>,
    u: &mut Mat<R>,
    sigma: &mut Vec<R>,
    v: &mut Mat<R>,
) -> Option<Result<(), BidiagDcError>>
where
    R: Field + Real + bytemuck::Zeroable,
    G: Field + Real + bytemuck::Zeroable + oxiblas_blas::level3::gemm_kernel::GemmKernel,
{
    let d = same_ref::<Vec<R>, Vec<G>>(d)?;
    let e = same_ref::<Vec<R>, Vec<G>>(e)?;
    let u = same_mut::<Mat<R>, Mat<G>>(u)?;
    let sigma = same_mut::<Vec<R>, Vec<G>>(sigma)?;
    let v = same_mut::<Mat<R>, Mat<G>>(v)?;
    Some(bidiag_svd_into::<G, BlasOps>(
        BlasOps::SEQ,
        d,
        e,
        u,
        sigma,
        v,
    ))
}

/// `n × n` identity.
fn identity<R: Field + bytemuck::Zeroable>(n: usize) -> Mat<R> {
    let mut m = Mat::zeros(n, n);
    for i in 0..n {
        m[(i, i)] = R::one();
    }
    m
}

/// Bidiagonal SVD `B = U·diag(sigma)·Vᵀ` written into caller-provided
/// buffers (`u`, `v`: `n×n`, `sigma`: `n`), with `V` stored by columns and
/// `sigma` descending.
///
/// `ops` selects the dense backend (see [`crate::svd::dc_kernels`]).
pub(crate) fn bidiag_svd_into<R, B>(
    ops: B,
    d: &[R],
    e: &[R],
    u: &mut Mat<R>,
    sigma: &mut [R],
    v: &mut Mat<R>,
) -> Result<(), BidiagDcError>
where
    R: Field + Real + bytemuck::Zeroable,
    B: DenseOps<R>,
{
    let n = d.len();
    if n == 0 {
        return Ok(());
    }
    if d.iter().chain(e.iter()).any(|x| !x.is_finite()) {
        return Err(BidiagDcError::NotConverged);
    }
    let mut dd = d.to_vec();
    let mut ee = e[..n - 1].to_vec();

    // Scale to unit max-norm (power of two, exact).
    let anorm = max_abs(&dd).max(max_abs(&ee));
    if anorm == R::zero() {
        sigma.fill(R::zero());
        for (target, dim) in [(&mut *u, n), (&mut *v, n)] {
            let ld = target.row_stride();
            let data = target.raw_data_mut();
            for j in 0..dim {
                data[j * ld..j * ld + dim].fill(R::zero());
                data[j * ld + j] = R::one();
            }
        }
        return Ok(());
    }
    let ks = pow2_exponent(anorm);
    scale_pow2(&mut dd, ks);
    scale_pow2(&mut ee, ks);

    let (mut uw, mut vw) = if n <= SMLSIZ {
        let mut uw = identity::<R>(n);
        let mut vw = identity::<R>(n);
        lasdq(0, &mut dd, &mut ee, &mut uw, &mut vw)?;
        (uw, vw)
    } else {
        let mut uw = Mat::<R>::zeros(n, n);
        let mut vw = Mat::<R>::zeros(n, n);
        solve_split(&ops, &mut dd, &mut ee, &mut uw, &mut vw)?;
        (uw, vw)
    };
    scale_pow2(&mut dd, -ks);

    // Descending order (NaN last), columns permuted alongside.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| cmp_desc_nan_last(dd[a], dd[b]));
    let ldu = u.row_stride();
    let ldv = v.row_stride();
    let lduw = uw.row_stride();
    let ldvw = vw.row_stride();
    {
        let ud = u.raw_data_mut();
        let uwd = uw.raw_data_mut();
        for (new_j, &old_j) in order.iter().enumerate() {
            ud[new_j * ldu..new_j * ldu + n].copy_from_slice(&uwd[old_j * lduw..old_j * lduw + n]);
        }
    }
    {
        let vd = v.raw_data_mut();
        let vwd = vw.raw_data_mut();
        for (new_j, &old_j) in order.iter().enumerate() {
            vd[new_j * ldv..new_j * ldv + n].copy_from_slice(&vwd[old_j * ldvw..old_j * ldvw + n]);
        }
    }
    for (new_j, &old_j) in order.iter().enumerate() {
        sigma[new_j] = dd[old_j];
    }
    Ok(())
}

/// Splits the scaled bidiagonal at negligible super-diagonal entries
/// (`|eᵢ| < 0.9·ε`, as `dbdsdc`) and solves every piece into the diagonal
/// blocks of the zero-initialised `u` and `v`.
fn solve_split<R, B>(
    ops: &B,
    d: &mut [R],
    e: &mut [R],
    u: &mut Mat<R>,
    v: &mut Mat<R>,
) -> Result<(), BidiagDcError>
where
    R: Field + Real + bytemuck::Zeroable,
    B: DenseOps<R>,
{
    let n = d.len();
    let eps = real_const::<R>(0.9) * lapack_eps::<R>();
    let mut start = 0usize;
    for i in 0..n - 1 {
        let small = fabs(e[i]) < eps;
        if !(small || i == n - 2) {
            continue;
        }
        let nsize = if i < n - 2 || !small {
            if i < n - 2 { i - start + 1 } else { n - start }
        } else {
            // e[n − 2] negligible: d[n − 1] is a 1×1 subproblem of its own.
            u[(n - 1, n - 1)] = sign_of(R::one(), d[n - 1]);
            v[(n - 1, n - 1)] = R::one();
            d[n - 1] = fabs(d[n - 1]);
            i - start + 1
        };
        let end = start + nsize;
        solve_piece(ops, &mut d[start..end], &mut e[start..end - 1], u, v, start)?;
        start = i + 1;
    }
    Ok(())
}

/// Solves one unreduced piece (`sqre = 0`) at diagonal offset `off`.
fn solve_piece<R, B>(
    ops: &B,
    d: &mut [R],
    e: &mut [R],
    u: &mut Mat<R>,
    v: &mut Mat<R>,
    off: usize,
) -> Result<(), BidiagDcError>
where
    R: Field + Real + bytemuck::Zeroable,
    B: DenseOps<R>,
{
    if d.len() == 1 {
        u[(off, off)] = sign_of(R::one(), d[0]);
        v[(off, off)] = R::one();
        d[0] = fabs(d[0]);
        return Ok(());
    }
    let mut idxq = vec![0usize; d.len()];
    divide(ops, 0, d, e, u, v, off, &mut idxq)
}

/// Divide-and-conquer SVD (LAPACK `dlasd0`) of the `n × (n + sqre)` upper
/// bidiagonal `(d, e)` (`e`: `n − 1 + sqre` entries) into the blocks of `u`
/// (`n × n`) and `v` (`(n + sqre) × (n + sqre)`) at diagonal offset `off`,
/// which must be zero outside the diagonal blocks this call writes.
///
/// On exit `d` holds the singular values and `idxq` the permutation that
/// sorts them ascending.
#[allow(clippy::too_many_arguments)]
fn divide<R, B>(
    ops: &B,
    sqre: usize,
    d: &mut [R],
    e: &mut [R],
    u: &mut Mat<R>,
    v: &mut Mat<R>,
    off: usize,
    idxq: &mut [usize],
) -> Result<(), BidiagDcError>
where
    R: Field + Real + bytemuck::Zeroable,
    B: DenseOps<R>,
{
    let n = d.len();
    let m = n + sqre;
    if n <= SMLSIZ {
        // Leaf: QR iteration from the identity, then copy into the blocks.
        let mut lu = identity::<R>(n);
        let mut lv = identity::<R>(m);
        lasdq(sqre, d, e, &mut lu, &mut lv)?;
        copy_block(&lu, u, off);
        copy_block(&lv, v, off);
        for (i, q) in idxq.iter_mut().enumerate() {
            *q = i;
        }
        return Ok(());
    }
    let nl = n / 2;
    let nr = n - nl - 1;
    let alpha = d[nl];
    let beta = e[nl];
    {
        let (dl, dr) = d.split_at_mut(nl);
        let (el, er) = e.split_at_mut(nl);
        let (ql, qr) = idxq.split_at_mut(nl);
        divide(ops, 1, dl, el, u, v, off, ql)?;
        divide(
            ops,
            sqre,
            &mut dr[1..],
            &mut er[1..],
            u,
            v,
            off + nl + 1,
            &mut qr[1..],
        )?;
    }
    merge(ops, Node { nl, nr, sqre, off }, d, alpha, beta, u, v, idxq)
}

/// Copies the square `src` into `dst` at diagonal offset `off`.
fn copy_block<R: Field>(src: &Mat<R>, dst: &mut Mat<R>, off: usize) {
    let k = src.nrows();
    let lds = src.row_stride();
    let ldd = dst.row_stride();
    let sd = src.raw_data();
    let dd = dst.raw_data_mut();
    for j in 0..src.ncols() {
        let to = (off + j) * ldd + off;
        dd[to..to + k].copy_from_slice(&sd[j * lds..j * lds + k]);
    }
}

#[cfg(test)]
#[path = "bidiag_dc_tests.rs"]
mod tests;
