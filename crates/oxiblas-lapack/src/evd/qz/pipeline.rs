//! The `DGGES` driver pipeline (JOBVSL = JOBVSR = 'V'; SORT = 'N' for
//! `Qz::compute`, SORT = 'S' through the step-7 seam for
//! `Qz::compute_sorted`).
//!
//! | step | `DGGES` call                    | here                                   |
//! |------|---------------------------------|----------------------------------------|
//! | 0    | argument checks                 | `validate`                             |
//! | 1    | `DLAMCH`                        | `Consts::new`                          |
//! | 2    | `DLANGE('M')` + `DLASCL`        | `run` (norm scaling)                   |
//! | 3    | `DGGBAL('P')`                   | `balance::ggbal_p`                     |
//! | 4    | `DGEQRF` + `DORMQR` + `DORGQR`  | `hessenberg::qr_reduce_b`              |
//! | 5    | `DGGHRD`                        | `hessenberg::gghrd`                    |
//! | 6    | `DHGEQZ('S', 'V', 'V')`         | `hgeqz::hgeqz`                         |
//! | 7    | `DTGSEN` (SORT = 'S' only)      | `reorder::tgsen0` (`compute_sorted`)   |
//! | 8    | `DGGBAK('P')` on Q and Z        | `QzWork::finish`                       |
//! | 9    | unscale guards + `DLASCL`       | `QzWork::finish`                       |
//! | 10   | eigenvalue triples              | `QzWork::finish`                       |
//!
//! # Step 7 seam (eigenvalue reordering)
//!
//! `run` returns the pencil exactly where `DGGES` would call `DTGSEN`: a
//! `QzWork` holding the generalized real Schur form of the *scaled and
//! permuted* pencil together with the accumulated (not yet back-permuted)
//! orthogonal factors. A reordering step operates on `h`, `t`, `q`, `z` of
//! that `QzWork` and recomputes `alphar`/`alphai`/`beta` from the reordered
//! (still scaled) blocks; selection predicates must see *unscaled*
//! eigenvalues, i.e. copies passed through `scaling::lascl_slice` with the
//! `ascale` / `bscale` pairs reversed (as `DGGES` does before calling
//! `SELCTG`). `QzWork::finish` then back-permutes and unscales exactly as
//! for the unsorted path.

use num_traits::Float;
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

use super::Qz;
use super::balance::{ggbak_p, ggbal_p};
use super::hessenberg::{gghrd, qr_reduce_b};
use super::hgeqz::hgeqz;
use super::scaling::{
    ScaleKind, guard_unscale_a, guard_unscale_b, lange_max, lascl_mat, lascl_slice,
};
use super::types::{GeneralizedEigenvalue, QzError};
use super::util::Consts;

/// Working state of the `DGGES` pipeline between `DHGEQZ` and `DGGBAK`.
///
/// Invariants after [`run`] (the step-7 seam):
/// * `h` is upper quasi-triangular and `t` upper triangular with exact zeros
///   below the (sub)diagonal; 1×1 blocks have `t[(j,j)] ≥ 0` and 2×2 blocks
///   a diagonal positive `t` block (`DHGEQZ` standardization).
/// * `Qᵀ·Ã·Z = h` and `Qᵀ·B̃·Z = t` with `Ã`, `B̃` the scaled, permuted
///   input, `q`/`z` orthogonal and not yet back-permuted.
/// * `alphar[j]`, `alphai[j]`, `beta[j]` are the `DHGEQZ` eigenvalue triples
///   of the scaled pencil (`alphar[j] = h[(j,j)]`, `beta[j] = t[(j,j)]`
///   bitwise for 1×1 blocks; complex pairs first with `alphai > 0`).
#[derive(Debug, Clone)]
pub(super) struct QzWork<T: Scalar> {
    /// `A` overwritten by the quasi-triangular factor `S` (scaled, permuted).
    pub(super) h: Mat<T>,
    /// `B` overwritten by the triangular factor `T` (scaled, permuted).
    pub(super) t: Mat<T>,
    /// Left Schur vectors `Q` (accumulated, not yet back-permuted).
    pub(super) q: Mat<T>,
    /// Right Schur vectors `Z` (accumulated, not yet back-permuted).
    pub(super) z: Mat<T>,
    /// `ALPHAR` of the scaled pencil.
    pub(super) alphar: Vec<T>,
    /// `ALPHAI` of the scaled pencil.
    pub(super) alphai: Vec<T>,
    /// `BETA` of the scaled pencil.
    pub(super) beta: Vec<T>,
    /// First index of the unreduced block found by `DGGBAL` (`ILO − 1`).
    pub(super) ilo: usize,
    /// Last index of the unreduced block found by `DGGBAL` (`IHI − 1`).
    pub(super) ihi: usize,
    /// `DGGBAL` row interchanges (`LSCALE`, 0-based).
    pub(super) lperm: Vec<usize>,
    /// `DGGBAL` column interchanges (`RSCALE`, 0-based).
    pub(super) rperm: Vec<usize>,
    /// `Some((anrm, anrmto))` when `A` was scaled (`DGGES` `ILASCL`).
    pub(super) ascale: Option<(T, T)>,
    /// `Some((bnrm, bnrmto))` when `B` was scaled (`DGGES` `ILBSCL`).
    pub(super) bscale: Option<(T, T)>,
}

/// Step 0: argument checks in the historical order of `Qz::compute`;
/// returns `n`. Non-finite entries are rejected with
/// `NotConverged { iterations: 0 }` (the QZ iteration cannot converge on
/// them).
pub(super) fn validate<T: Field + Real>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
) -> Result<usize, QzError> {
    let n = a.nrows();
    if n == 0 {
        return Err(QzError::EmptyMatrix);
    }
    if n != a.ncols() {
        return Err(QzError::NotSquare {
            nrows: a.nrows(),
            ncols: a.ncols(),
        });
    }
    if n != b.nrows() || n != b.ncols() {
        return Err(QzError::DimensionMismatch {
            nrows_a: a.nrows(),
            ncols_a: a.ncols(),
            nrows_b: b.nrows(),
            ncols_b: b.ncols(),
        });
    }
    for j in 0..n {
        for i in 0..n {
            if !Float::is_finite(a[(i, j)]) || !Float::is_finite(b[(i, j)]) {
                return Err(QzError::NotConverged { iterations: 0 });
            }
        }
    }
    Ok(n)
}

/// Step 2 helper: `DGGES` decides whether a matrix with max-norm `nrm`
/// must be scaled; returns `Some((nrm, nrmto))` when it must. Also used by
/// `Qz::reorder`, which scales `S` and `T` the same way around `DTGSEN`.
pub(super) fn scaling_target<T: Field + Real>(nrm: T, k: &Consts<T>) -> Option<(T, T)> {
    if nrm > T::zero() && nrm < k.smlnum {
        Some((nrm, k.smlnum))
    } else if nrm > k.bignum {
        Some((nrm, k.bignum))
    } else {
        None
    }
}

/// Steps 0–6 of `DGGES`: validation, norm scaling, permutation balancing,
/// QR of `B`, Hessenberg–triangular reduction and the QZ iteration.
///
/// Returns the working state at the step-7 seam (see the module docs);
/// `QzError::NotConverged { iterations }` when `DHGEQZ` exhausts its budget
/// (partial results are discarded).
pub(super) fn run<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
) -> Result<QzWork<T>, QzError> {
    let n = validate(a, b)?;
    let k = Consts::new();

    let mut h = Mat::zeros(n, n);
    let mut t = Mat::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            h[(i, j)] = a[(i, j)];
            t[(i, j)] = b[(i, j)];
        }
    }

    // Step 2: scale A and B independently if max |entry| is outside
    // [SMLNUM, BIGNUM].
    let ascale = scaling_target(lange_max(&h), &k);
    if let Some((from, to)) = ascale {
        lascl_mat(ScaleKind::General, from, to, &mut h, &k);
    }
    let bscale = scaling_target(lange_max(&t), &k);
    if let Some((from, to)) = bscale {
        lascl_mat(ScaleKind::General, from, to, &mut t, &k);
    }

    // Step 3: permute to isolate eigenvalues.
    let perm = ggbal_p(&mut h, &mut t);

    let mut work = QzWork {
        h,
        t,
        q: Mat::eye(n),
        z: Mat::eye(n),
        alphar: vec![T::zero(); n],
        alphai: vec![T::zero(); n],
        beta: vec![T::zero(); n],
        ilo: perm.ilo,
        ihi: perm.ihi,
        lperm: perm.lperm,
        rperm: perm.rperm,
        ascale,
        bscale,
    };

    // Step 4: B(ilo..=ihi, ilo..n) = Q·R, A ← Qᵀ·A, Q accumulated.
    qr_reduce_b(&mut work, &k);
    // Step 5: Hessenberg–triangular reduction.
    gghrd(&mut work, &k);
    // Step 6: QZ iteration (Schur form, Q and Z updated).
    hgeqz(&mut work, &k)?;
    Ok(work)
}

impl<T: Field + Real + bytemuck::Zeroable> QzWork<T> {
    /// Steps 8–10 of `DGGES`: back-permutation of `Q` and `Z`, undoing the
    /// norm scaling of `S`, `T` and the eigenvalue triples, and packaging.
    pub(super) fn finish(mut self) -> Qz<T> {
        let k = Consts::new();
        let n = self.h.nrows();

        // Step 8: DGGBAK('P', 'L') on Q and DGGBAK('P', 'R') on Z.
        ggbak_p(&mut self.q, self.ilo, self.ihi, &self.lperm);
        ggbak_p(&mut self.z, self.ilo, self.ihi, &self.rperm);

        // Step 9: over/underflow guards for complex pairs, then DLASCL with
        // the same multiplier sequence for the matrix and the triples (so a
        // 1×1 block keeps alphar == S(j,j) and beta == T(j,j) bitwise).
        if let Some((anrm, anrmto)) = self.ascale {
            guard_unscale_a(
                &self.h,
                &mut self.alphar,
                &mut self.alphai,
                &mut self.beta,
                anrm,
                anrmto,
                &k,
            );
        }
        if let Some((bnrm, bnrmto)) = self.bscale {
            guard_unscale_b(
                &self.t,
                &mut self.alphar,
                &mut self.alphai,
                &mut self.beta,
                bnrm,
                bnrmto,
                &k,
            );
        }
        if let Some((anrm, anrmto)) = self.ascale {
            lascl_mat(ScaleKind::UpperHessenberg, anrmto, anrm, &mut self.h, &k);
            lascl_slice(anrmto, anrm, &mut self.alphar, &k);
            lascl_slice(anrmto, anrm, &mut self.alphai, &k);
        }
        if let Some((bnrm, bnrmto)) = self.bscale {
            lascl_mat(ScaleKind::UpperTriangular, bnrmto, bnrm, &mut self.t, &k);
            lascl_slice(bnrmto, bnrm, &mut self.beta, &k);
        }

        // Step 10: eigenvalues exactly as computed by DHGEQZ (never
        // re-derived from S and T).
        let eigenvalues = (0..n)
            .map(|j| GeneralizedEigenvalue::complex(self.alphar[j], self.alphai[j], self.beta[j]))
            .collect();

        Qz {
            s: self.h,
            t: self.t,
            q: self.q,
            z: self.z,
            eigenvalues,
            n,
        }
    }
}
