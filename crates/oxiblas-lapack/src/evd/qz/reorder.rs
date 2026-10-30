//! Eigenvalue reordering of the generalized real Schur form (Netlib LAPACK
//! 3.12.0, BSD-3): `DTGEXC`, `DTGSEN` with `IJOB = 0`, the `DGGES`
//! `SORT = 'S'` driver steps, and the public entry points
//! [`Qz::compute_sorted`] and [`Qz::reorder`].
//!
//! `DTGSEN` moves every selected diagonal block (a complex pair is moved as
//! a whole when either member is selected) to the top-left corner by
//! sequences of adjacent `DTGEX2` swaps (`DTGEXC`), then recomputes the
//! eigenvalue triples from the reordered blocks (`DLAG2` with
//! `SMLNUM·EPS` for 2×2 blocks, the diagonal for 1×1 blocks) and makes the
//! diagonal of `T` non-negative. `DGGES` finally re-evaluates the selector
//! on the final eigenvalues (the LASTSL/LST2SL loop) to count `SDIM` and to
//! detect selections changed by rounding (`INFO = N+2`).

use oxiblas_core::scalar::{Field, Real};
use oxiblas_matrix::{Mat, MatRef};

use super::Qz;
use super::lag2::dlag2;
use super::pipeline::{self, scaling_target};
use super::scaling::{
    ScaleKind, guard_unscale_a, guard_unscale_b, lange_max, lascl_mat, lascl_slice,
};
use super::swap::{SchurPencil, tgex2};
use super::types::{GeneralizedEigenvalue, QzError};
use super::util::{Consts, sign};

/// Error of the eigenvalue-reordering entry points [`Qz::compute_sorted`]
/// and [`Qz::reorder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum QzReorderError {
    /// The unsorted QZ decomposition failed (only returned by
    /// [`Qz::compute_sorted`]); see [`QzError`].
    Qz(QzError),
    /// The swap of two adjacent diagonal blocks was rejected by the
    /// stability tests of LAPACK `DTGEX2`, so the requested ordering cannot
    /// be computed stably (LAPACK `DTGSEN` `INFO = 1`, `DGGES`
    /// `INFO = N+3`). Typical causes are blocks with (nearly) identical
    /// eigenvalues and, as in LAPACK, a pencil whose (internally scaled) `S`
    /// and `T` differ in norm by more than about `1/eps` when a 2×2 block
    /// has to move.
    SwapFailed {
        /// 0-based diagonal position (row/column of `S` and `T`) of the
        /// first row of the selected block whose next adjacent swap was
        /// rejected.
        position: usize,
    },
}

impl core::fmt::Display for QzReorderError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Qz(e) => write!(f, "QZ decomposition failed: {e}"),
            Self::SwapFailed { position } => write!(
                f,
                "Eigenvalue reordering failed: swapping the diagonal block at position {position} \
                 was rejected by the stability tests (eigenvalues too close, or the pencil is \
                 badly balanced)"
            ),
        }
    }
}

impl std::error::Error for QzReorderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Qz(e) => Some(e),
            Self::SwapFailed { .. } => None,
        }
    }
}

impl From<QzError> for QzReorderError {
    fn from(e: QzError) -> Self {
        Self::Qz(e)
    }
}

/// Outcome of a successful eigenvalue reordering ([`Qz::compute_sorted`],
/// [`Qz::reorder`]).
///
/// Fields may be added in future releases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct QzOrdering {
    /// Number of eigenvalues for which the selector holds after the
    /// reordering, a complex conjugate pair counting 2 when either member is
    /// selected (LAPACK `DGGES` `SDIM`). The leading `sdim` columns of `Q`
    /// and `Z` span the left and right deflating subspaces of the selected
    /// eigenvalues.
    pub sdim: usize,
    /// LAPACK `DGGES` `INFO = N+2`: after the reordering, rounding (or the
    /// internal scaling) changed some eigenvalues so that the selector no
    /// longer holds for all leading eigenvalues — a selected eigenvalue now
    /// follows an unselected one. The factorization itself is valid.
    pub selection_changed: bool,
}

/// The 2×2 block `m(i..=i+1, i..=i+1)` as `[row][column]`.
fn block2<T: Field + Real>(m: &Mat<T>, i: usize) -> [[T; 2]; 2] {
    [
        [m[(i, i)], m[(i, i + 1)]],
        [m[(i + 1, i)], m[(i + 1, i + 1)]],
    ]
}

/// True when `s(j+1, j) ≠ 0`, i.e. a 2×2 block starts at row `j`.
fn starts_pair<T: Field + Real>(s: &Mat<T>, j: usize) -> bool {
    j + 1 < s.nrows() && s[(j + 1, j)] != T::zero()
}

/// `DTGEXC`: moves the diagonal block whose row `ifst` belongs to (the
/// first row of a 2×2 block is used when `ifst` points to its second row)
/// to row `ilst` by a sequence of adjacent `DTGEX2` swaps, following 2×2
/// blocks that split into two 1×1 blocks during the move.
///
/// Returns the final first row of the block (which may differ from `ilst`
/// by ±1, as in LAPACK). On a rejected swap returns `Err(here)`, the first
/// row of the block at that moment (`DTGEXC` `ILST` with `INFO = 1`); the
/// swaps performed before the rejection are kept.
pub(super) fn tgexc<T: Field + Real>(
    p: &mut SchurPencil<'_, T>,
    ifst: usize,
    ilst: usize,
    k: &Consts<T>,
) -> Result<usize, usize> {
    let n = p.s.nrows();
    // Quick return if possible.
    if n <= 1 {
        return Ok(ilst);
    }
    debug_assert!(ifst < n && ilst < n, "DTGEXC: IFST/ILST out of range");

    // Determine the first row of the specified block and find out if it is
    // 1-by-1 or 2-by-2.
    let mut ifst = ifst;
    if ifst > 0 && p.s[(ifst, ifst - 1)] != T::zero() {
        ifst -= 1;
    }
    let mut nbf = if starts_pair(p.s, ifst) { 2 } else { 1 };

    // Determine the first row of the final block and find out if it is
    // 1-by-1 or 2-by-2.
    let mut ilst = ilst;
    if ilst > 0 && p.s[(ilst, ilst - 1)] != T::zero() {
        ilst -= 1;
    }
    let nbl = if starts_pair(p.s, ilst) { 2 } else { 1 };
    if ifst == ilst {
        return Ok(ilst);
    }

    let zero = T::zero();
    let mut here = ifst;
    if ifst < ilst {
        // Update ILST.
        if nbf == 2 && nbl == 1 {
            ilst -= 1;
        }
        if nbf == 1 && nbl == 2 {
            ilst += 1;
        }
        // Label 10: swap with the next block below.
        loop {
            if nbf == 1 || nbf == 2 {
                // Current block either 1-by-1 or 2-by-2.
                let nbnext = if here + nbf + 1 < n && p.s[(here + nbf + 1, here + nbf)] != zero {
                    2
                } else {
                    1
                };
                if tgex2(p, here, nbf, nbnext, k).is_err() {
                    return Err(here);
                }
                here += nbnext;
                // Test if 2-by-2 block breaks into two 1-by-1 blocks.
                if nbf == 2 && p.s[(here + 1, here)] == zero {
                    nbf = 3;
                }
            } else {
                // Current block consists of two 1-by-1 blocks, each of which
                // must be swapped individually.
                let mut nbnext = if here + 3 < n && p.s[(here + 3, here + 2)] != zero {
                    2
                } else {
                    1
                };
                if tgex2(p, here + 1, 1, nbnext, k).is_err() {
                    return Err(here);
                }
                if nbnext == 1 {
                    // Swap two 1-by-1 blocks.
                    if tgex2(p, here, 1, 1, k).is_err() {
                        return Err(here);
                    }
                    here += 1;
                } else {
                    // Recompute NBNEXT in case of 2-by-2 split.
                    if p.s[(here + 2, here + 1)] == zero {
                        nbnext = 1;
                    }
                    if nbnext == 2 {
                        // 2-by-2 block did not split.
                        if tgex2(p, here, 1, nbnext, k).is_err() {
                            return Err(here);
                        }
                        here += 2;
                    } else {
                        // 2-by-2 block did split.
                        if tgex2(p, here, 1, 1, k).is_err() {
                            return Err(here);
                        }
                        here += 1;
                        if tgex2(p, here, 1, 1, k).is_err() {
                            return Err(here);
                        }
                        here += 1;
                    }
                }
            }
            if here >= ilst {
                break;
            }
        }
    } else {
        // Label 20: swap with the next block above.
        loop {
            if nbf == 1 || nbf == 2 {
                // Current block either 1-by-1 or 2-by-2.
                let nbnext = if here >= 2 && p.s[(here - 1, here - 2)] != zero {
                    2
                } else {
                    1
                };
                if tgex2(p, here - nbnext, nbnext, nbf, k).is_err() {
                    return Err(here);
                }
                here -= nbnext;
                // Test if 2-by-2 block breaks into two 1-by-1 blocks.
                if nbf == 2 && p.s[(here + 1, here)] == zero {
                    nbf = 3;
                }
            } else {
                // Current block consists of two 1-by-1 blocks, each of which
                // must be swapped individually.
                let mut nbnext = if here >= 2 && p.s[(here - 1, here - 2)] != zero {
                    2
                } else {
                    1
                };
                if tgex2(p, here - nbnext, nbnext, 1, k).is_err() {
                    return Err(here);
                }
                if nbnext == 1 {
                    // Swap two 1-by-1 blocks.
                    if tgex2(p, here, nbnext, 1, k).is_err() {
                        return Err(here);
                    }
                    here -= 1;
                } else {
                    // Recompute NBNEXT in case of 2-by-2 split.
                    if p.s[(here, here - 1)] == zero {
                        nbnext = 1;
                    }
                    if nbnext == 2 {
                        // 2-by-2 block did not split.
                        if tgex2(p, here - 1, 2, 1, k).is_err() {
                            return Err(here);
                        }
                        here -= 2;
                    } else {
                        // 2-by-2 block did split.
                        if tgex2(p, here, 1, 1, k).is_err() {
                            return Err(here);
                        }
                        here -= 1;
                        if tgex2(p, here, 1, 1, k).is_err() {
                            return Err(here);
                        }
                        here -= 1;
                    }
                }
            }
            if here <= ilst {
                break;
            }
        }
    }
    Ok(here)
}

/// Negates row `r` of `S` and `T` (all columns) and column `r` of `Q`
/// (`DTGSEN` loop 70), keeping `Q·S·Zᵀ` and `Q·T·Zᵀ` unchanged.
fn negate_row<T: Field + Real>(p: &mut SchurPencil<'_, T>, r: usize) {
    let n = p.s.nrows();
    for i in 0..n {
        p.s[(r, i)] = -p.s[(r, i)];
        p.t[(r, i)] = -p.t[(r, i)];
        p.q[(i, r)] = -p.q[(i, r)];
    }
}

/// `DTGSEN` label 60 / loop 80: computes the generalized eigenvalues of the
/// (reordered) pencil and normalizes the generalized Schur form.
///
/// * 2×2 block: `DLAG2` with the safe minimum `SMLNUM·EPS` gives
///   `BETA(K) = SCALE1`, `BETA(K+1) = SCALE2`, `ALPHAR(K) = WR1`,
///   `ALPHAR(K+1) = WR2`, `ALPHAI(K) = WI` and `ALPHAI(K+1) = −ALPHAI(K)`.
///   Deliberate extension of the reference: a negative diagonal entry of the
///   (diagonal) `T` block is made positive first by negating its row of `S`
///   and `T` and its column of `Q`, the same transformation the reference
///   applies to 1×1 blocks. `DTGEX2`'s `DLAGV2` standardization does not
///   control these signs, and `DGGES` documents `T` with a non-negative
///   diagonal.
/// * 1×1 block: if `SIGN(1, B(K,K)) < 0` the row is negated likewise, then
///   `ALPHAR(K) = A(K,K)`, `ALPHAI(K) = 0`, `BETA(K) = B(K,K)` (bitwise).
fn normalize_and_eigenvalues<T: Field + Real>(
    p: &mut SchurPencil<'_, T>,
    alphar: &mut [T],
    alphai: &mut [T],
    beta: &mut [T],
    k: &Consts<T>,
) {
    let n = p.s.nrows();
    let (zero, one) = (T::zero(), T::one());
    // DTGSEN: SMLNUM = DLAMCH('S') / EPS; DLAG2 receives SMLNUM·EPS.
    let safmin = (k.safmin / k.eps) * k.eps;
    let mut kk = 0;
    while kk < n {
        if starts_pair(p.s, kk) {
            for r in [kk, kk + 1] {
                if sign(one, p.t[(r, r)]) < zero {
                    negate_row(p, r);
                }
            }
            // Compute the eigenvalue(s) at position K.
            let g = dlag2(block2(p.s, kk), block2(p.t, kk), safmin, k);
            beta[kk] = g.scale1;
            beta[kk + 1] = g.scale2;
            alphar[kk] = g.wr1;
            alphar[kk + 1] = g.wr2;
            alphai[kk] = g.wi;
            alphai[kk + 1] = -alphai[kk];
            kk += 2;
        } else {
            // If B(K,K) is negative, make it positive.
            if sign(one, p.t[(kk, kk)]) < zero {
                negate_row(p, kk);
            }
            alphar[kk] = p.s[(kk, kk)];
            alphai[kk] = zero;
            beta[kk] = p.t[(kk, kk)];
            kk += 1;
        }
    }
}

/// `DTGSEN(IJOB = 0, WANTQ = WANTZ = .TRUE.)`: reorders the generalized
/// real Schur form so that the blocks flagged in `select` (indexed by the
/// current diagonal position; a 2×2 block moves when either of its two
/// flags is set) occupy the top-left corner, then recomputes
/// `alphar`/`alphai`/`beta` and normalizes the form (label 60).
///
/// On a rejected swap the reference sets `INFO = 1` and still computes the
/// eigenvalues of the partially reordered pencil (`GO TO 60`); this returns
/// `Err(position)` (see [`tgexc`]) after doing the same.
pub(super) fn tgsen0<T: Field + Real>(
    select: &[bool],
    p: &mut SchurPencil<'_, T>,
    alphar: &mut [T],
    alphai: &mut [T],
    beta: &mut [T],
    k: &Consts<T>,
) -> Result<(), usize> {
    let n = p.s.nrows();

    // Set M to the dimension of the specified pair of deflating subspaces
    // (loop 10).
    let mut m = 0;
    let mut kk = 0;
    while kk < n {
        if starts_pair(p.s, kk) {
            if select[kk] || select[kk + 1] {
                m += 2;
            }
            kk += 2;
        } else {
            if select[kk] {
                m += 1;
            }
            kk += 1;
        }
    }

    let mut status = Ok(());
    // Quick return if possible: nothing to move.
    if m != n && m != 0 {
        // Collect the selected blocks at the top-left corner of (A, B)
        // (loop 30); `ks` counts the positions already filled.
        let mut ks = 0;
        let mut kk = 0;
        while kk < n {
            let pair = starts_pair(p.s, kk);
            let swap = select[kk] || (pair && select[kk + 1]);
            if swap {
                // Swap the K-th block to position KS.
                if kk != ks {
                    if let Err(position) = tgexc(p, kk, ks, k) {
                        // Swap is rejected: exit (GO TO 60).
                        status = Err(position);
                        break;
                    }
                }
                ks += if pair { 2 } else { 1 };
            }
            kk += if pair { 2 } else { 1 };
        }
    }

    // Label 60: eigenvalues of the reordered pair and normalization.
    normalize_and_eigenvalues(p, alphar, alphai, beta, k);
    status
}

/// Undoes the norm scaling of `S`, `T` and of the eigenvalue triples after
/// `DTGSEN`, exactly as `DGGES` does (the steps of `QzWork::finish` step 9):
/// the over/underflow guards for complex pairs, then `DLASCL` with the same
/// multiplier sequence for the matrices and the triples, so that 1×1 blocks
/// keep `alphar == S(j,j)` and `beta == T(j,j)` bitwise.
fn undo_scaling<T: Field + Real>(
    (s, t): (&mut Mat<T>, &mut Mat<T>),
    (alphar, alphai, beta): (&mut [T], &mut [T], &mut [T]),
    (ascale, bscale): (Option<(T, T)>, Option<(T, T)>),
    k: &Consts<T>,
) {
    if let Some((anrm, anrmto)) = ascale {
        guard_unscale_a(s, alphar, alphai, beta, anrm, anrmto, k);
    }
    if let Some((bnrm, bnrmto)) = bscale {
        guard_unscale_b(t, alphar, alphai, beta, bnrm, bnrmto, k);
    }
    if let Some((anrm, anrmto)) = ascale {
        lascl_mat(ScaleKind::UpperHessenberg, anrmto, anrm, s, k);
        lascl_slice(anrmto, anrm, alphar, k);
        lascl_slice(anrmto, anrm, alphai, k);
    }
    if let Some((bnrm, bnrmto)) = bscale {
        lascl_mat(ScaleKind::UpperTriangular, bnrmto, bnrm, t, k);
        lascl_slice(bnrmto, bnrm, beta, k);
    }
}

/// The `DGGES` final check (LASTSL/LST2SL loop): evaluates `select` once
/// per final eigenvalue, in order, and returns `SDIM` (a complex pair
/// counts 2 when either member is selected) and whether a selected
/// eigenvalue follows an unselected one (`INFO = N+2`).
pub(super) fn check_ordering<T, F>(
    eigenvalues: &[GeneralizedEigenvalue<T>],
    select: &mut F,
) -> QzOrdering
where
    T: Field + Real,
    F: FnMut(&GeneralizedEigenvalue<T>) -> bool,
{
    let mut lastsl = true;
    let mut lst2sl = true;
    let mut sdim = 0;
    let mut changed = false;
    // IP = 1 after the first member of a complex pair.
    let mut in_pair = false;
    for e in eigenvalues {
        let mut cursl = select(e);
        if e.alpha_imag == T::zero() {
            if cursl {
                sdim += 1;
            }
            in_pair = false;
            if cursl && !lastsl {
                changed = true;
            }
        } else if in_pair {
            // Last eigenvalue of conjugate pair.
            cursl = cursl || lastsl;
            lastsl = cursl;
            if cursl {
                sdim += 2;
            }
            in_pair = false;
            if cursl && !lst2sl {
                changed = true;
            }
        } else {
            // First eigenvalue of conjugate pair.
            in_pair = true;
        }
        lst2sl = lastsl;
        lastsl = cursl;
    }
    QzOrdering {
        sdim,
        selection_changed: changed,
    }
}

impl<T: Field + Real + bytemuck::Zeroable> Qz<T> {
    /// Computes the QZ decomposition with the selected eigenvalues moved to
    /// the leading diagonal positions: LAPACK `DGGES` with
    /// `JOBVSL = JOBVSR = 'V'` and `SORT = 'S'`.
    ///
    /// The steps of [`Qz::compute`] run first. `select` is then called once
    /// per generalized eigenvalue, in diagonal order, with the *unscaled*
    /// eigenvalues (`DGGES` undoes its internal norm scaling on copies
    /// before calling `SELCTG`); a complex conjugate pair is moved as a
    /// whole when `select` holds for either member. The selected blocks are
    /// moved to the top-left corner by orthogonal swaps of adjacent
    /// diagonal blocks (`DTGSEN` with `IJOB = 0`), after which the
    /// eigenvalues are recomputed from the reordered blocks and the
    /// factorization is back-permuted and unscaled as in the unsorted case.
    /// Finally `select` is evaluated once more on the final eigenvalues to
    /// count [`QzOrdering::sdim`] and to detect
    /// [`QzOrdering::selection_changed`] (`DGGES` `INFO = N+2`).
    ///
    /// The leading `sdim` columns of `Q` and `Z` span the left and right
    /// deflating subspaces of the selected eigenvalues.
    ///
    /// For a complex pair the reported triples come from `DLAG2` on the
    /// final 2×2 block (both members share `β` and `alpha_imag` of the
    /// second member is the exact negation of the first, unless the `DGGES`
    /// unscale guards had to rescale a member). An infinite eigenvalue
    /// that took part in a swap is in general no longer represented by an
    /// exact `β = 0` but by a rounding-level `β`; test finiteness relative to
    /// `|α|` (as in the example) rather than with
    /// [`GeneralizedEigenvalue::is_finite`].
    ///
    /// # Errors
    ///
    /// * [`QzReorderError::Qz`] with the errors of [`Qz::compute`] (the
    ///   selector is not called);
    /// * [`QzReorderError::SwapFailed`] if a swap is rejected as unstable
    ///   (`DTGSEN` `INFO = 1`, `DGGES` `INFO = N+3`); the factors are
    ///   discarded. Besides blocks with (nearly) equal eigenvalues, this
    ///   happens — exactly as in LAPACK — for badly unbalanced pencils:
    ///   `DGGES` scales `A` and `B` independently, and when the scaled
    ///   matrices still differ in norm by more than about `1/eps`, the
    ///   generalized Sylvester solve of every swap involving a 2×2 block is
    ///   rejected. Multiplying `A` or `B` by a scalar beforehand (which only
    ///   scales the eigenvalues) avoids this.
    ///
    /// # Example
    ///
    /// Moving the finite eigenvalues of a pencil with singular `B` to the
    /// front:
    ///
    /// ```
    /// use oxiblas_lapack::evd::{GeneralizedEigenvalue, Qz};
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 10.0]]);
    /// let b = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 0.0]]);
    ///
    /// // A swapped infinite eigenvalue carries a rounding-level β, so test
    /// // finiteness relative to |α| instead of `β == 0`.
    /// let finite =
    ///     |e: &GeneralizedEigenvalue<f64>| e.beta > 1e-10 * e.alpha_real.hypot(e.alpha_imag);
    /// let (qz, ordering) = Qz::compute_sorted(a.as_ref(), b.as_ref(), finite).unwrap();
    ///
    /// assert_eq!(ordering.sdim, 2);
    /// assert!(!ordering.selection_changed);
    /// assert!(qz.eigenvalues()[..2].iter().all(finite));
    /// assert!(!finite(&qz.eigenvalues()[2]));
    ///
    /// let (err_a, err_b) = qz.verify(a.as_ref(), b.as_ref());
    /// assert!(err_a < 1e-12 && err_b < 1e-12);
    /// ```
    pub fn compute_sorted<F>(
        a: MatRef<'_, T>,
        b: MatRef<'_, T>,
        mut select: F,
    ) -> Result<(Self, QzOrdering), QzReorderError>
    where
        F: FnMut(&GeneralizedEigenvalue<T>) -> bool,
    {
        // DGGES steps 0–6 (validation … DHGEQZ).
        let mut work = pipeline::run(a, b)?;
        let k = Consts::new();

        // Undo scaling on (copies of) the eigenvalues before selecting.
        let (mut ar, mut ai, mut be) =
            (work.alphar.clone(), work.alphai.clone(), work.beta.clone());
        if let Some((anrm, anrmto)) = work.ascale {
            lascl_slice(anrmto, anrm, &mut ar, &k);
            lascl_slice(anrmto, anrm, &mut ai, &k);
        }
        if let Some((bnrm, bnrmto)) = work.bscale {
            lascl_slice(bnrmto, bnrm, &mut be, &k);
        }
        let selected: Vec<bool> = (0..ar.len())
            .map(|j| select(&GeneralizedEigenvalue::complex(ar[j], ai[j], be[j])))
            .collect();

        // Step 7: DTGSEN(IJOB = 0) on the scaled, permuted pencil; it
        // recomputes ALPHAR/ALPHAI/BETA of the scaled pencil.
        let mut pencil = SchurPencil {
            s: &mut work.h,
            t: &mut work.t,
            q: &mut work.q,
            z: &mut work.z,
        };
        tgsen0(
            &selected,
            &mut pencil,
            &mut work.alphar,
            &mut work.alphai,
            &mut work.beta,
            &k,
        )
        .map_err(|position| QzReorderError::SwapFailed { position })?;

        // Steps 8–10: back-permutation, unscale guards, unscaling.
        let qz = work.finish();

        // Check if the reordering is correct (LASTSL/LST2SL loop).
        let ordering = check_ordering(&qz.eigenvalues, &mut select);
        Ok((qz, ordering))
    }

    /// Reorders this generalized Schur decomposition so that the selected
    /// eigenvalues occupy the leading diagonal positions: LAPACK `DTGSEN`
    /// with `IJOB = 0` (`WANTQ = WANTZ = .TRUE.`) on `S`, `T`, `Q`, `Z`,
    /// followed by the `DGGES` final selection check.
    ///
    /// `select` is called once per current eigenvalue, in diagonal order (a
    /// complex conjugate pair moves as a whole when it holds for either
    /// member), and once more per final eigenvalue to compute the returned
    /// [`QzOrdering`]. Afterwards `A = Q·S·Zᵀ` and `B = Q·T·Zᵀ` still hold
    /// with the same structure guarantees as for [`Qz::compute`], and the
    /// eigenvalues are recomputed from the reordered blocks (see
    /// [`Qz::compute_sorted`] for their representation and for infinite
    /// eigenvalues).
    ///
    /// As in `DGGES`, the swaps run on copies of `S` and `T` scaled into
    /// `[SMLNUM, BIGNUM]` (`SMLNUM = √safmin / eps`) when their largest entry
    /// lies outside that range, and the scaling is undone afterwards with
    /// the `DGGES` unscale guards. A selection that needs no swap (all or
    /// none selected) leaves `Q` and `Z` unchanged, and `S` and `T` too
    /// unless they had to be scaled.
    ///
    /// The operation is transactional: on error `self` is left exactly as
    /// it was.
    ///
    /// # Errors
    ///
    /// [`QzReorderError::SwapFailed`] if a swap is rejected as unstable
    /// (`DTGSEN` `INFO = 1`): blocks with (nearly) equal eigenvalues, or —
    /// as in LAPACK — `S` and `T` differing in norm by more than about
    /// `1/eps` when a 2×2 block has to move (see [`Qz::compute_sorted`]).
    ///
    /// # Example
    ///
    /// Moving the eigenvalues inside the unit circle to the front:
    ///
    /// ```
    /// use oxiblas_lapack::evd::{GeneralizedEigenvalue, Qz};
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[&[0.5f64, 2.0, 1.0], &[0.0, 3.0, 4.0], &[0.0, 0.0, -0.25]]);
    /// let b = Mat::<f64>::eye(3);
    /// let mut qz = Qz::compute(a.as_ref(), b.as_ref()).unwrap();
    ///
    /// let inside = |e: &GeneralizedEigenvalue<f64>| {
    ///     e.value_complex().is_some_and(|(re, im)| re.hypot(im) < 1.0)
    /// };
    /// let ordering = qz.reorder(inside).unwrap();
    ///
    /// assert_eq!(ordering.sdim, 2);
    /// assert!(!ordering.selection_changed);
    /// assert!(qz.eigenvalues()[..2].iter().all(inside));
    /// assert_eq!(qz.eigenvalues()[2].value().map(|v| (v - 3.0).abs() < 1e-12), Some(true));
    ///
    /// let (err_a, err_b) = qz.verify(a.as_ref(), b.as_ref());
    /// assert!(err_a < 1e-12 && err_b < 1e-12);
    /// ```
    pub fn reorder<F>(&mut self, mut select: F) -> Result<QzOrdering, QzReorderError>
    where
        F: FnMut(&GeneralizedEigenvalue<T>) -> bool,
    {
        let k = Consts::new();
        let n = self.n;
        let selected: Vec<bool> = self.eigenvalues.iter().map(&mut select).collect();

        // Work on copies; `self` is only replaced on success.
        let (mut s, mut t, mut q, mut z) = (
            self.s.clone(),
            self.t.clone(),
            self.q.clone(),
            self.z.clone(),
        );

        // DGGES never calls DTGSEN on a pencil whose largest entry lies
        // outside [SMLNUM, BIGNUM] (the products formed by the swaps would
        // under- or overflow): scale S and T the same way. Q and Z are
        // invariant under scalar scaling of S and T.
        let ascale = scaling_target(lange_max(&s), &k);
        if let Some((from, to)) = ascale {
            lascl_mat(ScaleKind::UpperHessenberg, from, to, &mut s, &k);
        }
        let bscale = scaling_target(lange_max(&t), &k);
        if let Some((from, to)) = bscale {
            lascl_mat(ScaleKind::UpperTriangular, from, to, &mut t, &k);
        }

        let (mut alphar, mut alphai, mut beta) =
            (vec![T::zero(); n], vec![T::zero(); n], vec![T::zero(); n]);
        let mut pencil = SchurPencil {
            s: &mut s,
            t: &mut t,
            q: &mut q,
            z: &mut z,
        };
        tgsen0(
            &selected,
            &mut pencil,
            &mut alphar,
            &mut alphai,
            &mut beta,
            &k,
        )
        .map_err(|position| QzReorderError::SwapFailed { position })?;
        undo_scaling(
            (&mut s, &mut t),
            (&mut alphar, &mut alphai, &mut beta),
            (ascale, bscale),
            &k,
        );
        let eigenvalues: Vec<GeneralizedEigenvalue<T>> = (0..n)
            .map(|j| GeneralizedEigenvalue::complex(alphar[j], alphai[j], beta[j]))
            .collect();

        // Check if the reordering is correct (LASTSL/LST2SL loop).
        let ordering = check_ordering(&eigenvalues, &mut select);
        self.s = s;
        self.t = t;
        self.q = q;
        self.z = z;
        self.eigenvalues = eigenvalues;
        Ok(ordering)
    }
}
