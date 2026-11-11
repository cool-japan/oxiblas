//! `DHGEQZ` (JOB = 'S', COMPQ = COMPZ = 'V'): the QZ iteration on a
//! Hessenberg–triangular pencil.
//!
//! This file holds the main loop of the reference routine: the splitting
//! tests (negligible subdiagonal of `H`, negligible diagonal of `T`), the
//! zero chasing that deflates infinite eigenvalues (labels 40, 50, 70), the
//! 1×1 standardization (label 80) and the iteration bookkeeping (`MAXIT`,
//! `IITER`, the exceptional-shift counter). Shift selection and the QZ
//! sweeps live in `hgeqz_sweep`.
//!
//! Schur form is always computed, so in the reference's notation
//! `IFRSTM = 1` and `ILASTM = N` (here `0` and `n − 1`): row operations
//! touch columns up to `n − 1` and column operations start at row `0`.

use num_traits::Float;
use oxiblas_core::scalar::{Field, Real, Scalar};

use super::hgeqz_sweep::{
    Shift, double_shift_sweep, select_shift, single_shift_sweep, standardize_2x2,
};
use super::pipeline::QzWork;
use super::types::QzError;
use super::util::{Consts, lanhs_fro, lartg, rot_cols, rot_rows};

/// Tolerances and scale factors fixed for a whole `DHGEQZ` run.
#[derive(Debug, Clone, Copy)]
pub(super) struct QzTol<T> {
    /// `ATOL = max(safmin, ulp · ‖H(ilo..=ihi, ilo..=ihi)‖_F)`.
    pub(super) atol: T,
    /// `BTOL = max(safmin, ulp · ‖T(ilo..=ihi, ilo..=ihi)‖_F)`.
    pub(super) btol: T,
    /// `ASCALE = 1 / max(safmin, ‖H(ilo..=ihi, ilo..=ihi)‖_F)`.
    pub(super) ascale: T,
    /// `BSCALE = 1 / max(safmin, ‖T(ilo..=ihi, ilo..=ihi)‖_F)`.
    pub(super) bscale: T,
}

/// Outcome of the splitting tests at the top of a `DHGEQZ` pass.
enum Split {
    /// `H(ilast, ilast−1) = 0` (or `ilast = ilo`): deflate a 1×1 block
    /// (label 80).
    Deflate,
    /// `T(ilast, ilast) = 0`: clear `H(ilast, ilast−1)` (label 70), then
    /// deflate an infinite eigenvalue (label 80).
    ZeroT,
    /// Run a QZ sweep on rows/columns `ifirst..=ilast` (label 110).
    Iterate(usize),
}

/// `|H(j, j−1)| ≤ max(safmin, ulp · (|H(j,j)| + |H(j−1,j−1)|))`.
fn subdiag_negligible<T: Field + Real>(w: &QzWork<T>, j: usize, k: &Consts<T>) -> bool {
    let bound = k.eps * (Scalar::abs(w.h[(j, j)]) + Scalar::abs(w.h[(j - 1, j - 1)]));
    Scalar::abs(w.h[(j, j - 1)]) <= Float::max(k.safmin, bound)
}

/// Label 80 (and loops 30/410): makes `T(j,j) ≥ 0` by negating column `j`
/// of `H` and `T` (rows `0..=j`) and of `Z`, then records the real
/// eigenvalue `(H(j,j), 0, T(j,j))`.
fn standardize_1x1<T: Field + Real>(w: &mut QzWork<T>, j: usize) {
    let n = w.h.nrows();
    if w.t[(j, j)] < T::zero() {
        for jr in 0..=j {
            w.h[(jr, j)] = -w.h[(jr, j)];
            w.t[(jr, j)] = -w.t[(jr, j)];
        }
        for jr in 0..n {
            w.z[(jr, j)] = -w.z[(jr, j)];
        }
    }
    w.alphar[j] = w.h[(j, j)];
    w.alphai[j] = T::zero();
    w.beta[j] = w.t[(j, j)];
}

/// Label 70: `T(ilast, ilast) = 0`; clears `H(ilast, ilast−1)` with a
/// rotation of columns `ilast, ilast−1` so that a 1×1 block splits off.
fn clear_bottom_subdiag<T: Field + Real>(w: &mut QzWork<T>, k: &Consts<T>, ilast: usize) {
    let n = w.h.nrows();
    let temp = w.h[(ilast, ilast)];
    let (c, s, r) = lartg(temp, w.h[(ilast, ilast - 1)], k);
    w.h[(ilast, ilast)] = r;
    w.h[(ilast, ilast - 1)] = T::zero();
    // DROT(ILAST−IFRSTM, H(IFRSTM,ILAST), 1, H(IFRSTM,ILAST−1), 1, C, S)
    rot_cols(&mut w.h, ilast, ilast - 1, 0..ilast, c, s);
    // DROT(ILAST−IFRSTM, T(IFRSTM,ILAST), 1, T(IFRSTM,ILAST−1), 1, C, S)
    rot_cols(&mut w.t, ilast, ilast - 1, 0..ilast, c, s);
    // DROT(N, Z(1,ILAST), 1, Z(1,ILAST−1), 1, C, S)
    rot_cols(&mut w.z, ilast, ilast - 1, 0..n, c, s);
}

/// Loop 40: `T(j,j) = 0` and `H(j,j−1)` is (or, with `ilazr2`, is treated
/// as) negligible. Rotations of rows `jch, jch+1` push the zero diagonal
/// entry of `T` down until the leading part splits off; returns where the
/// pass continues (label 80, 110 or 70).
fn chase_zero_to_split<T: Field + Real>(
    w: &mut QzWork<T>,
    tol: &QzTol<T>,
    k: &Consts<T>,
    j: usize,
    ilast: usize,
    mut ilazr2: bool,
) -> Split {
    let n = w.h.nrows();
    for jch in j..ilast {
        let temp = w.h[(jch, jch)];
        let (c, s, r) = lartg(temp, w.h[(jch + 1, jch)], k);
        w.h[(jch, jch)] = r;
        w.h[(jch + 1, jch)] = T::zero();
        // DROT(ILASTM−JCH, H(JCH,JCH+1), LDH, H(JCH+1,JCH+1), LDH, C, S)
        rot_rows(&mut w.h, jch, jch + 1, (jch + 1)..n, c, s);
        // DROT(ILASTM−JCH, T(JCH,JCH+1), LDT, T(JCH+1,JCH+1), LDT, C, S)
        rot_rows(&mut w.t, jch, jch + 1, (jch + 1)..n, c, s);
        // DROT(N, Q(1,JCH), 1, Q(1,JCH+1), 1, C, S)
        rot_cols(&mut w.q, jch, jch + 1, 0..n, c, s);
        if ilazr2 {
            // The fill-in −s·H(jch, jch−1) below the subdiagonal is
            // negligible by the ILAZR2 test and is dropped.
            w.h[(jch, jch - 1)] = w.h[(jch, jch - 1)] * c;
        }
        ilazr2 = false;
        if Scalar::abs(w.t[(jch + 1, jch + 1)]) >= tol.btol {
            return if jch + 1 >= ilast {
                Split::Deflate
            } else {
                Split::Iterate(jch + 1)
            };
        }
        w.t[(jch + 1, jch + 1)] = T::zero();
    }
    Split::ZeroT
}

/// Loop 50: only `T(j,j) = 0` holds (`j > ilo`). The zero is chased down to
/// `T(ilast, ilast)` by alternating row rotations (keeping `T` triangular)
/// and column rotations (restoring the Hessenberg form of `H`).
fn chase_zero_to_bottom<T: Field + Real>(w: &mut QzWork<T>, k: &Consts<T>, j: usize, ilast: usize) {
    let n = w.h.nrows();
    for jch in j..ilast {
        let temp = w.t[(jch, jch + 1)];
        let (c, s, r) = lartg(temp, w.t[(jch + 1, jch + 1)], k);
        w.t[(jch, jch + 1)] = r;
        w.t[(jch + 1, jch + 1)] = T::zero();
        // DROT(ILASTM−JCH−1, T(JCH,JCH+2), LDT, T(JCH+1,JCH+2), LDT, C, S)
        rot_rows(&mut w.t, jch, jch + 1, (jch + 2)..n, c, s);
        // DROT(ILASTM−JCH+2, H(JCH,JCH−1), LDH, H(JCH+1,JCH−1), LDH, C, S)
        rot_rows(&mut w.h, jch, jch + 1, (jch - 1)..n, c, s);
        // DROT(N, Q(1,JCH), 1, Q(1,JCH+1), 1, C, S)
        rot_cols(&mut w.q, jch, jch + 1, 0..n, c, s);

        let temp = w.h[(jch + 1, jch)];
        let (c, s, r) = lartg(temp, w.h[(jch + 1, jch - 1)], k);
        w.h[(jch + 1, jch)] = r;
        w.h[(jch + 1, jch - 1)] = T::zero();
        // DROT(JCH+1−IFRSTM, H(IFRSTM,JCH), 1, H(IFRSTM,JCH−1), 1, C, S)
        rot_cols(&mut w.h, jch, jch - 1, 0..(jch + 1), c, s);
        // DROT(JCH−IFRSTM, T(IFRSTM,JCH), 1, T(IFRSTM,JCH−1), 1, C, S)
        rot_cols(&mut w.t, jch, jch - 1, 0..jch, c, s);
        // DROT(N, Z(1,JCH), 1, Z(1,JCH−1), 1, C, S)
        rot_cols(&mut w.z, jch, jch - 1, 0..n, c, s);
    }
}

/// Splitting tests of one `DHGEQZ` pass (labels before 60 and loop 60).
///
/// Test 1: `H(j, j−1)` negligible or `j = ilo`; test 2: `T(j,j)`
/// negligible. Returns `None` for the "impossible" drop-through of the
/// reference (`INFO = N+1`).
fn find_split<T: Field + Real>(
    w: &mut QzWork<T>,
    tol: &QzTol<T>,
    k: &Consts<T>,
    ilo: usize,
    ilast: usize,
) -> Option<Split> {
    // Special case: j = ilast.
    if ilast == ilo {
        return Some(Split::Deflate);
    }
    if subdiag_negligible(w, ilast, k) {
        w.h[(ilast, ilast - 1)] = T::zero();
        return Some(Split::Deflate);
    }
    if Scalar::abs(w.t[(ilast, ilast)]) <= tol.btol {
        w.t[(ilast, ilast)] = T::zero();
        return Some(Split::ZeroT);
    }

    // General case: j < ilast.
    for j in (ilo..ilast).rev() {
        // Test 1: H(j, j−1) = 0 or j = ilo.
        let ilazro = if j == ilo {
            true
        } else if subdiag_negligible(w, j, k) {
            w.h[(j, j - 1)] = T::zero();
            true
        } else {
            false
        };

        // Test 2: T(j, j) = 0.
        if Scalar::abs(w.t[(j, j)]) < tol.btol {
            w.t[(j, j)] = T::zero();

            // Test 1a: two consecutive small subdiagonals in H.
            let mut ilazr2 = false;
            if !ilazro {
                let mut temp = Scalar::abs(w.h[(j, j - 1)]);
                let mut temp2 = Scalar::abs(w.h[(j, j)]);
                let tempr = Float::max(temp, temp2);
                if tempr < T::one() && tempr != T::zero() {
                    temp = temp / tempr;
                    temp2 = temp2 / tempr;
                }
                if temp * (tol.ascale * Scalar::abs(w.h[(j + 1, j)]))
                    <= temp2 * (tol.ascale * tol.atol)
                {
                    ilazr2 = true;
                }
            }

            // Both tests: split a 1×1 block off at the top (loop 40);
            // only test 2: chase the zero to T(ilast, ilast) (loop 50).
            if ilazro || ilazr2 {
                return Some(chase_zero_to_split(w, tol, k, j, ilast, ilazr2));
            }
            chase_zero_to_bottom(w, k, j, ilast);
            return Some(Split::ZeroT);
        } else if ilazro {
            // Only test 1 passed: work on j..=ilast.
            return Some(Split::Iterate(j));
        }
        // Neither test passed: try the next j.
    }
    None
}

/// `DHGEQZ('S', 'V', 'V')` on `w.h`/`w.t` (Hessenberg–triangular on entry),
/// updating `w.q` and `w.z` and writing the eigenvalue triples.
///
/// Returns `NotConverged { iterations: 30·(ihi−ilo+1) }` when the pass
/// budget `MAXIT` is exhausted (`DGGES` `INFO = 1..N`) or for the
/// "impossible" drop-through (`INFO = N+1`).
pub(super) fn hgeqz<T: Field + Real + bytemuck::Zeroable>(
    w: &mut QzWork<T>,
    k: &Consts<T>,
) -> Result<(), QzError> {
    let n = w.h.nrows();
    let (ilo, ihi) = (w.ilo, w.ihi);

    // Set eigenvalues ihi+1..n (loop 30).
    for j in (ihi + 1)..n {
        standardize_1x1(w, j);
    }

    // `DGGBAL` never produces ihi < ilo, but the reference skips the QZ
    // steps in that case (GO TO 380).
    if ihi >= ilo {
        // Tolerances from the active block (DLANHS('F') of H and T).
        let anorm = lanhs_fro(&w.h, ilo, ihi);
        let bnorm = lanhs_fro(&w.t, ilo, ihi);
        let tol = QzTol {
            atol: Float::max(k.safmin, k.eps * anorm),
            btol: Float::max(k.safmin, k.eps * bnorm),
            ascale: T::one() / Float::max(k.safmin, anorm),
            bscale: T::one() / Float::max(k.safmin, bnorm),
        };
        qz_iterate(w, &tol, k)?;
    }

    // Set eigenvalues 0..ilo (loop 410).
    for j in 0..ilo {
        standardize_1x1(w, j);
    }
    Ok(())
}

/// Main QZ iteration loop of `DHGEQZ` (loop 360) on the active block
/// `ilo..=ihi`.
fn qz_iterate<T: Field + Real>(
    w: &mut QzWork<T>,
    tol: &QzTol<T>,
    k: &Consts<T>,
) -> Result<(), QzError> {
    let (ilo, ihi) = (w.ilo, w.ihi);
    // MAXIT counts every pass of the loop, deflation passes included.
    let maxit = 30 * (ihi - ilo + 1);
    let not_converged = QzError::NotConverged { iterations: maxit };
    let maxit_t = T::from_usize(maxit).unwrap_or_else(<T as Scalar>::max_value);

    // Eigenvalues ilast+1..n have been found; IITER counts passes since
    // the last deflation (for the exceptional shift).
    let mut ilast = ihi;
    let mut iiter = 0usize;
    let mut eshift = T::zero();

    for _jiter in 0..maxit {
        let split = find_split(w, tol, k, ilo, ilast).ok_or(not_converged)?;
        let deflate_1x1 = match split {
            Split::Deflate => true,
            Split::ZeroT => {
                clear_bottom_subdiag(w, k, ilast);
                true
            }
            Split::Iterate(ifirst) => {
                // Label 110: QZ step on ifirst..=ilast (T diagonal entries
                // there are at least BTOL in magnitude).
                iiter += 1;
                match select_shift(w, k, ilast, iiter, maxit_t, &mut eshift) {
                    Shift::Real { s1, wr } => {
                        single_shift_sweep(w, tol, k, ifirst, ilast, s1, wr);
                    }
                    Shift::Complex if ifirst + 1 == ilast => {
                        if standardize_2x2(w, k, ilast) {
                            // A complex pair deflated: go to the next block.
                            if ifirst == ilo {
                                return Ok(());
                            }
                            ilast = ifirst - 1;
                            iiter = 0;
                            eshift = T::zero();
                        }
                    }
                    Shift::Complex => double_shift_sweep(w, tol, k, ifirst, ilast),
                }
                false
            }
        };
        if deflate_1x1 {
            // Label 80: standardize B, record the eigenvalue, next block.
            standardize_1x1(w, ilast);
            if ilast == ilo {
                return Ok(());
            }
            ilast -= 1;
            iiter = 0;
            eshift = T::zero();
        }
    }
    // Drop-through: non-convergence (INFO = ILAST).
    Err(not_converged)
}
