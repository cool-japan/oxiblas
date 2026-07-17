//! Shared Householder tridiagonalization of complex Hermitian matrices, WITH the
//! diagonal phase correction (LAPACK's implicit `D`) required to return *correct*
//! eigenvectors.
//!
//! Both [`HermitianEvd`](super::hermitian::HermitianEvd) (implicit-QR) and
//! [`HermitianEvdDc`](super::hermitian_dc::HermitianEvdDc) (divide-and-conquer) reduce a
//! complex Hermitian matrix to a *real* symmetric tridiagonal matrix and then run a real
//! tridiagonal eigensolver. They need the identical phase-tracking logic, so it lives here
//! once instead of being duplicated (and risking the two copies drifting apart).

use num_traits::{One, Zero};
use oxiblas_core::scalar::{ComplexScalar, Field, Real};
use oxiblas_matrix::Mat;

/// Reduces the Hermitian matrix stored (as a full Hermitian matrix) in `a` to a REAL
/// symmetric tridiagonal matrix `T`, accumulating the accompanying unitary transform —
/// **including the diagonal phase correction `D`** — into `u`.
///
/// Returns `(diag, off_diag)`: the real diagonal (length `n`) and the real, non-negative
/// off-diagonal (length `n - 1`) of `T`. On entry `u` must be the identity (or whatever
/// left factor the caller wants `Q·D` right-applied to); on exit `u == Q·D`.
///
/// # Why the phase correction is essential (the WHY, not the WHAT)
///
/// Householder reduction of a *complex* Hermitian `A` yields `A = Q·T'·Qᴴ`, where the
/// tridiagonal `T'` generally has **complex** sub-diagonal entries `e'_k` (its diagonal is
/// real because `A` is Hermitian). Real symmetric tridiagonal eigensolvers require a *real*
/// tridiagonal, so we realify `T'` with a diagonal *unitary* similarity: `T = Dᴴ·T'·D` is
/// real, where `D = diag(d_0, …, d_{n-1})`, every `|d_i| = 1`, chosen so each sub-diagonal
/// becomes real and non-negative:
///
/// ```text
/// d_0 = 1,   d_{k+1} = d_k · e'_k / |e'_k|
/// ```
///
/// (then `T[k+1,k] = conj(d_{k+1})·e'_k·d_k = |e'_k| ≥ 0`).
///
/// Consequently `A = (Q·D)·T·(Q·D)ᴴ`, so if `T = V·Λ·Vᵀ` with `V` real orthogonal, the
/// eigenvectors of `A` are the columns of `Q·D·V` — **not** `Q·V`. Dropping `D` leaves the
/// eigenVALUES correct (they are real, hence phase independent) but multiplies every
/// eigenVECTOR by a spurious per-row unit-modulus phase, so `A·u = λ·u` fails. We therefore
/// fold `D` into `u` here (`u := Q·D`); the caller's real tridiagonal solver then
/// right-multiplies by `V`, producing `Q·D·V` directly.
///
/// # Reflector convention (a non-obvious numerical invariant)
///
/// Each step uses a Hermitian reflector `H = I − τ·v·vᴴ` with a **real** `τ = 2/‖v‖²`,
/// which makes `H` both unitary *and* Hermitian. For such an `H` to map the sub-diagonal
/// column `x` to `β·e_1`, `β` must satisfy `|β| = ‖x‖` **and** `conj(β)·x_1 ∈ ℝ`. We pick
/// `β = −‖x‖ · x_1/|x_1|` (opposite phase to `x_1`), which meets both conditions and, by
/// making `v_1 = x_1 − β = (|x_1| + ‖x‖)·x_1/|x_1|` grow in magnitude, avoids the
/// cancellation that a same-phase choice would suffer. Because `H` is invariant to scaling
/// `v` by any nonzero complex scalar, we normalize `v_1 = 1`.
pub(crate) fn tridiagonalize_hermitian<T: Field + ComplexScalar>(
    a: &mut Mat<T>,
    u: &mut Mat<T>,
    n: usize,
) -> (Vec<T::Real>, Vec<T::Real>)
where
    T::Real: Real,
{
    let mut diag = vec![T::Real::zero(); n];
    let mut off_diag = vec![T::Real::zero(); n.saturating_sub(1)];

    // Diagonal phase-correction accumulator `D`; every entry has unit modulus.
    let mut d_phase: Vec<T> = vec![T::one(); n];

    // Scratch Householder vector reused across steps.
    let mut v: Vec<T> = vec![T::zero(); n];

    let two = T::Real::one() + T::Real::one();

    for k in 0..n.saturating_sub(2) {
        // x = a[k+1..n, k] — the part of column k below the diagonal.
        let mut norm_sq = T::Real::zero();
        for i in (k + 1)..n {
            norm_sq = norm_sq + a[(i, k)].abs_sq();
        }
        let xnorm = <T::Real as Real>::sqrt(norm_sq);

        if xnorm <= T::Real::zero() {
            // Column already zero below the diagonal: reflector is the identity, and the
            // sub-diagonal element is exactly zero, so the phase is carried through
            // unchanged (`off_diag[k]` stays 0).
            d_phase[k + 1] = d_phase[k];
            continue;
        }

        let x1 = a[(k + 1, k)];
        let x1_abs = x1.abs();

        // Complex sub-diagonal value the reflector produces: β = −‖x‖ · phase(x_1),
        // with |β| = ‖x‖. When x_1 == 0 the phase is undefined; β = −‖x‖ (real) works.
        let beta = if x1_abs > T::Real::zero() {
            T::from_real(-xnorm) * (x1 / T::from_real(x1_abs))
        } else {
            T::from_real(-xnorm)
        };

        // Householder vector, normalized so v[k+1] = 1 (H is scale-invariant in v).
        // v[k+1] = 1;  v[i] = x[i] / (x_1 − β)  for i > k+1.
        let vdenom = x1 - beta;
        v[k + 1] = T::one();
        for i in (k + 2)..n {
            v[i] = a[(i, k)] / vdenom;
        }

        // ‖v‖² (the v[k+1] = 1 entry contributes 1).
        let mut v_norm_sq = T::Real::one();
        for i in (k + 2)..n {
            v_norm_sq = v_norm_sq + v[i].abs_sq();
        }
        let tau = T::from_real(two / v_norm_sq);

        // Hermitian rank-2 update of the trailing block A[k+1.., k+1..]:
        //   p = τ·A·v ; w = p − (τ/2)(vᴴp)·v ; A := A − v·wᴴ − w·vᴴ.
        let mut p: Vec<T> = vec![T::zero(); n];
        for i in (k + 1)..n {
            let mut sum = T::zero();
            for j in (k + 1)..n {
                sum = sum + a[(i, j)] * v[j];
            }
            p[i] = tau * sum;
        }

        let mut vh_p = T::zero();
        for i in (k + 1)..n {
            vh_p = vh_p + v[i].conj() * p[i];
        }
        let half_tau_vhp = tau * vh_p / T::from_real(two);

        let mut w: Vec<T> = vec![T::zero(); n];
        for i in (k + 1)..n {
            w[i] = p[i] - half_tau_vhp * v[i];
        }

        for i in (k + 1)..n {
            for j in (k + 1)..n {
                a[(i, j)] = a[(i, j)] - v[i] * w[j].conj() - w[i] * v[j].conj();
            }
        }

        // Accumulate the reflector into Q:  u := u·(I − τ·v·vᴴ).
        for i in 0..n {
            let mut uv = T::zero();
            for j in (k + 1)..n {
                uv = uv + u[(i, j)] * v[j];
            }
            let tau_uv = tau * uv;
            for j in (k + 1)..n {
                u[(i, j)] = u[(i, j)] - tau_uv * v[j].conj();
            }
        }

        off_diag[k] = xnorm; // = |β|, the real sub-diagonal of T.
        // d_{k+1} = d_k · β / |β|  (unit modulus; realifies this sub-diagonal).
        d_phase[k + 1] = d_phase[k] * (beta / T::from_real(xnorm));
    }

    // Diagonal of T is real for a Hermitian matrix (and D leaves it unchanged since
    // conj(d_i)·T'[i,i]·d_i = |d_i|²·T'[i,i] = T'[i,i]).
    for i in 0..n {
        diag[i] = a[(i, i)].real();
    }

    // The final sub-diagonal e'_{n-2} = a[n-1, n-2] is produced by the trailing rank-2
    // updates (for n == 2 it is simply the input entry) and is never explicitly realified,
    // so fold its phase into D as well.
    if n >= 2 {
        let e_last = a[(n - 1, n - 2)];
        let e_last_abs = e_last.abs();
        off_diag[n - 2] = e_last_abs;
        if e_last_abs > T::Real::zero() {
            d_phase[n - 1] = d_phase[n - 2] * (e_last / T::from_real(e_last_abs));
        } else {
            d_phase[n - 1] = d_phase[n - 2];
        }
    }

    // Fold the diagonal phase correction into the accumulated transform: u := Q·D, i.e.
    // scale column j of Q by d_j. The caller's real tridiagonal solver then right-multiplies
    // by V, yielding the correct A-eigenvectors Q·D·V.
    for j in 0..n {
        let dj = d_phase[j];
        for i in 0..n {
            u[(i, j)] = u[(i, j)] * dj;
        }
    }

    (diag, off_diag)
}
