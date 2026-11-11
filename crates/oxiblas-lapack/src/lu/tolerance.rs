//! Scale-invariant, per-column pivot tolerances shared by the LU and Cholesky
//! families.
//!
//! A factorization reports numerical singularity (LU, LDLᵀ) or a loss of
//! positive definiteness (LLᵀ) when the pivot of column `j` is at or below
//! `eps · n · s_j`. The reference magnitude `s_j` belongs to column `j` of the
//! *input* matrix; the references of all columns are computed in one pass
//! before factorizing, stored in one vector and shared by every code path
//! (unblocked, blocked, recursive and parallel):
//!
//! - LU (partial, rook and full pivoting, band): `s_j = max_i |a_ij|` over the
//!   original column `j` (for band storage, the stored band entries of
//!   column `j`). Rook and full pivoting permute columns, so a pivot is
//!   compared with the reference of the original column it comes from.
//! - LDLᵀ (dense and packed): `s_j = max_i |a_ij|` over the whole original
//!   column `j` of the symmetric matrix, that is the stored entries of
//!   column `j` together with the mirrored entries of row `j`.
//! - LLᵀ Cholesky (dense in every variant, Hermitian, band, packed):
//!   `s_j = a_jj`, the original diagonal entry (its real part for a Hermitian
//!   matrix), against which the radicand `a_jj − Σ |l_jk|²` is tested.
//!
//! A per-column reference keeps scaling apart from conditioning. A global
//! reference such as `max |a_ij|` over the whole matrix reports
//! `diag(1e20, 1)` singular although it factors exactly, and likewise every
//! stiffness matrix that imposes a boundary condition through a penalty term
//! `K + 1e20 · e_i e_iᵀ`. LAPACK's `xGETRF` and `xPOTRF` only flag exactly
//! zero pivots (`xPOTRF`: non-positive or NaN radicands) and accept such
//! matrices.
//!
//! The outcome (success, or the error and its index), the pivots, and the
//! factors up to the matching power-of-two scaling do not change when the
//! input is rescaled by powers of two, as long as nothing overflows or
//! underflows:
//!
//! - every factorization is invariant under global scaling `A → s·A`;
//! - partial-pivoting and band LU are invariant under column scaling
//!   `A → A·diag(c)`: their pivot search compares entries of one column only,
//!   and column `j` of U and its reference both scale by `c_j`, while L does
//!   not change. Rook and full pivoting compare entries of different columns,
//!   so their pivot choice changes under column scaling;
//! - LLᵀ is invariant under symmetric scaling `A → D·A·D`: radicand and
//!   `a_jj` both scale by `d_j²`, and `L → D·L`. The LDLᵀ reference counts
//!   off-diagonal entries, which scale by `d_i·d_j` instead of `d_j²`, so LDLᵀ
//!   shares this invariance only while each column's maximum stays on its
//!   diagonal (diagonal and strongly diagonally dominant matrices). Counting
//!   them is deliberate: LDLᵀ does not pivot, and a pivot that is tiny next
//!   to its column, such as `1e-17` in `[[1e-17, 1], [1, 1e-17]]`, makes
//!   multipliers of `1e17`.
//!
//! An explicit tolerance passed by the caller is absolute and is used as
//! given for every column.
//!
//! Only finite values contribute to `s_j`, and `s_j` is never negative. A
//! NaN or ±∞ entry is skipped, so non-finite input keeps propagating through
//! the factors per IEEE-754: a NaN pivot fails every `<=` comparison and is
//! never reported as singular, and an infinite entry does not inflate a
//! threshold to ∞, which would reject every finite pivot of its column.

use num_traits::{Float, FromPrimitive, One, Zero};
use oxiblas_core::scalar::{Real, Scalar};
use oxiblas_matrix::MatRef;

/// Raises `slot` to `value` when `value` is finite and larger. NaN and ±∞
/// never count.
#[inline]
pub(crate) fn raise_finite<R: Real>(slot: &mut R, value: R) {
    if value > *slot && Float::is_finite(value) {
        *slot = value;
    }
}

/// Returns the largest finite value yielded by `values`, or zero if none is
/// larger than zero. NaN and ±∞ are skipped.
#[inline]
pub(crate) fn finite_max<R: Real>(values: impl IntoIterator<Item = R>) -> R {
    let mut max = <R as Zero>::zero();
    for value in values {
        raise_finite(&mut max, value);
    }
    max
}

/// Returns the pivot threshold `eps · n · scale`.
///
/// `eps · n` is formed first. It is far below one for any matrix that fits in
/// memory, so the product with `scale` cannot overflow even when `scale` is
/// close to the largest finite value. For a tiny `scale` the threshold may
/// round to zero, which leaves the exact-zero pivot test as the floor.
#[inline]
pub(crate) fn pivot_tolerance<R: Real>(n: usize, scale: R) -> R {
    let n_real = <R as FromPrimitive>::from_usize(n).unwrap_or(<R as One>::one());
    (<R as Scalar>::epsilon() * n_real) * scale
}

/// Turns the per-column references `s_j` into the thresholds
/// `eps · n · s_j`, in place.
pub(crate) fn into_pivot_tolerances<R: Real>(n: usize, mut scales: Vec<R>) -> Vec<R> {
    for scale in &mut scales {
        *scale = pivot_tolerance(n, *scale);
    }
    scales
}

/// Returns `max_i |a_ij|` for every column `j` of `a`, skipping non-finite
/// values. Contiguous columns are scanned as slices; other views fall back to
/// indexing.
pub(crate) fn column_max_abs_finite<T: Scalar>(a: MatRef<'_, T>) -> Vec<T::Real> {
    let nrows = a.nrows();
    (0..a.ncols())
        .map(|j| match a.col_as_slice(j) {
            Some(col) => finite_max(col.iter().map(|&x| Scalar::abs(x))),
            None => finite_max((0..nrows).map(|i| Scalar::abs(a[(i, j)]))),
        })
        .collect()
}

/// Returns `max_i |a_ij|` over every column `j` of the symmetric matrix whose
/// lower triangle (diagonal included) is stored in the square `a`: the
/// stored entries `a_ij` (`i >= j`) of column `j` and the mirrored entries
/// `a_ji` (`i < j`) of row `j`. One pass over the lower triangle, in which
/// each entry raises the references of both its column and its row;
/// non-finite values are skipped.
pub(crate) fn symmetric_column_max_abs_finite<T: Scalar>(a: MatRef<'_, T>) -> Vec<T::Real> {
    let n = a.nrows();
    let mut scales = vec![<T::Real as Zero>::zero(); n];
    for j in 0..n {
        match a.col_as_slice(j) {
            Some(col) => {
                for (i, &x) in col.iter().enumerate().skip(j) {
                    let value = Scalar::abs(x);
                    raise_finite(&mut scales[j], value);
                    raise_finite(&mut scales[i], value);
                }
            }
            None => {
                for i in j..n {
                    let value = Scalar::abs(a[(i, j)]);
                    raise_finite(&mut scales[j], value);
                    raise_finite(&mut scales[i], value);
                }
            }
        }
    }
    scales
}

/// Returns the diagonal references `max(0, Re a_jj)` of the square `a`,
/// skipping non-finite values.
pub(crate) fn diagonal_finite<T: Scalar>(a: MatRef<'_, T>) -> Vec<T::Real> {
    (0..a.nrows())
        .map(|j| finite_max([a[(j, j)].real()]))
        .collect()
}

/// Per-column LU thresholds `eps · n · max_i |a_ij|` of the square `a`.
#[inline]
pub(crate) fn lu_pivot_tolerances<T: Scalar>(a: MatRef<'_, T>) -> Vec<T::Real> {
    into_pivot_tolerances(a.nrows(), column_max_abs_finite(a))
}

/// Per-column LLᵀ thresholds `eps · n · max(0, Re a_jj)` of the square `a`.
#[inline]
pub(crate) fn llt_pivot_tolerances<T: Scalar>(a: MatRef<'_, T>) -> Vec<T::Real> {
    into_pivot_tolerances(a.nrows(), diagonal_finite(a))
}

#[cfg(test)]
#[path = "tolerance_tests.rs"]
mod tests;
