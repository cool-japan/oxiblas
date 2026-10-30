//! Private marshaling helpers shared by the banded, packed and complex dense
//! Level-2 CBLAS wrappers (`banded`, `packed`, `complex_level2_ext`).
//!
//! The native Level-2 kernels in [`crate::level2`] take contiguous slices and
//! column-major storage. These helpers own the CBLAS-side concerns around
//! them, following the pattern of `level2_real.rs` / `complex_level2.rs`:
//!
//! * gathering a strided (possibly negative-increment) vector into a `Vec`
//!   and scattering it back, optionally conjugating on the way (the
//!   row-major conjugate-transpose and Hermitian identities);
//! * the reference `y := beta*y` prologue and quick returns of the y-update
//!   routines (GBMV, SBMV, HBMV, SPMV, HPMV);
//! * the row-major reinterpretation of `uplo` / `trans`;
//! * overflow-checked packed-triangle lengths.
//!
//! # Increment convention
//!
//! Following reference BLAS, logical element `i` of a length-`n` vector with
//! increment `inc` lives at pointer offset `i*inc` when `inc > 0` and at
//! `(i - (n-1))*inc = (n-1-i)*|inc|` when `inc < 0`: a negative increment
//! walks the vector back-to-front from its far end, never before the pointer.
//! A zero increment is rejected by the callers (`validate::inc_valid`).

use super::types::*;
use oxiblas_core::scalar::Scalar;

/// Pointer offset (in elements) of logical index `i` of a length-`n` vector
/// accessed with increment `inc` (see the module docs).
#[inline]
pub(super) fn vec_offset(i: usize, n: usize, inc: isize) -> isize {
    if inc >= 0 {
        i as isize * inc
    } else {
        (i as isize - (n as isize - 1)) * inc
    }
}

/// Gathers a strided vector into a contiguous `Vec`, conjugating every
/// element when `conj` is set (a no-op for real `T`).
///
/// # Safety
///
/// `x` must be valid for reads at every offset `vec_offset(i, n, inc)` for
/// `i in 0..n`.
pub(super) unsafe fn gather<T: Scalar>(x: *const T, n: usize, inc: isize, conj: bool) -> Vec<T> {
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        // SAFETY: `i < n`, and the caller guarantees every such offset is
        // readable.
        let e = unsafe { *x.offset(vec_offset(i, n, inc)) };
        v.push(if conj { e.conj() } else { e });
    }
    v
}

/// Scatters `src` back to a strided vector of length `src.len()`,
/// conjugating every element when `conj` is set (a no-op for real `T`).
///
/// # Safety
///
/// `dst` must be valid for writes at every offset `vec_offset(i, n, inc)`
/// for `i in 0..n`, where `n = src.len()`.
pub(super) unsafe fn scatter<T: Scalar>(dst: *mut T, src: &[T], inc: isize, conj: bool) {
    let n = src.len();
    for (i, &e) in src.iter().enumerate() {
        // SAFETY: `i < n`, and the caller guarantees every such offset is
        // writable.
        unsafe { *dst.offset(vec_offset(i, n, inc)) = if conj { e.conj() } else { e } };
    }
}

/// Reference prologue of the y-update routines (GBMV, SBMV, HBMV, SPMV,
/// HPMV), applied to the caller's strided `y` before any native kernel runs:
///
/// * `alpha == 0 && beta == 1`: quick return, `y` is not touched at all;
/// * otherwise `y := beta*y`, where `beta == 0` zero-fills **without reading**
///   `y` (it need not hold initialized or finite values) and `beta == 1`
///   skips the pass (multiplying by `1 + 0i` would turn an infinite component
///   into NaN);
/// * then `alpha == 0` completes the call, so `A` and `x` are never read and
///   NaN/Inf in them cannot reach `y`.
///
/// Returns `true` when the `alpha*op(A)*x` term still has to be accumulated;
/// the caller then runs the native kernel with `beta = 1`, which leaves the
/// already scaled `y` unscaled.
///
/// Scaling the caller's `y` here (rather than a conjugated copy) also serves
/// the row-major Hermitian / conjugate-transpose paths: reference CBLAS
/// computes `conj(conj(beta)*conj(y))`, and conjugation commutes with IEEE
/// complex multiplication except for the sign of an exactly zero imaginary
/// part, so the result equals `beta*y` up to the sign of zero imaginary parts.
///
/// # Safety
///
/// `y` must be valid for reads and writes at every offset
/// `vec_offset(i, n, inc)` for `i in 0..n`.
pub(super) unsafe fn y_update_prologue<T: Scalar>(
    alpha: T,
    beta: T,
    y: *mut T,
    n: usize,
    inc: isize,
) -> bool {
    let zero = T::zero();
    let one = T::one();
    if alpha == zero && beta == one {
        return false;
    }
    if beta != one {
        for i in 0..n {
            // SAFETY: `i < n`, and the caller guarantees every such offset is
            // readable and writable. `beta == 0` stores without reading.
            unsafe {
                let p = y.offset(vec_offset(i, n, inc));
                *p = if beta == zero { zero } else { beta * *p };
            }
        }
    }
    alpha != zero
}

/// Returns the opposite triangle.
#[inline]
pub(super) fn flip_uplo(uplo: CblasUplo) -> CblasUplo {
    match uplo {
        CblasUplo::Upper => CblasUplo::Lower,
        CblasUplo::Lower => CblasUplo::Upper,
    }
}

/// Column-major reading `(uplo, conj)` of a symmetric / Hermitian operand.
///
/// A row-major buffer read column-major holds `A^T` in the opposite
/// triangle — for dense, band and packed storage alike. For a symmetric `A`
/// that is `A` itself, for a Hermitian `A` it is `conj(A)`. Row-major
/// therefore flips `uplo` and reports `conj = true`: the caller evaluates the
/// conjugated equation (which routine conjugates which operands follows
/// reference CBLAS and is documented at each call site). For real `T`
/// conjugation is the identity, so real callers may use the same flag.
#[inline]
pub(super) fn hermitian_view(layout: CblasLayout, uplo: CblasUplo) -> (CblasUplo, bool) {
    match layout {
        CblasLayout::ColMajor => (uplo, false),
        CblasLayout::RowMajor => (flip_uplo(uplo), true),
    }
}

/// Column-major reading `(trans, conj)` of `op(A)` for a general (GBMV) or
/// triangular operand, following reference CBLAS.
///
/// Row-major storage read column-major holds `A^T`, so `NoTrans` / `Trans`
/// swap. `op(A) = A^H = conj(A^T)` has no transpose-free native form; it is
/// evaluated as `NoTrans` on conjugated data — `conj(y) = conj(alpha) * A^T *
/// conj(x) + ...` for a product, `A^T * conj(x) = conj(b)` for a solve — so
/// `conj = true` asks the caller to conjugate the vectors (and scalars) on
/// the way in and the result on the way out, as reference `cblas_cgbmv.c` /
/// `cblas_ctbsv.c` do. For real `T` the conjugations are no-ops and the
/// mapping degenerates to reference `cblas_dgbmv.c` (`ConjTrans` -> `N`).
#[inline]
pub(super) fn transpose_view(layout: CblasLayout, trans: CblasTranspose) -> (CblasTranspose, bool) {
    match layout {
        CblasLayout::ColMajor => (trans, false),
        CblasLayout::RowMajor => match trans {
            CblasTranspose::NoTrans => (CblasTranspose::Trans, false),
            CblasTranspose::Trans => (CblasTranspose::NoTrans, false),
            CblasTranspose::ConjTrans => (CblasTranspose::NoTrans, true),
        },
    }
}

/// Column-major reading `(uplo, trans, conj)` of a triangular operand for
/// TRMV / TRSV / TBMV / TBSV / TPMV / TPSV: row-major additionally flips
/// `uplo`, because `A^T` occupies the opposite triangle (dense, band and
/// packed storage alike). See [`transpose_view`] for `trans` / `conj`.
#[inline]
pub(super) fn triangular_view(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
) -> (CblasUplo, CblasTranspose, bool) {
    let (trans, conj) = transpose_view(layout, trans);
    let uplo = match layout {
        CblasLayout::ColMajor => uplo,
        CblasLayout::RowMajor => flip_uplo(uplo),
    };
    (uplo, trans, conj)
}

/// Element count `n*(n+1)/2` of a packed triangle of order `n`, or `None`
/// when that count — or its size in bytes, which a slice may not let exceed
/// `isize::MAX` — overflows. Callers treat `None` as an invalid argument
/// (silent no-op): no real buffer can be that large.
#[inline]
pub(super) fn packed_len<T>(n: usize) -> Option<usize> {
    let len = n.checked_mul(n.checked_add(1)?)? / 2;
    let bytes = len.checked_mul(core::mem::size_of::<T>())?;
    (bytes <= isize::MAX.unsigned_abs()).then_some(len)
}

#[cfg(test)]
#[path = "strided_tests.rs"]
pub(super) mod tests;
