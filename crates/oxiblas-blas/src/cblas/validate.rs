//! Shared parameter validation for the CBLAS C-ABI surface.
//!
//! # Why every `extern "C"` entry point must validate
//!
//! The CBLAS entry points take raw pointers plus caller-supplied leading
//! dimensions and lengths, then index the buffers as `a.add(row + col * lda)`.
//! Two classes of bad input turn that indexing into out-of-bounds memory
//! access:
//!
//! * **Negative leading dimensions / lengths.** `lda as usize` maps `-1` to
//!   `usize::MAX`, so a single `a.add(col * lda)` walks arbitrarily far from the
//!   buffer. The same applies to a negative `k` used as a loop bound.
//! * **Undersized (but positive) leading dimensions.** With `lda = 1` and a
//!   100x100 `A`, distinct `(row, col)` pairs alias each other and the last
//!   column runs past the end of the allocation.
//!
//! Reference BLAS handles both by calling `xerbla` and returning without
//! touching the operands. These helpers reproduce that contract: they return
//! `false` and the caller **no-ops** (value-returning routines return zero).
//! Panicking is not an option — unwinding across an `extern "C"` boundary is
//! undefined behavior.
//!
//! Null-pointer checks live at the call sites (`is_null()`), because which
//! pointers a routine dereferences varies per routine.

use super::types::*;

/// Minimum leading dimension for a general `rows`x`cols` operand.
///
/// For column-major storage the leading dimension must span a full column
/// (`rows`); for row-major it must span a full row (`cols`). Reference BLAS
/// also imposes a floor of 1 so that a degenerate extent still has a usable
/// stride.
#[inline]
fn min_ld(layout: CblasLayout, rows: i32, cols: i32) -> i32 {
    match layout {
        CblasLayout::ColMajor => rows.max(1),
        CblasLayout::RowMajor => cols.max(1),
    }
}

/// Validates the leading dimension of the `m`x`n` matrix in a GEMV call.
///
/// Reference (x)GEMV requires `lda >= max(1, m)` for column-major storage and
/// `lda >= max(1, n)` for row-major storage.
#[inline]
pub(super) fn gemv_params_valid(layout: CblasLayout, m: i32, n: i32, lda: i32) -> bool {
    lda >= min_ld(layout, m, n)
}

/// Validates the shape parameters of a GEMM call.
///
/// Returns `false` when any of these reference-BLAS constraints is violated:
/// * `k < 0` — a negative contraction length, which otherwise becomes a huge
///   `usize` and drives an unbounded / out-of-bounds accumulation loop.
/// * `lda`/`ldb`/`ldc` smaller than the minimum leading dimension the storage
///   order requires.
///
/// `m`/`n` are assumed already validated (`> 0`) by the caller.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn gemm_params_valid(
    layout: CblasLayout,
    transa: CblasTranspose,
    transb: CblasTranspose,
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
) -> bool {
    if k < 0 {
        return false;
    }
    let is_notrans = |t: CblasTranspose| matches!(t, CblasTranspose::NoTrans);
    // Stored A is m×k (NoTrans) or k×m (Trans); B is k×n or n×k; C is m×n.
    let (a_rows, a_cols) = if is_notrans(transa) { (m, k) } else { (k, m) };
    let (b_rows, b_cols) = if is_notrans(transb) { (k, n) } else { (n, k) };
    lda >= min_ld(layout, a_rows, a_cols)
        && ldb >= min_ld(layout, b_rows, b_cols)
        && ldc >= min_ld(layout, m, n)
}

/// Validates a HEMM / SYMM call (`side` selects which operand `A` multiplies).
///
/// `A` is square of order `ka = m` (`Left`) or `ka = n` (`Right`), so its
/// leading dimension must be at least `max(1, ka)` in *either* storage order.
/// `B` and `C` are both `m`x`n`.
///
/// `m`/`n` are assumed already validated (`> 0`) by the caller.
#[inline]
pub(super) fn symm_params_valid(
    layout: CblasLayout,
    side: CblasSide,
    m: i32,
    n: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
) -> bool {
    let ka = match side {
        CblasSide::Left => m,
        CblasSide::Right => n,
    };
    let min_bc = min_ld(layout, m, n);
    lda >= ka.max(1) && ldb >= min_bc && ldc >= min_bc
}

/// Validates a HERK / SYRK call.
///
/// `C` is square of order `n`. `A` is `n`x`k` for the non-transposed form and
/// `k`x`n` otherwise. A negative `k` is rejected outright.
#[inline]
pub(super) fn syrk_params_valid(
    layout: CblasLayout,
    trans: CblasTranspose,
    n: i32,
    k: i32,
    lda: i32,
    ldc: i32,
) -> bool {
    if k < 0 {
        return false;
    }
    let (a_rows, a_cols) = if matches!(trans, CblasTranspose::NoTrans) {
        (n, k)
    } else {
        (k, n)
    };
    lda >= min_ld(layout, a_rows, a_cols) && ldc >= n.max(1)
}

/// Validates a HER2K / SYR2K call.
///
/// Identical to [`syrk_params_valid`] with a second operand `B` that has the
/// same shape (and therefore the same minimum leading dimension) as `A`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn syr2k_params_valid(
    layout: CblasLayout,
    trans: CblasTranspose,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
) -> bool {
    if k < 0 {
        return false;
    }
    let (a_rows, a_cols) = if matches!(trans, CblasTranspose::NoTrans) {
        (n, k)
    } else {
        (k, n)
    };
    let min_ab = min_ld(layout, a_rows, a_cols);
    lda >= min_ab && ldb >= min_ab && ldc >= n.max(1)
}

/// Validates a TRMM / TRSM call.
///
/// `A` is triangular of order `ka = m` (`Left`) or `ka = n` (`Right`) — square,
/// so `lda >= max(1, ka)` in either storage order — and `B` is `m`x`n`.
///
/// `m`/`n` are assumed already validated (`> 0`) by the caller.
#[inline]
pub(super) fn trmm_params_valid(
    layout: CblasLayout,
    side: CblasSide,
    m: i32,
    n: i32,
    lda: i32,
    ldb: i32,
) -> bool {
    let ka = match side {
        CblasSide::Left => m,
        CblasSide::Right => n,
    };
    lda >= ka.max(1) && ldb >= min_ld(layout, m, n)
}

/// Validates a HEMV / SYMV call: `A` is square of order `n`, so `lda` must be
/// at least `max(1, n)` in either storage order.
#[inline]
pub(super) fn square_lda_valid(n: i32, lda: i32) -> bool {
    lda >= n.max(1)
}

/// Validates a BLAS-1 vector increment.
///
/// The reference contract allows a negative increment (the vector is traversed
/// back-to-front) but **not** zero: `inc == 0` makes every logical element
/// alias element 0, which is meaningless for the mutating routines and is
/// rejected by reference BLAS. `n <= 0` is handled separately by the callers,
/// which return the identity result.
#[inline]
pub(super) fn inc_valid(inc: i32) -> bool {
    inc != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_lda_rejected_everywhere() {
        assert!(!gemv_params_valid(CblasLayout::ColMajor, 100, 100, -1));
        assert!(!symm_params_valid(
            CblasLayout::ColMajor,
            CblasSide::Left,
            100,
            100,
            -1,
            100,
            100
        ));
        assert!(!syrk_params_valid(
            CblasLayout::ColMajor,
            CblasTranspose::NoTrans,
            100,
            10,
            -1,
            100
        ));
        assert!(!syr2k_params_valid(
            CblasLayout::ColMajor,
            CblasTranspose::NoTrans,
            100,
            10,
            100,
            -1,
            100
        ));
        assert!(!trmm_params_valid(
            CblasLayout::ColMajor,
            CblasSide::Left,
            100,
            100,
            -1,
            100
        ));
        assert!(!square_lda_valid(100, -1));
    }

    #[test]
    fn undersized_lda_rejected() {
        // lda = 1 with a 100x100 A aliases columns and overruns the buffer.
        assert!(!symm_params_valid(
            CblasLayout::ColMajor,
            CblasSide::Left,
            100,
            100,
            1,
            100,
            100
        ));
        assert!(!syrk_params_valid(
            CblasLayout::ColMajor,
            CblasTranspose::NoTrans,
            100,
            50,
            1,
            100
        ));
        assert!(!trmm_params_valid(
            CblasLayout::RowMajor,
            CblasSide::Right,
            100,
            100,
            1,
            100
        ));
    }

    #[test]
    fn negative_k_rejected() {
        assert!(!syrk_params_valid(
            CblasLayout::ColMajor,
            CblasTranspose::NoTrans,
            10,
            -1,
            10,
            10
        ));
        assert!(!syr2k_params_valid(
            CblasLayout::ColMajor,
            CblasTranspose::NoTrans,
            10,
            -1,
            10,
            10,
            10
        ));
        assert!(!gemm_params_valid(
            CblasLayout::ColMajor,
            CblasTranspose::NoTrans,
            CblasTranspose::NoTrans,
            10,
            10,
            -1,
            10,
            10,
            10
        ));
    }

    #[test]
    fn valid_params_accepted() {
        assert!(symm_params_valid(
            CblasLayout::ColMajor,
            CblasSide::Left,
            4,
            3,
            4,
            4,
            4
        ));
        assert!(symm_params_valid(
            CblasLayout::RowMajor,
            CblasSide::Right,
            4,
            3,
            3,
            3,
            3
        ));
        assert!(syrk_params_valid(
            CblasLayout::ColMajor,
            CblasTranspose::ConjTrans,
            4,
            6,
            6,
            4
        ));
        assert!(syrk_params_valid(
            CblasLayout::RowMajor,
            CblasTranspose::ConjTrans,
            4,
            6,
            4,
            4
        ));
        assert!(trmm_params_valid(
            CblasLayout::ColMajor,
            CblasSide::Left,
            5,
            2,
            5,
            5
        ));
        assert!(square_lda_valid(0, 1));
        assert!(inc_valid(-1));
        assert!(!inc_valid(0));
    }
}
