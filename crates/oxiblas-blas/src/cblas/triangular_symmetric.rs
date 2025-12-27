//! CBLAS-compatible interface for BLAS-TESTER compatibility.
//!
//! This module provides a C-compatible interface that follows the standard
//! CBLAS specification, enabling interoperability with BLAS test suites
//! like BLAS-TESTER.
//!
//! # Layout
//!
//! CBLAS supports both row-major and column-major layouts. This library
//! uses column-major (Fortran) layout internally, so row-major operations
//! are converted using the identity: op(A) in row-major = op(A^T) in column-major.
//!
//! # Performance
//!
//! For unit stride vectors (incx=1, incy=1), this module uses the optimized
//! internal BLAS implementations with SIMD acceleration. For non-unit strides,
//! scalar fallbacks are used.

use super::types::*;
use crate::level3;

// Level 3 BLAS - TRSM (Triangular Solve with Multiple Right-Hand Sides)
// =============================================================================

/// Double precision TRSM: op(A) * X = alpha * B or X * op(A) = alpha * B.
///
/// Solves a triangular matrix equation. On exit, B is overwritten with X.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dtrsm(
    layout: CblasLayout,
    side: CblasSide,
    uplo: CblasUplo,
    transa: CblasTranspose,
    diag: CblasDiag,
    m: i32,
    n: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    b: *mut f64,
    ldb: i32,
) {
    if m <= 0 || n <= 0 {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;

    // Convert CBLAS parameters to internal format
    let (side_internal, uplo_internal, trans_internal, diag_internal) = match layout {
        CblasLayout::ColMajor => (
            match side {
                CblasSide::Left => level3::Side::Left,
                CblasSide::Right => level3::Side::Right,
            },
            match uplo {
                CblasUplo::Upper => level3::Uplo::Upper,
                CblasUplo::Lower => level3::Uplo::Lower,
            },
            match transa {
                CblasTranspose::NoTrans => level3::Trans::NoTrans,
                CblasTranspose::Trans => level3::Trans::Trans,
                CblasTranspose::ConjTrans => level3::Trans::ConjTrans,
            },
            match diag {
                CblasDiag::NonUnit => level3::Diag::NonUnit,
                CblasDiag::Unit => level3::Diag::Unit,
            },
        ),
        CblasLayout::RowMajor => {
            // Row-major: swap Left/Right and Upper/Lower
            (
                match side {
                    CblasSide::Left => level3::Side::Right,
                    CblasSide::Right => level3::Side::Left,
                },
                match uplo {
                    CblasUplo::Upper => level3::Uplo::Lower,
                    CblasUplo::Lower => level3::Uplo::Upper,
                },
                match transa {
                    CblasTranspose::NoTrans => level3::Trans::NoTrans,
                    CblasTranspose::Trans => level3::Trans::Trans,
                    CblasTranspose::ConjTrans => level3::Trans::ConjTrans,
                },
                match diag {
                    CblasDiag::NonUnit => level3::Diag::NonUnit,
                    CblasDiag::Unit => level3::Diag::Unit,
                },
            )
        }
    };

    // Determine A dimensions based on side
    let k = match side_internal {
        level3::Side::Left => m,
        level3::Side::Right => n,
    };

    use oxiblas_matrix::{MatMut, MatRef};

    // Create matrix views
    let a_ref = MatRef::<f64>::new(a, k, k, lda);
    let (bm, bn) = match layout {
        CblasLayout::ColMajor => (m, n),
        CblasLayout::RowMajor => (n, m), // Swap for row-major
    };
    let b_mut = MatMut::<f64>::new(b, bm, bn, ldb);

    // Scale B by alpha first
    for j in 0..bn {
        for i in 0..bm {
            let bp = b.add(i + j * ldb);
            *bp *= alpha;
        }
    }

    // Call internal TRSM
    let _ = level3::trsm_in_place(
        side_internal,
        uplo_internal,
        trans_internal,
        diag_internal,
        a_ref,
        b_mut,
    );
}

/// Single precision TRSM: op(A) * X = alpha * B or X * op(A) = alpha * B.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_strsm(
    layout: CblasLayout,
    side: CblasSide,
    uplo: CblasUplo,
    transa: CblasTranspose,
    diag: CblasDiag,
    m: i32,
    n: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    b: *mut f32,
    ldb: i32,
) {
    if m <= 0 || n <= 0 {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;

    let (side_internal, uplo_internal, trans_internal, diag_internal) = match layout {
        CblasLayout::ColMajor => (
            match side {
                CblasSide::Left => level3::Side::Left,
                CblasSide::Right => level3::Side::Right,
            },
            match uplo {
                CblasUplo::Upper => level3::Uplo::Upper,
                CblasUplo::Lower => level3::Uplo::Lower,
            },
            match transa {
                CblasTranspose::NoTrans => level3::Trans::NoTrans,
                CblasTranspose::Trans => level3::Trans::Trans,
                CblasTranspose::ConjTrans => level3::Trans::ConjTrans,
            },
            match diag {
                CblasDiag::NonUnit => level3::Diag::NonUnit,
                CblasDiag::Unit => level3::Diag::Unit,
            },
        ),
        CblasLayout::RowMajor => (
            match side {
                CblasSide::Left => level3::Side::Right,
                CblasSide::Right => level3::Side::Left,
            },
            match uplo {
                CblasUplo::Upper => level3::Uplo::Lower,
                CblasUplo::Lower => level3::Uplo::Upper,
            },
            match transa {
                CblasTranspose::NoTrans => level3::Trans::NoTrans,
                CblasTranspose::Trans => level3::Trans::Trans,
                CblasTranspose::ConjTrans => level3::Trans::ConjTrans,
            },
            match diag {
                CblasDiag::NonUnit => level3::Diag::NonUnit,
                CblasDiag::Unit => level3::Diag::Unit,
            },
        ),
    };

    let k = match side_internal {
        level3::Side::Left => m,
        level3::Side::Right => n,
    };

    use oxiblas_matrix::{MatMut, MatRef};

    let a_ref = MatRef::<f32>::new(a, k, k, lda);
    let (bm, bn) = match layout {
        CblasLayout::ColMajor => (m, n),
        CblasLayout::RowMajor => (n, m),
    };
    let b_mut = MatMut::<f32>::new(b, bm, bn, ldb);

    for j in 0..bn {
        for i in 0..bm {
            let bp = b.add(i + j * ldb);
            *bp *= alpha;
        }
    }

    let _ = level3::trsm_in_place(
        side_internal,
        uplo_internal,
        trans_internal,
        diag_internal,
        a_ref,
        b_mut,
    );
}

// =============================================================================
// Level 3 BLAS - TRMM (Triangular Matrix-Matrix Multiply)
// =============================================================================

/// Double precision TRMM: B = alpha * op(A) * B or B = alpha * B * op(A).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dtrmm(
    layout: CblasLayout,
    side: CblasSide,
    uplo: CblasUplo,
    transa: CblasTranspose,
    diag: CblasDiag,
    m: i32,
    n: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    b: *mut f64,
    ldb: i32,
) {
    use crate::level3::{TrmmDiag, TrmmSide, TrmmTrans, TrmmUplo};

    if m <= 0 || n <= 0 {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;

    let (side_internal, uplo_internal, trans_internal, diag_internal) = match layout {
        CblasLayout::ColMajor => (
            match side {
                CblasSide::Left => TrmmSide::Left,
                CblasSide::Right => TrmmSide::Right,
            },
            match uplo {
                CblasUplo::Upper => TrmmUplo::Upper,
                CblasUplo::Lower => TrmmUplo::Lower,
            },
            match transa {
                CblasTranspose::NoTrans => TrmmTrans::NoTrans,
                CblasTranspose::Trans => TrmmTrans::Trans,
                CblasTranspose::ConjTrans => TrmmTrans::ConjTrans,
            },
            match diag {
                CblasDiag::NonUnit => TrmmDiag::NonUnit,
                CblasDiag::Unit => TrmmDiag::Unit,
            },
        ),
        CblasLayout::RowMajor => (
            match side {
                CblasSide::Left => TrmmSide::Right,
                CblasSide::Right => TrmmSide::Left,
            },
            match uplo {
                CblasUplo::Upper => TrmmUplo::Lower,
                CblasUplo::Lower => TrmmUplo::Upper,
            },
            match transa {
                CblasTranspose::NoTrans => TrmmTrans::NoTrans,
                CblasTranspose::Trans => TrmmTrans::Trans,
                CblasTranspose::ConjTrans => TrmmTrans::ConjTrans,
            },
            match diag {
                CblasDiag::NonUnit => TrmmDiag::NonUnit,
                CblasDiag::Unit => TrmmDiag::Unit,
            },
        ),
    };

    let k = match side_internal {
        TrmmSide::Left => m,
        TrmmSide::Right => n,
    };

    use oxiblas_matrix::{MatMut, MatRef};

    let a_ref = MatRef::<f64>::new(a, k, k, lda);
    let (bm, bn) = match layout {
        CblasLayout::ColMajor => (m, n),
        CblasLayout::RowMajor => (n, m),
    };
    let b_mut = MatMut::<f64>::new(b, bm, bn, ldb);

    let _ = level3::trmm_in_place(
        side_internal,
        uplo_internal,
        trans_internal,
        diag_internal,
        alpha,
        a_ref,
        b_mut,
    );
}

/// Single precision TRMM: B = alpha * op(A) * B or B = alpha * B * op(A).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_strmm(
    layout: CblasLayout,
    side: CblasSide,
    uplo: CblasUplo,
    transa: CblasTranspose,
    diag: CblasDiag,
    m: i32,
    n: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    b: *mut f32,
    ldb: i32,
) {
    use crate::level3::{TrmmDiag, TrmmSide, TrmmTrans, TrmmUplo};

    if m <= 0 || n <= 0 {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;

    let (side_internal, uplo_internal, trans_internal, diag_internal) = match layout {
        CblasLayout::ColMajor => (
            match side {
                CblasSide::Left => TrmmSide::Left,
                CblasSide::Right => TrmmSide::Right,
            },
            match uplo {
                CblasUplo::Upper => TrmmUplo::Upper,
                CblasUplo::Lower => TrmmUplo::Lower,
            },
            match transa {
                CblasTranspose::NoTrans => TrmmTrans::NoTrans,
                CblasTranspose::Trans => TrmmTrans::Trans,
                CblasTranspose::ConjTrans => TrmmTrans::ConjTrans,
            },
            match diag {
                CblasDiag::NonUnit => TrmmDiag::NonUnit,
                CblasDiag::Unit => TrmmDiag::Unit,
            },
        ),
        CblasLayout::RowMajor => (
            match side {
                CblasSide::Left => TrmmSide::Right,
                CblasSide::Right => TrmmSide::Left,
            },
            match uplo {
                CblasUplo::Upper => TrmmUplo::Lower,
                CblasUplo::Lower => TrmmUplo::Upper,
            },
            match transa {
                CblasTranspose::NoTrans => TrmmTrans::NoTrans,
                CblasTranspose::Trans => TrmmTrans::Trans,
                CblasTranspose::ConjTrans => TrmmTrans::ConjTrans,
            },
            match diag {
                CblasDiag::NonUnit => TrmmDiag::NonUnit,
                CblasDiag::Unit => TrmmDiag::Unit,
            },
        ),
    };

    let k = match side_internal {
        TrmmSide::Left => m,
        TrmmSide::Right => n,
    };

    use oxiblas_matrix::{MatMut, MatRef};

    let a_ref = MatRef::<f32>::new(a, k, k, lda);
    let (bm, bn) = match layout {
        CblasLayout::ColMajor => (m, n),
        CblasLayout::RowMajor => (n, m),
    };
    let b_mut = MatMut::<f32>::new(b, bm, bn, ldb);

    let _ = level3::trmm_in_place(
        side_internal,
        uplo_internal,
        trans_internal,
        diag_internal,
        alpha,
        a_ref,
        b_mut,
    );
}

// =============================================================================
// Level 3 BLAS - SYRK (Symmetric Rank-K Update)
// =============================================================================

/// Double precision SYRK: C = alpha * A * A^T + beta * C or C = alpha * A^T * A + beta * C.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dsyrk(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    n: i32,
    k: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    beta: f64,
    c: *mut f64,
    ldc: i32,
) {
    if n <= 0 {
        return;
    }

    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldc = ldc as usize;

    let (uplo_internal, trans_internal) = match layout {
        CblasLayout::ColMajor => (
            match uplo {
                CblasUplo::Upper => level3::Uplo::Upper,
                CblasUplo::Lower => level3::Uplo::Lower,
            },
            match trans {
                CblasTranspose::NoTrans => level3::Trans::NoTrans,
                CblasTranspose::Trans | CblasTranspose::ConjTrans => level3::Trans::Trans,
            },
        ),
        CblasLayout::RowMajor => (
            // Swap Upper/Lower and NoTrans/Trans for row-major
            match uplo {
                CblasUplo::Upper => level3::Uplo::Lower,
                CblasUplo::Lower => level3::Uplo::Upper,
            },
            match trans {
                CblasTranspose::NoTrans => level3::Trans::Trans,
                CblasTranspose::Trans | CblasTranspose::ConjTrans => level3::Trans::NoTrans,
            },
        ),
    };

    // A dimensions depend on trans
    let (a_rows, a_cols) = match trans_internal {
        level3::Trans::NoTrans => (n, k),
        level3::Trans::Trans | level3::Trans::ConjTrans => (k, n),
    };

    use oxiblas_matrix::{MatMut, MatRef};

    let a_ref = MatRef::<f64>::new(a, a_rows, a_cols, lda);
    let c_mut = MatMut::<f64>::new(c, n, n, ldc);

    let _ = level3::syrk(uplo_internal, trans_internal, alpha, a_ref, beta, c_mut);
}

/// Single precision SYRK: C = alpha * A * A^T + beta * C or C = alpha * A^T * A + beta * C.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_ssyrk(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    n: i32,
    k: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    beta: f32,
    c: *mut f32,
    ldc: i32,
) {
    if n <= 0 {
        return;
    }

    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldc = ldc as usize;

    let (uplo_internal, trans_internal) = match layout {
        CblasLayout::ColMajor => (
            match uplo {
                CblasUplo::Upper => level3::Uplo::Upper,
                CblasUplo::Lower => level3::Uplo::Lower,
            },
            match trans {
                CblasTranspose::NoTrans => level3::Trans::NoTrans,
                CblasTranspose::Trans | CblasTranspose::ConjTrans => level3::Trans::Trans,
            },
        ),
        CblasLayout::RowMajor => (
            match uplo {
                CblasUplo::Upper => level3::Uplo::Lower,
                CblasUplo::Lower => level3::Uplo::Upper,
            },
            match trans {
                CblasTranspose::NoTrans => level3::Trans::Trans,
                CblasTranspose::Trans | CblasTranspose::ConjTrans => level3::Trans::NoTrans,
            },
        ),
    };

    let (a_rows, a_cols) = match trans_internal {
        level3::Trans::NoTrans => (n, k),
        level3::Trans::Trans | level3::Trans::ConjTrans => (k, n),
    };

    use oxiblas_matrix::{MatMut, MatRef};

    let a_ref = MatRef::<f32>::new(a, a_rows, a_cols, lda);
    let c_mut = MatMut::<f32>::new(c, n, n, ldc);

    let _ = level3::syrk(uplo_internal, trans_internal, alpha, a_ref, beta, c_mut);
}

// =============================================================================
// Level 3 BLAS - SYR2K (Symmetric Rank-2K Update)
// =============================================================================

/// Double precision SYR2K: C = alpha * A * B^T + alpha * B * A^T + beta * C.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dsyr2k(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    n: i32,
    k: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    b: *const f64,
    ldb: i32,
    beta: f64,
    c: *mut f64,
    ldc: i32,
) {
    if n <= 0 {
        return;
    }

    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;

    let (uplo_internal, trans_internal) = match layout {
        CblasLayout::ColMajor => (
            match uplo {
                CblasUplo::Upper => level3::Uplo::Upper,
                CblasUplo::Lower => level3::Uplo::Lower,
            },
            match trans {
                CblasTranspose::NoTrans => level3::Trans::NoTrans,
                CblasTranspose::Trans | CblasTranspose::ConjTrans => level3::Trans::Trans,
            },
        ),
        CblasLayout::RowMajor => (
            match uplo {
                CblasUplo::Upper => level3::Uplo::Lower,
                CblasUplo::Lower => level3::Uplo::Upper,
            },
            match trans {
                CblasTranspose::NoTrans => level3::Trans::Trans,
                CblasTranspose::Trans | CblasTranspose::ConjTrans => level3::Trans::NoTrans,
            },
        ),
    };

    let (ab_rows, ab_cols) = match trans_internal {
        level3::Trans::NoTrans => (n, k),
        level3::Trans::Trans | level3::Trans::ConjTrans => (k, n),
    };

    use oxiblas_matrix::{MatMut, MatRef};

    let a_ref = MatRef::<f64>::new(a, ab_rows, ab_cols, lda);
    let b_ref = MatRef::<f64>::new(b, ab_rows, ab_cols, ldb);
    let c_mut = MatMut::<f64>::new(c, n, n, ldc);

    let _ = level3::syr2k(
        uplo_internal,
        trans_internal,
        alpha,
        a_ref,
        b_ref,
        beta,
        c_mut,
    );
}

/// Single precision SYR2K: C = alpha * A * B^T + alpha * B * A^T + beta * C.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_ssyr2k(
    layout: CblasLayout,
    uplo: CblasUplo,
    trans: CblasTranspose,
    n: i32,
    k: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    b: *const f32,
    ldb: i32,
    beta: f32,
    c: *mut f32,
    ldc: i32,
) {
    if n <= 0 {
        return;
    }

    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;

    let (uplo_internal, trans_internal) = match layout {
        CblasLayout::ColMajor => (
            match uplo {
                CblasUplo::Upper => level3::Uplo::Upper,
                CblasUplo::Lower => level3::Uplo::Lower,
            },
            match trans {
                CblasTranspose::NoTrans => level3::Trans::NoTrans,
                CblasTranspose::Trans | CblasTranspose::ConjTrans => level3::Trans::Trans,
            },
        ),
        CblasLayout::RowMajor => (
            match uplo {
                CblasUplo::Upper => level3::Uplo::Lower,
                CblasUplo::Lower => level3::Uplo::Upper,
            },
            match trans {
                CblasTranspose::NoTrans => level3::Trans::Trans,
                CblasTranspose::Trans | CblasTranspose::ConjTrans => level3::Trans::NoTrans,
            },
        ),
    };

    let (ab_rows, ab_cols) = match trans_internal {
        level3::Trans::NoTrans => (n, k),
        level3::Trans::Trans | level3::Trans::ConjTrans => (k, n),
    };

    use oxiblas_matrix::{MatMut, MatRef};

    let a_ref = MatRef::<f32>::new(a, ab_rows, ab_cols, lda);
    let b_ref = MatRef::<f32>::new(b, ab_rows, ab_cols, ldb);
    let c_mut = MatMut::<f32>::new(c, n, n, ldc);

    let _ = level3::syr2k(
        uplo_internal,
        trans_internal,
        alpha,
        a_ref,
        b_ref,
        beta,
        c_mut,
    );
}

// =============================================================================
// Level 3 BLAS - SYMM (Symmetric Matrix-Matrix Multiply)
// =============================================================================

/// Double precision SYMM: C = alpha * A * B + beta * C or C = alpha * B * A + beta * C.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_dsymm(
    layout: CblasLayout,
    side: CblasSide,
    uplo: CblasUplo,
    m: i32,
    n: i32,
    alpha: f64,
    a: *const f64,
    lda: i32,
    b: *const f64,
    ldb: i32,
    beta: f64,
    c: *mut f64,
    ldc: i32,
) {
    use crate::level3::{Side, Uplo};

    if m <= 0 || n <= 0 {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;

    let (side_internal, uplo_internal, bm, bn) = match layout {
        CblasLayout::ColMajor => (
            match side {
                CblasSide::Left => Side::Left,
                CblasSide::Right => Side::Right,
            },
            match uplo {
                CblasUplo::Upper => Uplo::Upper,
                CblasUplo::Lower => Uplo::Lower,
            },
            m,
            n,
        ),
        CblasLayout::RowMajor => (
            match side {
                CblasSide::Left => Side::Right,
                CblasSide::Right => Side::Left,
            },
            match uplo {
                CblasUplo::Upper => Uplo::Lower,
                CblasUplo::Lower => Uplo::Upper,
            },
            n,
            m,
        ),
    };

    let k = match side_internal {
        Side::Left => bm,
        Side::Right => bn,
    };

    use oxiblas_matrix::{MatMut, MatRef};

    let a_ref = MatRef::<f64>::new(a, k, k, lda);
    let b_ref = MatRef::<f64>::new(b, bm, bn, ldb);
    let c_mut = MatMut::<f64>::new(c, bm, bn, ldc);

    let _ = level3::symm(
        side_internal,
        uplo_internal,
        alpha,
        a_ref,
        b_ref,
        beta,
        c_mut,
    );
}

/// Single precision SYMM: C = alpha * A * B + beta * C or C = alpha * B * A + beta * C.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cblas_ssymm(
    layout: CblasLayout,
    side: CblasSide,
    uplo: CblasUplo,
    m: i32,
    n: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    b: *const f32,
    ldb: i32,
    beta: f32,
    c: *mut f32,
    ldc: i32,
) {
    use crate::level3::{Side, Uplo};

    if m <= 0 || n <= 0 {
        return;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldc = ldc as usize;

    let (side_internal, uplo_internal, bm, bn) = match layout {
        CblasLayout::ColMajor => (
            match side {
                CblasSide::Left => Side::Left,
                CblasSide::Right => Side::Right,
            },
            match uplo {
                CblasUplo::Upper => Uplo::Upper,
                CblasUplo::Lower => Uplo::Lower,
            },
            m,
            n,
        ),
        CblasLayout::RowMajor => (
            match side {
                CblasSide::Left => Side::Right,
                CblasSide::Right => Side::Left,
            },
            match uplo {
                CblasUplo::Upper => Uplo::Lower,
                CblasUplo::Lower => Uplo::Upper,
            },
            n,
            m,
        ),
    };

    let k = match side_internal {
        Side::Left => bm,
        Side::Right => bn,
    };

    use oxiblas_matrix::{MatMut, MatRef};

    let a_ref = MatRef::<f32>::new(a, k, k, lda);
    let b_ref = MatRef::<f32>::new(b, bm, bn, ldb);
    let c_mut = MatMut::<f32>::new(c, bm, bn, ldc);

    let _ = level3::symm(
        side_internal,
        uplo_internal,
        alpha,
        a_ref,
        b_ref,
        beta,
        c_mut,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cblas::{cblas_daxpy, cblas_ddot, cblas_dgemm, cblas_dnrm2, cblas_zdotc_sub};
    use num_complex::Complex64;

    #[test]
    fn test_cblas_ddot() {
        let x = [1.0f64, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0f64, 3.0, 4.0, 5.0, 6.0];

        unsafe {
            let result = cblas_ddot(5, x.as_ptr(), 1, y.as_ptr(), 1);
            // 1*2 + 2*3 + 3*4 + 4*5 + 5*6 = 2 + 6 + 12 + 20 + 30 = 70
            assert!((result - 70.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cblas_dnrm2() {
        let x = [3.0f64, 4.0];

        unsafe {
            let result = cblas_dnrm2(2, x.as_ptr(), 1);
            // sqrt(9 + 16) = sqrt(25) = 5
            assert!((result - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cblas_daxpy() {
        let x = [1.0f64, 2.0, 3.0];
        let mut y = [4.0f64, 5.0, 6.0];

        unsafe {
            cblas_daxpy(3, 2.0, x.as_ptr(), 1, y.as_mut_ptr(), 1);
            // y = 2*x + y = [2+4, 4+5, 6+6] = [6, 9, 12]
            assert!((y[0] - 6.0).abs() < 1e-10);
            assert!((y[1] - 9.0).abs() < 1e-10);
            assert!((y[2] - 12.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cblas_dgemm_colmajor() {
        // A = [[1, 2], [3, 4]] in column-major = [1, 3, 2, 4]
        // B = [[5, 6], [7, 8]] in column-major = [5, 7, 6, 8]
        // C = A * B = [[19, 22], [43, 50]] in column-major = [19, 43, 22, 50]
        let a = [1.0f64, 3.0, 2.0, 4.0];
        let b = [5.0f64, 7.0, 6.0, 8.0];
        let mut c = [0.0f64; 4];

        unsafe {
            cblas_dgemm(
                CblasLayout::ColMajor,
                CblasTranspose::NoTrans,
                CblasTranspose::NoTrans,
                2,
                2,
                2,
                1.0,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                0.0,
                c.as_mut_ptr(),
                2,
            );

            assert!((c[0] - 19.0).abs() < 1e-10);
            assert!((c[1] - 43.0).abs() < 1e-10);
            assert!((c[2] - 22.0).abs() < 1e-10);
            assert!((c[3] - 50.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cblas_dgemm_with_beta() {
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let b = [1.0f64; 4];
        let mut c = [1.0f64; 4];

        unsafe {
            cblas_dgemm(
                CblasLayout::ColMajor,
                CblasTranspose::NoTrans,
                CblasTranspose::NoTrans,
                2,
                2,
                2,
                1.0,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                2.0, // beta = 2
                c.as_mut_ptr(),
                2,
            );

            // C = 1.0 * A * B + 2.0 * C_initial
            // A * B = [[1+3, 1+3], [2+4, 2+4]] = [[4, 4], [6, 6]]
            // C = [[4, 4], [6, 6]] + [[2, 2], [2, 2]] = [[6, 6], [8, 8]]
            assert!((c[0] - 6.0).abs() < 1e-10);
            assert!((c[1] - 8.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cblas_zdotc() {
        let x = [Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
        let y = [Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)];
        let mut result = Complex64::new(0.0, 0.0);

        unsafe {
            cblas_zdotc_sub(2, x.as_ptr(), 1, y.as_ptr(), 1, &mut result);
            // conj(x[0]) * y[0] + conj(x[1]) * y[1]
            // = (1-2i)(5+6i) + (3-4i)(7+8i)
            // = (5+6i-10i+12) + (21+24i-28i+32)
            // = (17-4i) + (53-4i)
            // = 70 - 8i
            assert!((result.re - 70.0).abs() < 1e-10);
            assert!((result.im - (-8.0)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cblas_dtrsm() {
        // Lower triangular matrix A (column-major):
        // [2  0]   stored as [2, 3, 0, 4]
        // [3  4]
        // B = [6, 11] (column vector stored as 2x1)
        // Solve A * X = B => X = A^{-1} * B
        // X[0] = 6/2 = 3
        // X[1] = (11 - 3*3)/4 = 2/4 = 0.5
        let a = [2.0f64, 3.0, 0.0, 4.0];
        let mut b = [6.0f64, 11.0];

        unsafe {
            cblas_dtrsm(
                CblasLayout::ColMajor,
                CblasSide::Left,
                CblasUplo::Lower,
                CblasTranspose::NoTrans,
                CblasDiag::NonUnit,
                2,
                1,
                1.0,
                a.as_ptr(),
                2,
                b.as_mut_ptr(),
                2,
            );

            assert!((b[0] - 3.0).abs() < 1e-10);
            assert!((b[1] - 0.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cblas_dsyrk() {
        // A = [[1, 2], [3, 4]] in column-major = [1, 3, 2, 4]
        // C = A * A^T (NoTrans case)
        // C[0,0] = 1*1 + 2*2 = 5
        // C[1,0] = C[0,1] = 1*3 + 2*4 = 11
        // C[1,1] = 3*3 + 4*4 = 25
        let a = [1.0f64, 3.0, 2.0, 4.0];
        let mut c = [0.0f64; 4];

        unsafe {
            cblas_dsyrk(
                CblasLayout::ColMajor,
                CblasUplo::Lower,
                CblasTranspose::NoTrans,
                2,
                2,
                1.0,
                a.as_ptr(),
                2,
                0.0,
                c.as_mut_ptr(),
                2,
            );

            // Lower triangle only
            assert!((c[0] - 5.0).abs() < 1e-10); // C[0,0]
            assert!((c[1] - 11.0).abs() < 1e-10); // C[1,0]
            assert!((c[3] - 25.0).abs() < 1e-10); // C[1,1]
        }
    }

    #[test]
    fn test_cblas_dsymm() {
        // Symmetric matrix A (stored in lower):
        // [1  2]
        // [2  3]
        // B = [[1, 0], [0, 1]] (identity)
        // C = A * B = A
        let a = [1.0f64, 2.0, 0.0, 3.0]; // Lower stored
        let b = [1.0f64, 0.0, 0.0, 1.0];
        let mut c = [0.0f64; 4];

        unsafe {
            cblas_dsymm(
                CblasLayout::ColMajor,
                CblasSide::Left,
                CblasUplo::Lower,
                2,
                2,
                1.0,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                0.0,
                c.as_mut_ptr(),
                2,
            );

            // Result should be symmetric matrix A
            assert!((c[0] - 1.0).abs() < 1e-10);
            assert!((c[1] - 2.0).abs() < 1e-10);
            assert!((c[2] - 2.0).abs() < 1e-10);
            assert!((c[3] - 3.0).abs() < 1e-10);
        }
    }
}
