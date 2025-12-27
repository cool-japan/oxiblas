//! LAPACK FFI - Linear system solve routines.

use crate::types::*;
use oxiblas_matrix::Mat;
use std::ffi::c_int;
use std::slice;

// =============================================================================
// SGESVX - Expert linear system solve (single precision)
// =============================================================================

/// Expert driver for solving general linear systems with equilibration
/// and condition estimation.
///
/// Solves A*X = B with optional equilibration, and provides:
/// - Reciprocal condition number estimate
/// - Forward and backward error bounds
/// - Row and/or column scaling factors
///
/// # Safety
/// - `a` must point to a valid n x n matrix
/// - `b` must point to a valid n x nrhs matrix
/// - `x` must point to storage for n x nrhs matrix (solution output)
/// - `rcond` must point to valid f32 (output)
/// - `ferr` must point to nrhs f32 values (forward error output)
/// - `berr` must point to nrhs f32 values (backward error output)
/// - `r` may be null or point to n f32 values (row scale output)
/// - `c` may be null or point to n f32 values (column scale output)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sgesvx(
    layout: OblasLayout,
    equil: u8, // 'N' = none, 'R' = row, 'C' = column, 'B' = both
    n: c_int,
    nrhs: c_int,
    a: *const f32,
    lda: c_int,
    b: *const f32,
    ldb: c_int,
    x: *mut f32,
    ldx: c_int,
    rcond: *mut f32,
    ferr: *mut f32,
    berr: *mut f32,
    r: *mut f32,
    c: *mut f32,
) -> c_int {
    use oxiblas_lapack::solve::expert::{Equilibrate, solve_expert};

    if n <= 0 || nrhs <= 0 || a.is_null() || b.is_null() || x.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }
    if rcond.is_null() || ferr.is_null() || berr.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let nrhs = nrhs as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldx = ldx as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Determine equilibration type
    let eq_type = match equil {
        b'N' | b'n' => Equilibrate::None,
        b'R' | b'r' => Equilibrate::Row,
        b'C' | b'c' => Equilibrate::Column,
        b'B' | b'b' => Equilibrate::Both,
        _ => return OblasReturn::InvalidArg as c_int,
    };

    // Convert A to Mat
    let mut mat_a: Mat<f32> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    // Convert B to Mat
    let mut mat_b: Mat<f32> = Mat::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            let idx = if row_major { i * ldb + j } else { j * ldb + i };
            mat_b[(i, j)] = *b.add(idx);
        }
    }

    // Call expert solve
    match solve_expert(mat_a.as_ref(), mat_b.as_ref(), eq_type) {
        Ok(result) => {
            // Copy solution to x
            for i in 0..n {
                for j in 0..nrhs {
                    let idx = if row_major { i * ldx + j } else { j * ldx + i };
                    *x.add(idx) = result.solution[(i, j)];
                }
            }

            // Copy rcond
            *rcond = result.rcond;

            // Copy error bounds
            let ferr_slice = slice::from_raw_parts_mut(ferr, nrhs);
            let berr_slice = slice::from_raw_parts_mut(berr, nrhs);
            for j in 0..nrhs {
                ferr_slice[j] = result.forward_error[j];
                berr_slice[j] = result.backward_error[j];
            }

            // Copy scaling factors if requested
            if !r.is_null() {
                if let Some(ref row_scale) = result.row_scale {
                    let r_slice = slice::from_raw_parts_mut(r, n);
                    for i in 0..n {
                        r_slice[i] = row_scale[i];
                    }
                }
            }

            if !c.is_null() {
                if let Some(ref col_scale) = result.col_scale {
                    let c_slice = slice::from_raw_parts_mut(c, n);
                    for i in 0..n {
                        c_slice[i] = col_scale[i];
                    }
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::Singular as c_int,
    }
}

// =============================================================================
// DGESVX - Expert linear system solve (double precision)
// =============================================================================

/// Expert driver for solving general linear systems with equilibration
/// and condition estimation.
///
/// Solves A*X = B with optional equilibration, and provides:
/// - Reciprocal condition number estimate
/// - Forward and backward error bounds
/// - Row and/or column scaling factors
///
/// # Safety
/// - `a` must point to a valid n x n matrix
/// - `b` must point to a valid n x nrhs matrix
/// - `x` must point to storage for n x nrhs matrix (solution output)
/// - `rcond` must point to valid f64 (output)
/// - `ferr` must point to nrhs f64 values (forward error output)
/// - `berr` must point to nrhs f64 values (backward error output)
/// - `r` may be null or point to n f64 values (row scale output)
/// - `c` may be null or point to n f64 values (column scale output)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dgesvx(
    layout: OblasLayout,
    equil: u8, // 'N' = none, 'R' = row, 'C' = column, 'B' = both
    n: c_int,
    nrhs: c_int,
    a: *const f64,
    lda: c_int,
    b: *const f64,
    ldb: c_int,
    x: *mut f64,
    ldx: c_int,
    rcond: *mut f64,
    ferr: *mut f64,
    berr: *mut f64,
    r: *mut f64,
    c: *mut f64,
) -> c_int {
    use oxiblas_lapack::solve::expert::{Equilibrate, solve_expert};

    if n <= 0 || nrhs <= 0 || a.is_null() || b.is_null() || x.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }
    if rcond.is_null() || ferr.is_null() || berr.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let nrhs = nrhs as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldx = ldx as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Determine equilibration type
    let eq_type = match equil {
        b'N' | b'n' => Equilibrate::None,
        b'R' | b'r' => Equilibrate::Row,
        b'C' | b'c' => Equilibrate::Column,
        b'B' | b'b' => Equilibrate::Both,
        _ => return OblasReturn::InvalidArg as c_int,
    };

    // Convert A to Mat
    let mut mat_a: Mat<f64> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    // Convert B to Mat
    let mut mat_b: Mat<f64> = Mat::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            let idx = if row_major { i * ldb + j } else { j * ldb + i };
            mat_b[(i, j)] = *b.add(idx);
        }
    }

    // Call expert solve
    match solve_expert(mat_a.as_ref(), mat_b.as_ref(), eq_type) {
        Ok(result) => {
            // Copy solution to x
            for i in 0..n {
                for j in 0..nrhs {
                    let idx = if row_major { i * ldx + j } else { j * ldx + i };
                    *x.add(idx) = result.solution[(i, j)];
                }
            }

            // Copy rcond
            *rcond = result.rcond;

            // Copy error bounds
            let ferr_slice = slice::from_raw_parts_mut(ferr, nrhs);
            let berr_slice = slice::from_raw_parts_mut(berr, nrhs);
            for j in 0..nrhs {
                ferr_slice[j] = result.forward_error[j];
                berr_slice[j] = result.backward_error[j];
            }

            // Copy scaling factors if requested
            if !r.is_null() {
                if let Some(ref row_scale) = result.row_scale {
                    let r_slice = slice::from_raw_parts_mut(r, n);
                    for i in 0..n {
                        r_slice[i] = row_scale[i];
                    }
                }
            }

            if !c.is_null() {
                if let Some(ref col_scale) = result.col_scale {
                    let c_slice = slice::from_raw_parts_mut(c, n);
                    for i in 0..n {
                        c_slice[i] = col_scale[i];
                    }
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::Singular as c_int,
    }
}

// =============================================================================
// SPOSVX - Expert Cholesky solve (single precision)
// =============================================================================

/// Expert driver for solving symmetric positive definite linear systems
/// with equilibration and condition estimation.
///
/// Solves A*X = B (A symmetric positive definite) with optional equilibration,
/// and provides:
/// - Reciprocal condition number estimate
/// - Forward and backward error bounds
/// - Scaling factors
///
/// # Safety
/// - `a` must point to a valid n x n symmetric positive definite matrix
/// - `b` must point to a valid n x nrhs matrix
/// - `x` must point to storage for n x nrhs matrix (solution output)
/// - `rcond` must point to valid f32 (output)
/// - `ferr` must point to nrhs f32 values (forward error output)
/// - `berr` must point to nrhs f32 values (backward error output)
/// - `s` may be null or point to n f32 values (scaling output)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sposvx(
    layout: OblasLayout,
    equil: u8, // 'N' = none, 'Y' = equilibrate
    n: c_int,
    nrhs: c_int,
    a: *const f32,
    lda: c_int,
    b: *const f32,
    ldb: c_int,
    x: *mut f32,
    ldx: c_int,
    rcond: *mut f32,
    ferr: *mut f32,
    berr: *mut f32,
    s: *mut f32,
) -> c_int {
    use oxiblas_lapack::solve::expert_cholesky::solve_cholesky_expert;

    if n <= 0 || nrhs <= 0 || a.is_null() || b.is_null() || x.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }
    if rcond.is_null() || ferr.is_null() || berr.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let nrhs = nrhs as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldx = ldx as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Determine if equilibration is requested
    let do_equil = equil == b'Y' || equil == b'y';

    // Convert A to Mat
    let mut mat_a: Mat<f32> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    // Convert B to Mat
    let mut mat_b: Mat<f32> = Mat::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            let idx = if row_major { i * ldb + j } else { j * ldb + i };
            mat_b[(i, j)] = *b.add(idx);
        }
    }

    // Call expert solve
    match solve_cholesky_expert(mat_a.as_ref(), mat_b.as_ref(), do_equil) {
        Ok(result) => {
            // Copy solution to x
            for i in 0..n {
                for j in 0..nrhs {
                    let idx = if row_major { i * ldx + j } else { j * ldx + i };
                    *x.add(idx) = result.solution[(i, j)];
                }
            }

            // Copy rcond
            *rcond = result.rcond;

            // Copy error bounds
            let ferr_slice = slice::from_raw_parts_mut(ferr, nrhs);
            let berr_slice = slice::from_raw_parts_mut(berr, nrhs);
            for j in 0..nrhs {
                ferr_slice[j] = result.forward_error[j];
                berr_slice[j] = result.backward_error[j];
            }

            // Copy scaling factors if requested
            if !s.is_null() {
                if let Some(ref scale) = result.scale {
                    let s_slice = slice::from_raw_parts_mut(s, n);
                    for i in 0..n {
                        s_slice[i] = scale[i];
                    }
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::Singular as c_int,
    }
}

// =============================================================================
// DPOSVX - Expert Cholesky solve (double precision)
// =============================================================================

/// Expert driver for solving symmetric positive definite linear systems
/// with equilibration and condition estimation.
///
/// Solves A*X = B (A symmetric positive definite) with optional equilibration,
/// and provides:
/// - Reciprocal condition number estimate
/// - Forward and backward error bounds
/// - Scaling factors
///
/// # Safety
/// - `a` must point to a valid n x n symmetric positive definite matrix
/// - `b` must point to a valid n x nrhs matrix
/// - `x` must point to storage for n x nrhs matrix (solution output)
/// - `rcond` must point to valid f64 (output)
/// - `ferr` must point to nrhs f64 values (forward error output)
/// - `berr` must point to nrhs f64 values (backward error output)
/// - `s` may be null or point to n f64 values (scaling output)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dposvx(
    layout: OblasLayout,
    equil: u8, // 'N' = none, 'Y' = equilibrate
    n: c_int,
    nrhs: c_int,
    a: *const f64,
    lda: c_int,
    b: *const f64,
    ldb: c_int,
    x: *mut f64,
    ldx: c_int,
    rcond: *mut f64,
    ferr: *mut f64,
    berr: *mut f64,
    s: *mut f64,
) -> c_int {
    use oxiblas_lapack::solve::expert_cholesky::solve_cholesky_expert;

    if n <= 0 || nrhs <= 0 || a.is_null() || b.is_null() || x.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }
    if rcond.is_null() || ferr.is_null() || berr.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let nrhs = nrhs as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldx = ldx as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Determine if equilibration is requested
    let do_equil = equil == b'Y' || equil == b'y';

    // Convert A to Mat
    let mut mat_a: Mat<f64> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    // Convert B to Mat
    let mut mat_b: Mat<f64> = Mat::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            let idx = if row_major { i * ldb + j } else { j * ldb + i };
            mat_b[(i, j)] = *b.add(idx);
        }
    }

    // Call expert solve
    match solve_cholesky_expert(mat_a.as_ref(), mat_b.as_ref(), do_equil) {
        Ok(result) => {
            // Copy solution to x
            for i in 0..n {
                for j in 0..nrhs {
                    let idx = if row_major { i * ldx + j } else { j * ldx + i };
                    *x.add(idx) = result.solution[(i, j)];
                }
            }

            // Copy rcond
            *rcond = result.rcond;

            // Copy error bounds
            let ferr_slice = slice::from_raw_parts_mut(ferr, nrhs);
            let berr_slice = slice::from_raw_parts_mut(berr, nrhs);
            for j in 0..nrhs {
                ferr_slice[j] = result.forward_error[j];
                berr_slice[j] = result.backward_error[j];
            }

            // Copy scaling factors if requested
            if !s.is_null() {
                if let Some(ref scale) = result.scale {
                    let s_slice = slice::from_raw_parts_mut(s, n);
                    for i in 0..n {
                        s_slice[i] = scale[i];
                    }
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::Singular as c_int,
    }
}

// =============================================================================
// SSYSVX - Expert symmetric solve (single precision)
// =============================================================================

/// Expert driver for solving symmetric linear systems (possibly indefinite)
/// with equilibration and condition estimation.
///
/// Solves A*X = B (A symmetric) using LDL^T factorization, with:
/// - Reciprocal condition number estimate
/// - Forward and backward error bounds
/// - Matrix inertia (positive/negative/zero eigenvalue counts)
///
/// # Safety
/// - `a` must point to a valid n x n symmetric matrix
/// - `b` must point to a valid n x nrhs matrix
/// - `x` must point to storage for n x nrhs matrix (solution output)
/// - `rcond` must point to valid f32 (output)
/// - `ferr` must point to nrhs f32 values (forward error output)
/// - `berr` must point to nrhs f32 values (backward error output)
/// - `inertia_pos`, `inertia_neg` must point to valid i32 (output)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ssysvx(
    layout: OblasLayout,
    equil: u8, // 'N' = none, 'Y' = equilibrate
    n: c_int,
    nrhs: c_int,
    a: *const f32,
    lda: c_int,
    b: *const f32,
    ldb: c_int,
    x: *mut f32,
    ldx: c_int,
    rcond: *mut f32,
    ferr: *mut f32,
    berr: *mut f32,
    inertia_pos: *mut c_int,
    inertia_neg: *mut c_int,
) -> c_int {
    use oxiblas_lapack::solve::expert_symmetric::solve_symmetric_expert;

    if n <= 0 || nrhs <= 0 || a.is_null() || b.is_null() || x.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }
    if rcond.is_null() || ferr.is_null() || berr.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let nrhs = nrhs as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldx = ldx as usize;
    let row_major = layout == OblasLayout::RowMajor;

    let do_equil = equil == b'Y' || equil == b'y';

    // Convert A to Mat
    let mut mat_a: Mat<f32> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    // Convert B to Mat
    let mut mat_b: Mat<f32> = Mat::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            let idx = if row_major { i * ldb + j } else { j * ldb + i };
            mat_b[(i, j)] = *b.add(idx);
        }
    }

    match solve_symmetric_expert(mat_a.as_ref(), mat_b.as_ref(), do_equil) {
        Ok(result) => {
            // Copy solution to x
            for i in 0..n {
                for j in 0..nrhs {
                    let idx = if row_major { i * ldx + j } else { j * ldx + i };
                    *x.add(idx) = result.solution[(i, j)];
                }
            }

            // Copy rcond
            *rcond = result.rcond;

            // Copy error bounds
            let ferr_slice = slice::from_raw_parts_mut(ferr, nrhs);
            let berr_slice = slice::from_raw_parts_mut(berr, nrhs);
            for j in 0..nrhs {
                ferr_slice[j] = result.forward_error[j];
                berr_slice[j] = result.backward_error[j];
            }

            // Copy inertia if requested
            if !inertia_pos.is_null() {
                *inertia_pos = result.inertia.0 as c_int;
            }
            if !inertia_neg.is_null() {
                *inertia_neg = result.inertia.1 as c_int;
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::Singular as c_int,
    }
}

// =============================================================================
// DSYSVX - Expert symmetric solve (double precision)
// =============================================================================

/// Expert driver for solving symmetric linear systems (possibly indefinite)
/// with equilibration and condition estimation.
///
/// Solves A*X = B (A symmetric) using LDL^T factorization, with:
/// - Reciprocal condition number estimate
/// - Forward and backward error bounds
/// - Matrix inertia (positive/negative/zero eigenvalue counts)
///
/// # Safety
/// - `a` must point to a valid n x n symmetric matrix
/// - `b` must point to a valid n x nrhs matrix
/// - `x` must point to storage for n x nrhs matrix (solution output)
/// - `rcond` must point to valid f64 (output)
/// - `ferr` must point to nrhs f64 values (forward error output)
/// - `berr` must point to nrhs f64 values (backward error output)
/// - `inertia_pos`, `inertia_neg` must point to valid i32 (output)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dsysvx(
    layout: OblasLayout,
    equil: u8, // 'N' = none, 'Y' = equilibrate
    n: c_int,
    nrhs: c_int,
    a: *const f64,
    lda: c_int,
    b: *const f64,
    ldb: c_int,
    x: *mut f64,
    ldx: c_int,
    rcond: *mut f64,
    ferr: *mut f64,
    berr: *mut f64,
    inertia_pos: *mut c_int,
    inertia_neg: *mut c_int,
) -> c_int {
    use oxiblas_lapack::solve::expert_symmetric::solve_symmetric_expert;

    if n <= 0 || nrhs <= 0 || a.is_null() || b.is_null() || x.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }
    if rcond.is_null() || ferr.is_null() || berr.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let nrhs = nrhs as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldx = ldx as usize;
    let row_major = layout == OblasLayout::RowMajor;

    let do_equil = equil == b'Y' || equil == b'y';

    // Convert A to Mat
    let mut mat_a: Mat<f64> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    // Convert B to Mat
    let mut mat_b: Mat<f64> = Mat::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            let idx = if row_major { i * ldb + j } else { j * ldb + i };
            mat_b[(i, j)] = *b.add(idx);
        }
    }

    match solve_symmetric_expert(mat_a.as_ref(), mat_b.as_ref(), do_equil) {
        Ok(result) => {
            // Copy solution to x
            for i in 0..n {
                for j in 0..nrhs {
                    let idx = if row_major { i * ldx + j } else { j * ldx + i };
                    *x.add(idx) = result.solution[(i, j)];
                }
            }

            // Copy rcond
            *rcond = result.rcond;

            // Copy error bounds
            let ferr_slice = slice::from_raw_parts_mut(ferr, nrhs);
            let berr_slice = slice::from_raw_parts_mut(berr, nrhs);
            for j in 0..nrhs {
                ferr_slice[j] = result.forward_error[j];
                berr_slice[j] = result.backward_error[j];
            }

            // Copy inertia if requested
            if !inertia_pos.is_null() {
                *inertia_pos = result.inertia.0 as c_int;
            }
            if !inertia_neg.is_null() {
                *inertia_neg = result.inertia.1 as c_int;
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::Singular as c_int,
    }
}

// =============================================================================
