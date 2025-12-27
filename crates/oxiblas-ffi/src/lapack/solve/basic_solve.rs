//! LAPACK FFI - Linear system solve routines.

use crate::types::*;
use oxiblas_lapack::{lu, qr};
use oxiblas_matrix::Mat;
use std::ffi::c_int;
use std::slice;

// =============================================================================
// SGESV - Solve linear system (single precision)
// =============================================================================

/// Solves the linear system A*X = B using LU factorization with partial pivoting.
///
/// # Safety
/// - `a` must point to a valid n x n matrix (overwritten with LU factors)
/// - `ipiv` must point to an array of n integers
/// - `b` must point to a valid n x nrhs matrix (overwritten with solution)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sgesv(
    layout: OblasLayout,
    n: c_int,
    nrhs: c_int,
    a: *mut f32,
    lda: c_int,
    ipiv: *mut c_int,
    b: *mut f32,
    ldb: c_int,
) -> c_int {
    use oxiblas_lapack::solve::solve_multiple;

    if n <= 0 || nrhs <= 0 || a.is_null() || ipiv.is_null() || b.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let nrhs = nrhs as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let row_major = layout == OblasLayout::RowMajor;

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

    // Solve system
    match solve_multiple(mat_a.as_ref(), mat_b.as_ref()) {
        Ok(x) => {
            // Copy solution back to b
            for i in 0..n {
                for j in 0..nrhs {
                    let idx = if row_major { i * ldb + j } else { j * ldb + i };
                    *b.add(idx) = x[(i, j)];
                }
            }

            // Also compute LU and copy to a (for compatibility)
            if let Ok(lu_result) = lu::Lu::compute(mat_a.as_ref()) {
                let l = lu_result.l_factor();
                let u = lu_result.u_factor();
                let perm = lu_result.pivot();

                for i in 0..n {
                    for j in 0..n {
                        let idx = if row_major { i * lda + j } else { j * lda + i };
                        if i > j {
                            *a.add(idx) = l[(i, j)];
                        } else {
                            *a.add(idx) = u[(i, j)];
                        }
                    }
                }

                // Copy pivot indices
                let ipiv_slice = slice::from_raw_parts_mut(ipiv, n);
                for i in 0..n {
                    ipiv_slice[i] = (perm[i] + 1) as c_int;
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::Singular as c_int,
    }
}

// =============================================================================
// DGESV - Solve linear system (double precision)
// =============================================================================

/// Solves the linear system A*X = B using LU factorization with partial pivoting.
///
/// # Safety
/// - `a` must point to a valid n x n matrix (overwritten with LU factors)
/// - `ipiv` must point to an array of n integers
/// - `b` must point to a valid n x nrhs matrix (overwritten with solution)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dgesv(
    layout: OblasLayout,
    n: c_int,
    nrhs: c_int,
    a: *mut f64,
    lda: c_int,
    ipiv: *mut c_int,
    b: *mut f64,
    ldb: c_int,
) -> c_int {
    use oxiblas_lapack::solve::solve_multiple;

    if n <= 0 || nrhs <= 0 || a.is_null() || ipiv.is_null() || b.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let nrhs = nrhs as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let row_major = layout == OblasLayout::RowMajor;

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

    // Solve system
    match solve_multiple(mat_a.as_ref(), mat_b.as_ref()) {
        Ok(x) => {
            // Copy solution back to b
            for i in 0..n {
                for j in 0..nrhs {
                    let idx = if row_major { i * ldb + j } else { j * ldb + i };
                    *b.add(idx) = x[(i, j)];
                }
            }

            // Also compute LU and copy to a (for compatibility)
            if let Ok(lu_result) = lu::Lu::compute(mat_a.as_ref()) {
                let l = lu_result.l_factor();
                let u = lu_result.u_factor();
                let perm = lu_result.pivot();

                for i in 0..n {
                    for j in 0..n {
                        let idx = if row_major { i * lda + j } else { j * lda + i };
                        if i > j {
                            *a.add(idx) = l[(i, j)];
                        } else {
                            *a.add(idx) = u[(i, j)];
                        }
                    }
                }

                // Copy pivot indices
                let ipiv_slice = slice::from_raw_parts_mut(ipiv, n);
                for i in 0..n {
                    ipiv_slice[i] = (perm[i] + 1) as c_int;
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::Singular as c_int,
    }
}

// =============================================================================
// ZGESV - Complex linear solve (double precision)
// =============================================================================

/// Solves a complex system A*X = B using LU factorization.
///
/// # Safety
/// - `a` must point to a valid n x n complex matrix (overwritten with LU factors)
/// - `ipiv` must point to an array of n integers
/// - `b` must point to a valid n x nrhs complex matrix (overwritten with solution)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zgesv(
    layout: OblasLayout,
    n: c_int,
    nrhs: c_int,
    a: *mut OblasComplex64,
    lda: c_int,
    ipiv: *mut c_int,
    b: *mut OblasComplex64,
    ldb: c_int,
) -> c_int {
    use num_complex::Complex64;

    if n <= 0 || nrhs <= 0 || a.is_null() || ipiv.is_null() || b.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let nrhs = nrhs as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert A to Mat<Complex64>
    let mut mat_a: Mat<Complex64> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            let c = *a.add(idx);
            mat_a[(i, j)] = Complex64::new(c.re, c.im);
        }
    }

    // Convert B to Mat<Complex64>
    let mut mat_b: Mat<Complex64> = Mat::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            let idx = if row_major { i * ldb + j } else { j * ldb + i };
            let c = *b.add(idx);
            mat_b[(i, j)] = Complex64::new(c.re, c.im);
        }
    }

    // Compute LU factorization
    match lu::Lu::compute(mat_a.as_ref()) {
        Ok(lu_result) => {
            // Solve the system using LU
            match lu_result.solve(mat_b.as_ref()) {
                Ok(x) => {
                    // Copy solution back to b
                    for i in 0..n {
                        for j in 0..nrhs {
                            let idx = if row_major { i * ldb + j } else { j * ldb + i };
                            let v = x[(i, j)];
                            (*b.add(idx)).re = v.re;
                            (*b.add(idx)).im = v.im;
                        }
                    }

                    // Copy LU factors to a
                    let l = lu_result.l_factor();
                    let u = lu_result.u_factor();
                    let perm = lu_result.pivot();

                    for i in 0..n {
                        for j in 0..n {
                            let idx = if row_major { i * lda + j } else { j * lda + i };
                            let v = if i > j { l[(i, j)] } else { u[(i, j)] };
                            (*a.add(idx)).re = v.re;
                            (*a.add(idx)).im = v.im;
                        }
                    }

                    // Copy pivot indices
                    let ipiv_slice = slice::from_raw_parts_mut(ipiv, n);
                    for i in 0..n {
                        ipiv_slice[i] = (perm[i] + 1) as c_int;
                    }

                    OblasReturn::Success as c_int
                }
                Err(_) => OblasReturn::Singular as c_int,
            }
        }
        Err(_) => OblasReturn::Singular as c_int,
    }
}

// =============================================================================
// CGESV - Complex linear solve (single precision)
// =============================================================================

/// Solves a complex system A*X = B using LU factorization.
///
/// # Safety
/// - `a` must point to a valid n x n complex matrix (overwritten with LU factors)
/// - `ipiv` must point to an array of n integers
/// - `b` must point to a valid n x nrhs complex matrix (overwritten with solution)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_cgesv(
    layout: OblasLayout,
    n: c_int,
    nrhs: c_int,
    a: *mut OblasComplex32,
    lda: c_int,
    ipiv: *mut c_int,
    b: *mut OblasComplex32,
    ldb: c_int,
) -> c_int {
    use num_complex::Complex32;

    if n <= 0 || nrhs <= 0 || a.is_null() || ipiv.is_null() || b.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let nrhs = nrhs as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert A to Mat<Complex32>
    let mut mat_a: Mat<Complex32> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            let c = *a.add(idx);
            mat_a[(i, j)] = Complex32::new(c.re, c.im);
        }
    }

    // Convert B to Mat<Complex32>
    let mut mat_b: Mat<Complex32> = Mat::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            let idx = if row_major { i * ldb + j } else { j * ldb + i };
            let c = *b.add(idx);
            mat_b[(i, j)] = Complex32::new(c.re, c.im);
        }
    }

    // Compute LU factorization
    match lu::Lu::compute(mat_a.as_ref()) {
        Ok(lu_result) => {
            // Solve the system using LU
            match lu_result.solve(mat_b.as_ref()) {
                Ok(x) => {
                    // Copy solution back to b
                    for i in 0..n {
                        for j in 0..nrhs {
                            let idx = if row_major { i * ldb + j } else { j * ldb + i };
                            let v = x[(i, j)];
                            (*b.add(idx)).re = v.re;
                            (*b.add(idx)).im = v.im;
                        }
                    }

                    // Copy LU factors to a
                    let l = lu_result.l_factor();
                    let u = lu_result.u_factor();
                    let perm = lu_result.pivot();

                    for i in 0..n {
                        for j in 0..n {
                            let idx = if row_major { i * lda + j } else { j * lda + i };
                            let v = if i > j { l[(i, j)] } else { u[(i, j)] };
                            (*a.add(idx)).re = v.re;
                            (*a.add(idx)).im = v.im;
                        }
                    }

                    // Copy pivot indices
                    let ipiv_slice = slice::from_raw_parts_mut(ipiv, n);
                    for i in 0..n {
                        ipiv_slice[i] = (perm[i] + 1) as c_int;
                    }

                    OblasReturn::Success as c_int
                }
                Err(_) => OblasReturn::Singular as c_int,
            }
        }
        Err(_) => OblasReturn::Singular as c_int,
    }
}

// =============================================================================
// SGETRS - Solve from LU factors (single precision)
// =============================================================================

/// Solves a system of linear equations A*X = B or A^T*X = B using LU factors.
///
/// The LU factorization should have been computed by oblas_sgetrf.
///
/// # Safety
/// - `a` must point to a valid n x n matrix containing LU factors
/// - `ipiv` must point to an array of n integers (pivot indices from sgetrf)
/// - `b` must point to a valid n x nrhs matrix (overwritten with solution)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sgetrs(
    layout: OblasLayout,
    trans: OblasTranspose,
    n: c_int,
    nrhs: c_int,
    a: *const f32,
    lda: c_int,
    ipiv: *const c_int,
    b: *mut f32,
    ldb: c_int,
) -> c_int {
    if n <= 0 || nrhs <= 0 || a.is_null() || ipiv.is_null() || b.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let nrhs = nrhs as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Read LU factors
    let mut lu = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            lu[i * n + j] = *a.add(idx);
        }
    }

    // Read pivot indices (convert from 1-based to 0-based)
    let ipiv_slice = slice::from_raw_parts(ipiv, n);

    // Read B
    let mut x = vec![0.0f32; n * nrhs];
    for i in 0..n {
        for j in 0..nrhs {
            let idx = if row_major { i * ldb + j } else { j * ldb + i };
            x[i * nrhs + j] = *b.add(idx);
        }
    }

    match trans {
        OblasTranspose::NoTrans => {
            // Solve A*X = B using PA = LU
            // 1. Apply permutation to B: P*B
            for k in 0..n {
                let pk = (ipiv_slice[k] - 1) as usize; // Convert to 0-based
                if k != pk {
                    for j in 0..nrhs {
                        let tmp = x[k * nrhs + j];
                        x[k * nrhs + j] = x[pk * nrhs + j];
                        x[pk * nrhs + j] = tmp;
                    }
                }
            }

            // 2. Forward substitution: L*Y = P*B (L has unit diagonal)
            for k in 0..n {
                for i in (k + 1)..n {
                    let mult = lu[i * n + k];
                    for j in 0..nrhs {
                        x[i * nrhs + j] -= mult * x[k * nrhs + j];
                    }
                }
            }

            // 3. Back substitution: U*X = Y
            for k in (0..n).rev() {
                let diag = lu[k * n + k];
                if diag.abs() < f32::EPSILON {
                    return OblasReturn::Singular as c_int;
                }
                for j in 0..nrhs {
                    x[k * nrhs + j] /= diag;
                }
                for i in 0..k {
                    let mult = lu[i * n + k];
                    for j in 0..nrhs {
                        x[i * nrhs + j] -= mult * x[k * nrhs + j];
                    }
                }
            }
        }
        OblasTranspose::Trans | OblasTranspose::ConjTrans => {
            // Solve A^T*X = B using PA = LU => A^T = U^T * L^T * P^T
            // 1. Forward substitution: U^T*Y = B
            for k in 0..n {
                let diag = lu[k * n + k];
                if diag.abs() < f32::EPSILON {
                    return OblasReturn::Singular as c_int;
                }
                for j in 0..nrhs {
                    x[k * nrhs + j] /= diag;
                }
                for i in (k + 1)..n {
                    let mult = lu[k * n + i];
                    for j in 0..nrhs {
                        x[i * nrhs + j] -= mult * x[k * nrhs + j];
                    }
                }
            }

            // 2. Back substitution: L^T*Z = Y (L has unit diagonal)
            for k in (0..n).rev() {
                for i in 0..k {
                    let mult = lu[k * n + i];
                    for j in 0..nrhs {
                        x[i * nrhs + j] -= mult * x[k * nrhs + j];
                    }
                }
            }

            // 3. Apply permutation: X = P^T*Z
            for k in (0..n).rev() {
                let pk = (ipiv_slice[k] - 1) as usize;
                if k != pk {
                    for j in 0..nrhs {
                        let tmp = x[k * nrhs + j];
                        x[k * nrhs + j] = x[pk * nrhs + j];
                        x[pk * nrhs + j] = tmp;
                    }
                }
            }
        }
    }

    // Copy solution back to b
    for i in 0..n {
        for j in 0..nrhs {
            let idx = if row_major { i * ldb + j } else { j * ldb + i };
            *b.add(idx) = x[i * nrhs + j];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DGETRS - Solve from LU factors (double precision)
// =============================================================================

/// Solves a system of linear equations A*X = B or A^T*X = B using LU factors.
///
/// The LU factorization should have been computed by oblas_dgetrf.
///
/// # Safety
/// - `a` must point to a valid n x n matrix containing LU factors
/// - `ipiv` must point to an array of n integers (pivot indices from dgetrf)
/// - `b` must point to a valid n x nrhs matrix (overwritten with solution)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dgetrs(
    layout: OblasLayout,
    trans: OblasTranspose,
    n: c_int,
    nrhs: c_int,
    a: *const f64,
    lda: c_int,
    ipiv: *const c_int,
    b: *mut f64,
    ldb: c_int,
) -> c_int {
    if n <= 0 || nrhs <= 0 || a.is_null() || ipiv.is_null() || b.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let nrhs = nrhs as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Read LU factors
    let mut lu = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            lu[i * n + j] = *a.add(idx);
        }
    }

    // Read pivot indices (convert from 1-based to 0-based)
    let ipiv_slice = slice::from_raw_parts(ipiv, n);

    // Read B
    let mut x = vec![0.0f64; n * nrhs];
    for i in 0..n {
        for j in 0..nrhs {
            let idx = if row_major { i * ldb + j } else { j * ldb + i };
            x[i * nrhs + j] = *b.add(idx);
        }
    }

    match trans {
        OblasTranspose::NoTrans => {
            // Solve A*X = B using PA = LU
            // 1. Apply permutation to B: P*B
            for k in 0..n {
                let pk = (ipiv_slice[k] - 1) as usize;
                if k != pk {
                    for j in 0..nrhs {
                        let tmp = x[k * nrhs + j];
                        x[k * nrhs + j] = x[pk * nrhs + j];
                        x[pk * nrhs + j] = tmp;
                    }
                }
            }

            // 2. Forward substitution: L*Y = P*B
            for k in 0..n {
                for i in (k + 1)..n {
                    let mult = lu[i * n + k];
                    for j in 0..nrhs {
                        x[i * nrhs + j] -= mult * x[k * nrhs + j];
                    }
                }
            }

            // 3. Back substitution: U*X = Y
            for k in (0..n).rev() {
                let diag = lu[k * n + k];
                if diag.abs() < f64::EPSILON {
                    return OblasReturn::Singular as c_int;
                }
                for j in 0..nrhs {
                    x[k * nrhs + j] /= diag;
                }
                for i in 0..k {
                    let mult = lu[i * n + k];
                    for j in 0..nrhs {
                        x[i * nrhs + j] -= mult * x[k * nrhs + j];
                    }
                }
            }
        }
        OblasTranspose::Trans | OblasTranspose::ConjTrans => {
            // Solve A^T*X = B
            // 1. Forward substitution: U^T*Y = B
            for k in 0..n {
                let diag = lu[k * n + k];
                if diag.abs() < f64::EPSILON {
                    return OblasReturn::Singular as c_int;
                }
                for j in 0..nrhs {
                    x[k * nrhs + j] /= diag;
                }
                for i in (k + 1)..n {
                    let mult = lu[k * n + i];
                    for j in 0..nrhs {
                        x[i * nrhs + j] -= mult * x[k * nrhs + j];
                    }
                }
            }

            // 2. Back substitution: L^T*Z = Y
            for k in (0..n).rev() {
                for i in 0..k {
                    let mult = lu[k * n + i];
                    for j in 0..nrhs {
                        x[i * nrhs + j] -= mult * x[k * nrhs + j];
                    }
                }
            }

            // 3. Apply permutation: X = P^T*Z
            for k in (0..n).rev() {
                let pk = (ipiv_slice[k] - 1) as usize;
                if k != pk {
                    for j in 0..nrhs {
                        let tmp = x[k * nrhs + j];
                        x[k * nrhs + j] = x[pk * nrhs + j];
                        x[pk * nrhs + j] = tmp;
                    }
                }
            }
        }
    }

    // Copy solution back to b
    for i in 0..n {
        for j in 0..nrhs {
            let idx = if row_major { i * ldb + j } else { j * ldb + i };
            *b.add(idx) = x[i * nrhs + j];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// SPOTRS - Solve from Cholesky factors (single precision)
// =============================================================================

/// Solves a system of linear equations A*X = B using Cholesky factorization.
///
/// The Cholesky factorization should have been computed by oblas_spotrf.
///
/// # Safety
/// - `a` must point to a valid n x n matrix containing the Cholesky factor
/// - `b` must point to a valid n x nrhs matrix (overwritten with solution)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_spotrs(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    nrhs: c_int,
    a: *const f32,
    lda: c_int,
    b: *mut f32,
    ldb: c_int,
) -> c_int {
    if n <= 0 || nrhs <= 0 || a.is_null() || b.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let nrhs = nrhs as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;

    // Read the Cholesky factor (L or U)
    let mut l = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            l[i * n + j] = *a.add(idx);
        }
    }

    // Read B
    let mut x = vec![0.0f32; n * nrhs];
    for i in 0..n {
        for j in 0..nrhs {
            let idx = if row_major { i * ldb + j } else { j * ldb + i };
            x[i * nrhs + j] = *b.add(idx);
        }
    }

    if upper {
        // A = U^T * U, solve U^T * U * X = B
        // 1. Forward substitution: U^T * Y = B
        for k in 0..n {
            let diag = l[k * n + k];
            if diag.abs() < f32::EPSILON {
                return OblasReturn::Singular as c_int;
            }
            for j in 0..nrhs {
                x[k * nrhs + j] /= diag;
            }
            for i in (k + 1)..n {
                let mult = l[k * n + i]; // U^T[i,k] = U[k,i]
                for j in 0..nrhs {
                    x[i * nrhs + j] -= mult * x[k * nrhs + j];
                }
            }
        }

        // 2. Back substitution: U * X = Y
        for k in (0..n).rev() {
            let diag = l[k * n + k];
            for j in 0..nrhs {
                x[k * nrhs + j] /= diag;
            }
            for i in 0..k {
                let mult = l[i * n + k]; // U[i,k]
                for j in 0..nrhs {
                    x[i * nrhs + j] -= mult * x[k * nrhs + j];
                }
            }
        }
    } else {
        // A = L * L^T, solve L * L^T * X = B
        // 1. Forward substitution: L * Y = B
        for k in 0..n {
            let diag = l[k * n + k];
            if diag.abs() < f32::EPSILON {
                return OblasReturn::Singular as c_int;
            }
            for j in 0..nrhs {
                x[k * nrhs + j] /= diag;
            }
            for i in (k + 1)..n {
                let mult = l[i * n + k]; // L[i,k]
                for j in 0..nrhs {
                    x[i * nrhs + j] -= mult * x[k * nrhs + j];
                }
            }
        }

        // 2. Back substitution: L^T * X = Y
        for k in (0..n).rev() {
            let diag = l[k * n + k];
            for j in 0..nrhs {
                x[k * nrhs + j] /= diag;
            }
            for i in 0..k {
                let mult = l[k * n + i]; // L^T[i,k] = L[k,i]
                for j in 0..nrhs {
                    x[i * nrhs + j] -= mult * x[k * nrhs + j];
                }
            }
        }
    }

    // Copy solution back to b
    for i in 0..n {
        for j in 0..nrhs {
            let idx = if row_major { i * ldb + j } else { j * ldb + i };
            *b.add(idx) = x[i * nrhs + j];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DPOTRS - Solve from Cholesky factors (double precision)
// =============================================================================

/// Solves a system of linear equations A*X = B using Cholesky factorization.
///
/// The Cholesky factorization should have been computed by oblas_dpotrf.
///
/// # Safety
/// - `a` must point to a valid n x n matrix containing the Cholesky factor
/// - `b` must point to a valid n x nrhs matrix (overwritten with solution)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dpotrs(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    nrhs: c_int,
    a: *const f64,
    lda: c_int,
    b: *mut f64,
    ldb: c_int,
) -> c_int {
    if n <= 0 || nrhs <= 0 || a.is_null() || b.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let nrhs = nrhs as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;

    // Read the Cholesky factor
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            l[i * n + j] = *a.add(idx);
        }
    }

    // Read B
    let mut x = vec![0.0f64; n * nrhs];
    for i in 0..n {
        for j in 0..nrhs {
            let idx = if row_major { i * ldb + j } else { j * ldb + i };
            x[i * nrhs + j] = *b.add(idx);
        }
    }

    if upper {
        // A = U^T * U, solve U^T * U * X = B
        // 1. Forward substitution: U^T * Y = B
        for k in 0..n {
            let diag = l[k * n + k];
            if diag.abs() < f64::EPSILON {
                return OblasReturn::Singular as c_int;
            }
            for j in 0..nrhs {
                x[k * nrhs + j] /= diag;
            }
            for i in (k + 1)..n {
                let mult = l[k * n + i];
                for j in 0..nrhs {
                    x[i * nrhs + j] -= mult * x[k * nrhs + j];
                }
            }
        }

        // 2. Back substitution: U * X = Y
        for k in (0..n).rev() {
            let diag = l[k * n + k];
            for j in 0..nrhs {
                x[k * nrhs + j] /= diag;
            }
            for i in 0..k {
                let mult = l[i * n + k];
                for j in 0..nrhs {
                    x[i * nrhs + j] -= mult * x[k * nrhs + j];
                }
            }
        }
    } else {
        // A = L * L^T, solve L * L^T * X = B
        // 1. Forward substitution: L * Y = B
        for k in 0..n {
            let diag = l[k * n + k];
            if diag.abs() < f64::EPSILON {
                return OblasReturn::Singular as c_int;
            }
            for j in 0..nrhs {
                x[k * nrhs + j] /= diag;
            }
            for i in (k + 1)..n {
                let mult = l[i * n + k];
                for j in 0..nrhs {
                    x[i * nrhs + j] -= mult * x[k * nrhs + j];
                }
            }
        }

        // 2. Back substitution: L^T * X = Y
        for k in (0..n).rev() {
            let diag = l[k * n + k];
            for j in 0..nrhs {
                x[k * nrhs + j] /= diag;
            }
            for i in 0..k {
                let mult = l[k * n + i];
                for j in 0..nrhs {
                    x[i * nrhs + j] -= mult * x[k * nrhs + j];
                }
            }
        }
    }

    // Copy solution back to b
    for i in 0..n {
        for j in 0..nrhs {
            let idx = if row_major { i * ldb + j } else { j * ldb + i };
            *b.add(idx) = x[i * nrhs + j];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// SGELS - Least squares solve (single precision)
// =============================================================================

/// Solves overdetermined or underdetermined linear systems using QR or LQ
/// factorization.
///
/// Solves the least squares problem: min ||A*X - B||_2
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `b` must point to a valid max(m,n) x nrhs matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sgels(
    layout: OblasLayout,
    trans: OblasTranspose,
    m: c_int,
    n: c_int,
    nrhs: c_int,
    a: *mut f32,
    lda: c_int,
    b: *mut f32,
    ldb: c_int,
) -> c_int {
    if m <= 0 || n <= 0 || nrhs <= 0 || a.is_null() || b.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let nrhs = nrhs as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let no_trans = trans == OblasTranspose::NoTrans;

    // For overdetermined systems (m >= n) with no transpose:
    // Solve min ||Ax - b|| via QR factorization
    // For underdetermined systems (m < n):
    // Find minimum norm solution via QR of A^T

    if no_trans && m >= n {
        // Overdetermined: Use QR factorization
        // Build matrix A
        let mut mat_a = Mat::<f32>::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                let idx = if row_major { i * lda + j } else { j * lda + i };
                mat_a[(i, j)] = *a.add(idx);
            }
        }

        // Build matrix B
        let mut mat_b = Mat::<f32>::zeros(m, nrhs);
        for i in 0..m {
            for j in 0..nrhs {
                let idx = if row_major { i * ldb + j } else { j * ldb + i };
                mat_b[(i, j)] = *b.add(idx);
            }
        }

        // Compute QR factorization
        let qr_result = match qr::Qr::compute(mat_a.as_ref()) {
            Ok(qr) => qr,
            Err(_) => return OblasReturn::InvalidArg as c_int,
        };

        let q = qr_result.q();
        let r = qr_result.r();

        // Compute Q^T * B
        let mut qt_b = Mat::<f32>::zeros(m, nrhs);
        for i in 0..m {
            for j in 0..nrhs {
                let mut sum = 0.0f32;
                for k in 0..m {
                    sum += q[(k, i)] * mat_b[(k, j)];
                }
                qt_b[(i, j)] = sum;
            }
        }

        // Back-substitution to solve R * x = Q^T * b
        let mut x = Mat::<f32>::zeros(n, nrhs);
        for col in 0..nrhs {
            for i in (0..n).rev() {
                let diag = r[(i, i)];
                if diag.abs() < f32::EPSILON {
                    return OblasReturn::Singular as c_int;
                }
                let mut sum = qt_b[(i, col)];
                for j in (i + 1)..n {
                    sum -= r[(i, j)] * x[(j, col)];
                }
                x[(i, col)] = sum / diag;
            }
        }

        // Write solution back to b
        let max_mn = m.max(n);
        for i in 0..max_mn {
            for j in 0..nrhs {
                let idx = if row_major { i * ldb + j } else { j * ldb + i };
                if i < n {
                    *b.add(idx) = x[(i, j)];
                } else {
                    *b.add(idx) = 0.0;
                }
            }
        }

        // Write R back to A (upper triangular part)
        for i in 0..m {
            for j in 0..n {
                let idx = if row_major { i * lda + j } else { j * lda + i };
                if i < r.nrows() && j < r.ncols() {
                    *a.add(idx) = r[(i, j)];
                }
            }
        }
    } else if no_trans && m < n {
        // Underdetermined: minimum norm solution via QR of A^T
        // Build A^T
        let mut mat_at = Mat::<f32>::zeros(n, m);
        for i in 0..m {
            for j in 0..n {
                let idx = if row_major { i * lda + j } else { j * lda + i };
                mat_at[(j, i)] = *a.add(idx);
            }
        }

        // Build B
        let mut mat_b = Mat::<f32>::zeros(m, nrhs);
        for i in 0..m {
            for j in 0..nrhs {
                let idx = if row_major { i * ldb + j } else { j * ldb + i };
                mat_b[(i, j)] = *b.add(idx);
            }
        }

        // QR of A^T
        let qr_result = match qr::Qr::compute(mat_at.as_ref()) {
            Ok(qr) => qr,
            Err(_) => return OblasReturn::InvalidArg as c_int,
        };

        let q = qr_result.q();
        let r = qr_result.r();

        // Solve R^T * y = b (forward substitution)
        let mut y = Mat::<f32>::zeros(m, nrhs);
        for col in 0..nrhs {
            for i in 0..m {
                let diag = r[(i, i)];
                if diag.abs() < f32::EPSILON {
                    return OblasReturn::Singular as c_int;
                }
                let mut sum = mat_b[(i, col)];
                for j in 0..i {
                    sum -= r[(j, i)] * y[(j, col)]; // R^T element
                }
                y[(i, col)] = sum / diag;
            }
        }

        // x = Q * y (minimum norm solution)
        let mut x = Mat::<f32>::zeros(n, nrhs);
        for i in 0..n {
            for j in 0..nrhs {
                let mut sum = 0.0f32;
                for k in 0..m {
                    sum += q[(i, k)] * y[(k, j)];
                }
                x[(i, j)] = sum;
            }
        }

        // Write solution back
        for i in 0..n {
            for j in 0..nrhs {
                let idx = if row_major { i * ldb + j } else { j * ldb + i };
                *b.add(idx) = x[(i, j)];
            }
        }
    } else {
        // Transpose case: solve min ||A^T x - b||
        // Build A
        let mut mat_a = Mat::<f32>::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                let idx = if row_major { i * lda + j } else { j * lda + i };
                mat_a[(i, j)] = *a.add(idx);
            }
        }

        // Build B
        let mut mat_b = Mat::<f32>::zeros(n, nrhs);
        for i in 0..n {
            for j in 0..nrhs {
                let idx = if row_major { i * ldb + j } else { j * ldb + i };
                mat_b[(i, j)] = *b.add(idx);
            }
        }

        // QR of A^T
        let mut mat_at = Mat::<f32>::zeros(n, m);
        for i in 0..m {
            for j in 0..n {
                mat_at[(j, i)] = mat_a[(i, j)];
            }
        }

        let qr_result = match qr::Qr::compute(mat_at.as_ref()) {
            Ok(qr) => qr,
            Err(_) => return OblasReturn::InvalidArg as c_int,
        };

        let q = qr_result.q();
        let r = qr_result.r();
        let rank = m.min(n);

        // Q^T * b
        let mut qt_b = Mat::<f32>::zeros(n, nrhs);
        for i in 0..n {
            for j in 0..nrhs {
                let mut sum = 0.0f32;
                for k in 0..n {
                    sum += q[(k, i)] * mat_b[(k, j)];
                }
                qt_b[(i, j)] = sum;
            }
        }

        // Back-substitution
        let mut x = Mat::<f32>::zeros(m, nrhs);
        for col in 0..nrhs {
            for i in (0..rank).rev() {
                let diag = r[(i, i)];
                if diag.abs() < f32::EPSILON {
                    return OblasReturn::Singular as c_int;
                }
                let mut sum = qt_b[(i, col)];
                for j in (i + 1)..rank {
                    sum -= r[(i, j)] * x[(j, col)];
                }
                x[(i, col)] = sum / diag;
            }
        }

        // Write solution back
        for i in 0..m {
            for j in 0..nrhs {
                let idx = if row_major { i * ldb + j } else { j * ldb + i };
                *b.add(idx) = x[(i, j)];
            }
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DGELS - Least squares solve (double precision)
// =============================================================================

/// Solves overdetermined or underdetermined linear systems using QR or LQ
/// factorization.
///
/// Solves the least squares problem: min ||A*X - B||_2
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `b` must point to a valid max(m,n) x nrhs matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dgels(
    layout: OblasLayout,
    trans: OblasTranspose,
    m: c_int,
    n: c_int,
    nrhs: c_int,
    a: *mut f64,
    lda: c_int,
    b: *mut f64,
    ldb: c_int,
) -> c_int {
    if m <= 0 || n <= 0 || nrhs <= 0 || a.is_null() || b.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let nrhs = nrhs as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let no_trans = trans == OblasTranspose::NoTrans;

    if no_trans && m >= n {
        // Overdetermined: Use QR factorization
        let mut mat_a = Mat::<f64>::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                let idx = if row_major { i * lda + j } else { j * lda + i };
                mat_a[(i, j)] = *a.add(idx);
            }
        }

        let mut mat_b = Mat::<f64>::zeros(m, nrhs);
        for i in 0..m {
            for j in 0..nrhs {
                let idx = if row_major { i * ldb + j } else { j * ldb + i };
                mat_b[(i, j)] = *b.add(idx);
            }
        }

        let qr_result = match qr::Qr::compute(mat_a.as_ref()) {
            Ok(qr) => qr,
            Err(_) => return OblasReturn::InvalidArg as c_int,
        };

        let q = qr_result.q();
        let r = qr_result.r();

        // Compute Q^T * B
        let mut qt_b = Mat::<f64>::zeros(m, nrhs);
        for i in 0..m {
            for j in 0..nrhs {
                let mut sum = 0.0f64;
                for k in 0..m {
                    sum += q[(k, i)] * mat_b[(k, j)];
                }
                qt_b[(i, j)] = sum;
            }
        }

        // Back-substitution
        let mut x = Mat::<f64>::zeros(n, nrhs);
        for col in 0..nrhs {
            for i in (0..n).rev() {
                let diag = r[(i, i)];
                if diag.abs() < f64::EPSILON {
                    return OblasReturn::Singular as c_int;
                }
                let mut sum = qt_b[(i, col)];
                for j in (i + 1)..n {
                    sum -= r[(i, j)] * x[(j, col)];
                }
                x[(i, col)] = sum / diag;
            }
        }

        // Write solution back
        let max_mn = m.max(n);
        for i in 0..max_mn {
            for j in 0..nrhs {
                let idx = if row_major { i * ldb + j } else { j * ldb + i };
                if i < n {
                    *b.add(idx) = x[(i, j)];
                } else {
                    *b.add(idx) = 0.0;
                }
            }
        }

        // Write R back to A
        for i in 0..m {
            for j in 0..n {
                let idx = if row_major { i * lda + j } else { j * lda + i };
                if i < r.nrows() && j < r.ncols() {
                    *a.add(idx) = r[(i, j)];
                }
            }
        }
    } else if no_trans && m < n {
        // Underdetermined: minimum norm solution
        let mut mat_at = Mat::<f64>::zeros(n, m);
        for i in 0..m {
            for j in 0..n {
                let idx = if row_major { i * lda + j } else { j * lda + i };
                mat_at[(j, i)] = *a.add(idx);
            }
        }

        let mut mat_b = Mat::<f64>::zeros(m, nrhs);
        for i in 0..m {
            for j in 0..nrhs {
                let idx = if row_major { i * ldb + j } else { j * ldb + i };
                mat_b[(i, j)] = *b.add(idx);
            }
        }

        let qr_result = match qr::Qr::compute(mat_at.as_ref()) {
            Ok(qr) => qr,
            Err(_) => return OblasReturn::InvalidArg as c_int,
        };

        let q = qr_result.q();
        let r = qr_result.r();

        // Solve R^T * y = b
        let mut y = Mat::<f64>::zeros(m, nrhs);
        for col in 0..nrhs {
            for i in 0..m {
                let diag = r[(i, i)];
                if diag.abs() < f64::EPSILON {
                    return OblasReturn::Singular as c_int;
                }
                let mut sum = mat_b[(i, col)];
                for j in 0..i {
                    sum -= r[(j, i)] * y[(j, col)];
                }
                y[(i, col)] = sum / diag;
            }
        }

        // x = Q * y
        let mut x = Mat::<f64>::zeros(n, nrhs);
        for i in 0..n {
            for j in 0..nrhs {
                let mut sum = 0.0f64;
                for k in 0..m {
                    sum += q[(i, k)] * y[(k, j)];
                }
                x[(i, j)] = sum;
            }
        }

        // Write solution back
        for i in 0..n {
            for j in 0..nrhs {
                let idx = if row_major { i * ldb + j } else { j * ldb + i };
                *b.add(idx) = x[(i, j)];
            }
        }
    } else {
        // Transpose case
        let mut mat_a = Mat::<f64>::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                let idx = if row_major { i * lda + j } else { j * lda + i };
                mat_a[(i, j)] = *a.add(idx);
            }
        }

        let mut mat_b = Mat::<f64>::zeros(n, nrhs);
        for i in 0..n {
            for j in 0..nrhs {
                let idx = if row_major { i * ldb + j } else { j * ldb + i };
                mat_b[(i, j)] = *b.add(idx);
            }
        }

        let mut mat_at = Mat::<f64>::zeros(n, m);
        for i in 0..m {
            for j in 0..n {
                mat_at[(j, i)] = mat_a[(i, j)];
            }
        }

        let qr_result = match qr::Qr::compute(mat_at.as_ref()) {
            Ok(qr) => qr,
            Err(_) => return OblasReturn::InvalidArg as c_int,
        };

        let q = qr_result.q();
        let r = qr_result.r();
        let rank = m.min(n);

        let mut qt_b = Mat::<f64>::zeros(n, nrhs);
        for i in 0..n {
            for j in 0..nrhs {
                let mut sum = 0.0f64;
                for k in 0..n {
                    sum += q[(k, i)] * mat_b[(k, j)];
                }
                qt_b[(i, j)] = sum;
            }
        }

        let mut x = Mat::<f64>::zeros(m, nrhs);
        for col in 0..nrhs {
            for i in (0..rank).rev() {
                let diag = r[(i, i)];
                if diag.abs() < f64::EPSILON {
                    return OblasReturn::Singular as c_int;
                }
                let mut sum = qt_b[(i, col)];
                for j in (i + 1)..rank {
                    sum -= r[(i, j)] * x[(j, col)];
                }
                x[(i, col)] = sum / diag;
            }
        }

        for i in 0..m {
            for j in 0..nrhs {
                let idx = if row_major { i * ldb + j } else { j * ldb + i };
                *b.add(idx) = x[(i, j)];
            }
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
