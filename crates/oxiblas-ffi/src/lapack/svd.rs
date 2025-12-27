//! LAPACK FFI - Singular Value Decomposition routines.

use crate::types::*;
use num_complex::{Complex32, Complex64};
use oxiblas_lapack::svd;
use oxiblas_matrix::Mat;
use std::ffi::c_int;
use std::slice;

// =============================================================================
// SGESVD - SVD (single precision)
// =============================================================================

/// Computes the singular value decomposition of a general m x n matrix.
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `s` must point to an array of min(m, n) elements
/// - `u` must point to an m x m matrix (or NULL if not needed)
/// - `vt` must point to an n x n matrix (or NULL if not needed)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sgesvd(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *mut f32,
    lda: c_int,
    s: *mut f32,
    u: *mut f32,
    ldu: c_int,
    vt: *mut f32,
    ldvt: c_int,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || s.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldu = ldu as usize;
    let ldvt = ldvt as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat
    let mut mat: Mat<f32> = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat[(i, j)] = *a.add(idx);
        }
    }

    // Compute SVD
    match svd::Svd::compute(mat.as_ref()) {
        Ok(svd_result) => {
            let singular_values = svd_result.singular_values();
            let u_mat = svd_result.u();
            let vt_mat = svd_result.vt();

            let min_mn = m.min(n);
            let s_slice = slice::from_raw_parts_mut(s, min_mn);
            for i in 0..min_mn {
                s_slice[i] = singular_values[i];
            }

            if !u.is_null() {
                for i in 0..m {
                    for j in 0..m.min(min_mn) {
                        let idx = if row_major { i * ldu + j } else { j * ldu + i };
                        if j < u_mat.ncols() && i < u_mat.nrows() {
                            *u.add(idx) = u_mat[(i, j)];
                        }
                    }
                }
            }

            if !vt.is_null() {
                for i in 0..min_mn.min(n) {
                    for j in 0..n {
                        let idx = if row_major {
                            i * ldvt + j
                        } else {
                            j * ldvt + i
                        };
                        if i < vt_mat.nrows() && j < vt_mat.ncols() {
                            *vt.add(idx) = vt_mat[(i, j)];
                        }
                    }
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::NotConverged as c_int,
    }
}

// =============================================================================
// DGESVD - SVD (double precision)
// =============================================================================

/// Computes the singular value decomposition of a general m x n matrix.
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `s` must point to an array of min(m, n) elements
/// - `u` must point to an m x m matrix (or NULL if not needed)
/// - `vt` must point to an n x n matrix (or NULL if not needed)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dgesvd(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *mut f64,
    lda: c_int,
    s: *mut f64,
    u: *mut f64,
    ldu: c_int,
    vt: *mut f64,
    ldvt: c_int,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || s.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldu = ldu as usize;
    let ldvt = ldvt as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat
    let mut mat = Mat::<f64>::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat[(i, j)] = *a.add(idx);
        }
    }

    // Compute SVD
    match svd::Svd::compute(mat.as_ref()) {
        Ok(svd_result) => {
            let singular_values = svd_result.singular_values();
            let u_mat = svd_result.u();
            let vt_mat = svd_result.vt();

            // Copy singular values
            let min_mn = m.min(n);
            let s_slice = slice::from_raw_parts_mut(s, min_mn);
            for i in 0..min_mn {
                s_slice[i] = singular_values[i];
            }

            // Copy U if requested
            if !u.is_null() {
                for i in 0..m {
                    for j in 0..m.min(min_mn) {
                        let idx = if row_major { i * ldu + j } else { j * ldu + i };
                        if j < u_mat.ncols() && i < u_mat.nrows() {
                            *u.add(idx) = u_mat[(i, j)];
                        }
                    }
                }
            }

            // Copy VT if requested
            if !vt.is_null() {
                for i in 0..min_mn.min(n) {
                    for j in 0..n {
                        let idx = if row_major {
                            i * ldvt + j
                        } else {
                            j * ldvt + i
                        };
                        if i < vt_mat.nrows() && j < vt_mat.ncols() {
                            *vt.add(idx) = vt_mat[(i, j)];
                        }
                    }
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::NotConverged as c_int,
    }
}

// =============================================================================
// SGESDD - SVD divide-and-conquer (single precision)
// =============================================================================

/// Computes the SVD using divide-and-conquer algorithm.
///
/// This is typically faster for large matrices than the standard SVD.
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `s` must point to an array of min(m, n) elements
/// - `u` must point to an m x m matrix (or NULL if not needed)
/// - `vt` must point to an n x n matrix (or NULL if not needed)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sgesdd(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *mut f32,
    lda: c_int,
    s: *mut f32,
    u: *mut f32,
    ldu: c_int,
    vt: *mut f32,
    ldvt: c_int,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || s.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldu = ldu as usize;
    let ldvt = ldvt as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat
    let mut mat: Mat<f32> = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat[(i, j)] = *a.add(idx);
        }
    }

    // Compute SVD using divide-and-conquer
    match svd::SvdDc::compute(mat.as_ref()) {
        Ok(svd_result) => {
            let singular_values = svd_result.singular_values();
            let u_mat = svd_result.u();
            let vt_mat = svd_result.vt();

            let min_mn = m.min(n);
            let s_slice = slice::from_raw_parts_mut(s, min_mn);
            for i in 0..min_mn {
                s_slice[i] = singular_values[i];
            }

            if !u.is_null() {
                for i in 0..m {
                    for j in 0..m {
                        let idx = if row_major { i * ldu + j } else { j * ldu + i };
                        if i < u_mat.nrows() && j < u_mat.ncols() {
                            *u.add(idx) = u_mat[(i, j)];
                        }
                    }
                }
            }

            if !vt.is_null() {
                for i in 0..n {
                    for j in 0..n {
                        let idx = if row_major {
                            i * ldvt + j
                        } else {
                            j * ldvt + i
                        };
                        if i < vt_mat.nrows() && j < vt_mat.ncols() {
                            *vt.add(idx) = vt_mat[(i, j)];
                        }
                    }
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::NotConverged as c_int,
    }
}

// =============================================================================
// DGESDD - SVD divide-and-conquer (double precision)
// =============================================================================

/// Computes the SVD using divide-and-conquer algorithm.
///
/// This is typically faster for large matrices than the standard SVD.
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `s` must point to an array of min(m, n) elements
/// - `u` must point to an m x m matrix (or NULL if not needed)
/// - `vt` must point to an n x n matrix (or NULL if not needed)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dgesdd(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *mut f64,
    lda: c_int,
    s: *mut f64,
    u: *mut f64,
    ldu: c_int,
    vt: *mut f64,
    ldvt: c_int,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || s.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldu = ldu as usize;
    let ldvt = ldvt as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat
    let mut mat = Mat::<f64>::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat[(i, j)] = *a.add(idx);
        }
    }

    // Compute SVD using divide-and-conquer
    match svd::SvdDc::compute(mat.as_ref()) {
        Ok(svd_result) => {
            let singular_values = svd_result.singular_values();
            let u_mat = svd_result.u();
            let vt_mat = svd_result.vt();

            // Copy singular values
            let min_mn = m.min(n);
            let s_slice = slice::from_raw_parts_mut(s, min_mn);
            for i in 0..min_mn {
                s_slice[i] = singular_values[i];
            }

            // Copy U if requested
            if !u.is_null() {
                for i in 0..m {
                    for j in 0..m {
                        let idx = if row_major { i * ldu + j } else { j * ldu + i };
                        if i < u_mat.nrows() && j < u_mat.ncols() {
                            *u.add(idx) = u_mat[(i, j)];
                        }
                    }
                }
            }

            // Copy VT if requested
            if !vt.is_null() {
                for i in 0..n {
                    for j in 0..n {
                        let idx = if row_major {
                            i * ldvt + j
                        } else {
                            j * ldvt + i
                        };
                        if i < vt_mat.nrows() && j < vt_mat.ncols() {
                            *vt.add(idx) = vt_mat[(i, j)];
                        }
                    }
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::NotConverged as c_int,
    }
}

// =============================================================================
// SGESVD_THIN - Thin SVD (single precision)
// =============================================================================

/// Computes the thin (economy) SVD of a general m x n matrix.
///
/// Returns U as m x min(m,n) and Vt as min(m,n) x n, which is more
/// memory efficient than full SVD for rectangular matrices.
///
/// # Arguments
/// * `layout` - Matrix layout
/// * `m` - Number of rows
/// * `n` - Number of columns
/// * `a` - Input matrix (m x n)
/// * `lda` - Leading dimension of A
/// * `s` - Output singular values (length min(m,n))
/// * `u` - Output thin U matrix (m x min(m,n))
/// * `ldu` - Leading dimension of U
/// * `vt` - Output thin Vt matrix (min(m,n) x n)
/// * `ldvt` - Leading dimension of Vt
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `s` must point to an array of min(m,n) elements
/// - `u` must point to an m x min(m,n) matrix (or NULL if not needed)
/// - `vt` must point to a min(m,n) x n matrix (or NULL if not needed)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sgesvd_thin(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *mut f32,
    lda: c_int,
    s: *mut f32,
    u: *mut f32,
    ldu: c_int,
    vt: *mut f32,
    ldvt: c_int,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || s.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldu = ldu as usize;
    let ldvt = ldvt as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let min_mn = m.min(n);

    // Convert to Mat
    let mut mat: Mat<f32> = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat[(i, j)] = *a.add(idx);
        }
    }

    // Compute SVD
    match svd::Svd::compute(mat.as_ref()) {
        Ok(svd_result) => {
            let singular_values = svd_result.singular_values();
            let u_thin = svd_result.u_thin();
            let vt_thin = svd_result.vt_thin();

            // Copy singular values
            let s_slice = slice::from_raw_parts_mut(s, min_mn);
            for i in 0..min_mn {
                s_slice[i] = singular_values[i];
            }

            // Copy thin U (m x min_mn) if requested
            if !u.is_null() {
                for i in 0..m {
                    for j in 0..min_mn {
                        let idx = if row_major { i * ldu + j } else { j * ldu + i };
                        if i < u_thin.nrows() && j < u_thin.ncols() {
                            *u.add(idx) = u_thin[(i, j)];
                        }
                    }
                }
            }

            // Copy thin Vt (min_mn x n) if requested
            if !vt.is_null() {
                for i in 0..min_mn {
                    for j in 0..n {
                        let idx = if row_major {
                            i * ldvt + j
                        } else {
                            j * ldvt + i
                        };
                        if i < vt_thin.nrows() && j < vt_thin.ncols() {
                            *vt.add(idx) = vt_thin[(i, j)];
                        }
                    }
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::NotConverged as c_int,
    }
}

// =============================================================================
// DGESVD_THIN - Thin SVD (double precision)
// =============================================================================

/// Computes the thin (economy) SVD of a general m x n matrix.
///
/// Returns U as m x min(m,n) and Vt as min(m,n) x n, which is more
/// memory efficient than full SVD for rectangular matrices.
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `s` must point to an array of min(m,n) elements
/// - `u` must point to an m x min(m,n) matrix (or NULL if not needed)
/// - `vt` must point to a min(m,n) x n matrix (or NULL if not needed)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dgesvd_thin(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *mut f64,
    lda: c_int,
    s: *mut f64,
    u: *mut f64,
    ldu: c_int,
    vt: *mut f64,
    ldvt: c_int,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || s.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldu = ldu as usize;
    let ldvt = ldvt as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let min_mn = m.min(n);

    // Convert to Mat
    let mut mat: Mat<f64> = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat[(i, j)] = *a.add(idx);
        }
    }

    // Compute SVD
    match svd::Svd::compute(mat.as_ref()) {
        Ok(svd_result) => {
            let singular_values = svd_result.singular_values();
            let u_thin = svd_result.u_thin();
            let vt_thin = svd_result.vt_thin();

            // Copy singular values
            let s_slice = slice::from_raw_parts_mut(s, min_mn);
            for i in 0..min_mn {
                s_slice[i] = singular_values[i];
            }

            // Copy thin U (m x min_mn) if requested
            if !u.is_null() {
                for i in 0..m {
                    for j in 0..min_mn {
                        let idx = if row_major { i * ldu + j } else { j * ldu + i };
                        if i < u_thin.nrows() && j < u_thin.ncols() {
                            *u.add(idx) = u_thin[(i, j)];
                        }
                    }
                }
            }

            // Copy thin Vt (min_mn x n) if requested
            if !vt.is_null() {
                for i in 0..min_mn {
                    for j in 0..n {
                        let idx = if row_major {
                            i * ldvt + j
                        } else {
                            j * ldvt + i
                        };
                        if i < vt_thin.nrows() && j < vt_thin.ncols() {
                            *vt.add(idx) = vt_thin[(i, j)];
                        }
                    }
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::NotConverged as c_int,
    }
}

// =============================================================================
// SGESDD_THIN - Thin SVD divide-and-conquer (single precision)
// =============================================================================

/// Computes the thin SVD using divide-and-conquer algorithm.
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `s` must point to an array of min(m,n) elements
/// - `u` must point to an m x min(m,n) matrix (or NULL if not needed)
/// - `vt` must point to a min(m,n) x n matrix (or NULL if not needed)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sgesdd_thin(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *mut f32,
    lda: c_int,
    s: *mut f32,
    u: *mut f32,
    ldu: c_int,
    vt: *mut f32,
    ldvt: c_int,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || s.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldu = ldu as usize;
    let ldvt = ldvt as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let min_mn = m.min(n);

    // Convert to Mat
    let mut mat: Mat<f32> = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat[(i, j)] = *a.add(idx);
        }
    }

    // Compute SVD using divide-and-conquer
    match svd::SvdDc::compute(mat.as_ref()) {
        Ok(svd_result) => {
            let singular_values = svd_result.singular_values();
            let u_thin = svd_result.u_thin();
            let vt_thin = svd_result.vt_thin();

            // Copy singular values
            let s_slice = slice::from_raw_parts_mut(s, min_mn);
            for i in 0..min_mn {
                s_slice[i] = singular_values[i];
            }

            // Copy thin U (m x min_mn) if requested
            if !u.is_null() {
                for i in 0..m {
                    for j in 0..min_mn {
                        let idx = if row_major { i * ldu + j } else { j * ldu + i };
                        if i < u_thin.nrows() && j < u_thin.ncols() {
                            *u.add(idx) = u_thin[(i, j)];
                        }
                    }
                }
            }

            // Copy thin Vt (min_mn x n) if requested
            if !vt.is_null() {
                for i in 0..min_mn {
                    for j in 0..n {
                        let idx = if row_major {
                            i * ldvt + j
                        } else {
                            j * ldvt + i
                        };
                        if i < vt_thin.nrows() && j < vt_thin.ncols() {
                            *vt.add(idx) = vt_thin[(i, j)];
                        }
                    }
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::NotConverged as c_int,
    }
}

// =============================================================================
// DGESDD_THIN - Thin SVD divide-and-conquer (double precision)
// =============================================================================

/// Computes the thin SVD using divide-and-conquer algorithm.
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `s` must point to an array of min(m,n) elements
/// - `u` must point to an m x min(m,n) matrix (or NULL if not needed)
/// - `vt` must point to a min(m,n) x n matrix (or NULL if not needed)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dgesdd_thin(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *mut f64,
    lda: c_int,
    s: *mut f64,
    u: *mut f64,
    ldu: c_int,
    vt: *mut f64,
    ldvt: c_int,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || s.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldu = ldu as usize;
    let ldvt = ldvt as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let min_mn = m.min(n);

    // Convert to Mat
    let mut mat: Mat<f64> = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat[(i, j)] = *a.add(idx);
        }
    }

    // Compute SVD using divide-and-conquer
    match svd::SvdDc::compute(mat.as_ref()) {
        Ok(svd_result) => {
            let singular_values = svd_result.singular_values();
            let u_thin = svd_result.u_thin();
            let vt_thin = svd_result.vt_thin();

            // Copy singular values
            let s_slice = slice::from_raw_parts_mut(s, min_mn);
            for i in 0..min_mn {
                s_slice[i] = singular_values[i];
            }

            // Copy thin U (m x min_mn) if requested
            if !u.is_null() {
                for i in 0..m {
                    for j in 0..min_mn {
                        let idx = if row_major { i * ldu + j } else { j * ldu + i };
                        if i < u_thin.nrows() && j < u_thin.ncols() {
                            *u.add(idx) = u_thin[(i, j)];
                        }
                    }
                }
            }

            // Copy thin Vt (min_mn x n) if requested
            if !vt.is_null() {
                for i in 0..min_mn {
                    for j in 0..n {
                        let idx = if row_major {
                            i * ldvt + j
                        } else {
                            j * ldvt + i
                        };
                        if i < vt_thin.nrows() && j < vt_thin.ncols() {
                            *vt.add(idx) = vt_thin[(i, j)];
                        }
                    }
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::NotConverged as c_int,
    }
}

// =============================================================================
// ZGESVD - Complex SVD (double precision)
// =============================================================================

/// Computes the singular value decomposition of a complex general m x n matrix.
///
/// The SVD is: A = U * Σ * V^H, where:
/// - U is m x m unitary
/// - Σ is m x n diagonal with real non-negative singular values
/// - V^H is n x n unitary (conjugate transpose of V)
///
/// # Arguments
/// * `layout` - Matrix layout (row or column major)
/// * `m` - Number of rows
/// * `n` - Number of columns
/// * `a` - Input complex matrix (m x n)
/// * `lda` - Leading dimension of A
/// * `s` - Output real singular values (length min(m,n))
/// * `u` - Output complex U matrix (m x m, or NULL if not needed)
/// * `ldu` - Leading dimension of U
/// * `vh` - Output complex V^H matrix (n x n, or NULL if not needed)
/// * `ldvh` - Leading dimension of V^H
///
/// # Safety
/// - `a` must point to a valid m x n complex matrix
/// - `s` must point to an array of min(m, n) real elements
/// - `u` must point to an m x m complex matrix (or NULL if not needed)
/// - `vh` must point to an n x n complex matrix (or NULL if not needed)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zgesvd(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *mut OblasComplex64,
    lda: c_int,
    s: *mut f64,
    u: *mut OblasComplex64,
    ldu: c_int,
    vh: *mut OblasComplex64,
    ldvh: c_int,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || s.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldu = ldu as usize;
    let ldvh = ldvh as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat<Complex64>
    let mut mat: Mat<Complex64> = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            let c = *a.add(idx);
            mat[(i, j)] = Complex64::new(c.re, c.im);
        }
    }

    // Compute complex SVD
    match svd::ComplexSvd::compute(mat.as_ref()) {
        Ok(svd_result) => {
            let singular_values = svd_result.singular_values();
            let u_mat = svd_result.u();
            let vh_mat = svd_result.vh();

            // Copy singular values (real)
            let min_mn = m.min(n);
            let s_slice = slice::from_raw_parts_mut(s, min_mn);
            for i in 0..min_mn {
                s_slice[i] = singular_values[i];
            }

            // Copy U if requested
            if !u.is_null() {
                for i in 0..m {
                    for j in 0..m.min(min_mn) {
                        let idx = if row_major { i * ldu + j } else { j * ldu + i };
                        if j < u_mat.ncols() && i < u_mat.nrows() {
                            let c = u_mat[(i, j)];
                            *u.add(idx) = OblasComplex64 { re: c.re, im: c.im };
                        }
                    }
                }
            }

            // Copy V^H if requested
            if !vh.is_null() {
                for i in 0..min_mn.min(n) {
                    for j in 0..n {
                        let idx = if row_major {
                            i * ldvh + j
                        } else {
                            j * ldvh + i
                        };
                        if i < vh_mat.nrows() && j < vh_mat.ncols() {
                            let c = vh_mat[(i, j)];
                            *vh.add(idx) = OblasComplex64 { re: c.re, im: c.im };
                        }
                    }
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::NotConverged as c_int,
    }
}

// =============================================================================
// CGESVD - Complex SVD (single precision)
// =============================================================================

/// Computes the singular value decomposition of a complex general m x n matrix.
///
/// Single precision version of complex SVD.
///
/// # Safety
/// - `a` must point to a valid m x n complex matrix
/// - `s` must point to an array of min(m, n) real elements
/// - `u` must point to an m x m complex matrix (or NULL if not needed)
/// - `vh` must point to an n x n complex matrix (or NULL if not needed)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_cgesvd(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *mut OblasComplex32,
    lda: c_int,
    s: *mut f32,
    u: *mut OblasComplex32,
    ldu: c_int,
    vh: *mut OblasComplex32,
    ldvh: c_int,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || s.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldu = ldu as usize;
    let ldvh = ldvh as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat<Complex32>
    let mut mat: Mat<Complex32> = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            let c = *a.add(idx);
            mat[(i, j)] = Complex32::new(c.re, c.im);
        }
    }

    // Compute complex SVD
    match svd::ComplexSvd::compute(mat.as_ref()) {
        Ok(svd_result) => {
            let singular_values = svd_result.singular_values();
            let u_mat = svd_result.u();
            let vh_mat = svd_result.vh();

            // Copy singular values (real)
            let min_mn = m.min(n);
            let s_slice = slice::from_raw_parts_mut(s, min_mn);
            for i in 0..min_mn {
                s_slice[i] = singular_values[i];
            }

            // Copy U if requested
            if !u.is_null() {
                for i in 0..m {
                    for j in 0..m.min(min_mn) {
                        let idx = if row_major { i * ldu + j } else { j * ldu + i };
                        if j < u_mat.ncols() && i < u_mat.nrows() {
                            let c = u_mat[(i, j)];
                            *u.add(idx) = OblasComplex32 { re: c.re, im: c.im };
                        }
                    }
                }
            }

            // Copy V^H if requested
            if !vh.is_null() {
                for i in 0..min_mn.min(n) {
                    for j in 0..n {
                        let idx = if row_major {
                            i * ldvh + j
                        } else {
                            j * ldvh + i
                        };
                        if i < vh_mat.nrows() && j < vh_mat.ncols() {
                            let c = vh_mat[(i, j)];
                            *vh.add(idx) = OblasComplex32 { re: c.re, im: c.im };
                        }
                    }
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::NotConverged as c_int,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dgesdd_diagonal() {
        // Test SVD divide-and-conquer on a diagonal matrix
        // [[3, 0], [0, 4]] should have singular values 4 and 3
        let mut a = [3.0f64, 0.0, 0.0, 4.0]; // Row major
        let mut s = [0.0f64; 2];
        let mut u = [0.0f64; 4];
        let mut vt = [0.0f64; 4];

        unsafe {
            let result = oblas_dgesdd(
                OblasLayout::RowMajor,
                2,
                2,
                a.as_mut_ptr(),
                2,
                s.as_mut_ptr(),
                u.as_mut_ptr(),
                2,
                vt.as_mut_ptr(),
                2,
            );

            assert_eq!(result, 0);

            // Singular values should be sorted descending
            assert!((s[0] - 4.0).abs() < 1e-10);
            assert!((s[1] - 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dgesdd_reconstruction() {
        // Test that U * S * Vt gives back A
        let mut a = [1.0f64, 2.0, 3.0, 4.0]; // Row major [[1,2],[3,4]]
        let a_orig = [1.0f64, 2.0, 3.0, 4.0];
        let mut s = [0.0f64; 2];
        let mut u = [0.0f64; 4];
        let mut vt = [0.0f64; 4];

        unsafe {
            let result = oblas_dgesdd(
                OblasLayout::RowMajor,
                2,
                2,
                a.as_mut_ptr(),
                2,
                s.as_mut_ptr(),
                u.as_mut_ptr(),
                2,
                vt.as_mut_ptr(),
                2,
            );

            assert_eq!(result, 0);

            // Reconstruct: A = U * diag(s) * Vt
            // In row major:
            // u[i*2+j] = U[i,j], vt[i*2+j] = Vt[i,j]
            let mut reconstructed = [0.0f64; 4];
            for i in 0..2 {
                for j in 0..2 {
                    let mut sum = 0.0;
                    for k in 0..2 {
                        // A[i,j] = sum_k U[i,k] * s[k] * Vt[k,j]
                        sum += u[i * 2 + k] * s[k] * vt[k * 2 + j];
                    }
                    reconstructed[i * 2 + j] = sum;
                }
            }

            // Check reconstruction error
            for i in 0..4 {
                assert!(
                    (a_orig[i] - reconstructed[i]).abs() < 1e-10,
                    "Reconstruction error at {}: {} vs {}",
                    i,
                    a_orig[i],
                    reconstructed[i]
                );
            }
        }
    }

    #[test]
    fn test_sgesdd_basic() {
        let mut a = [3.0f32, 0.0, 0.0, 4.0]; // Row major diagonal
        let mut s = [0.0f32; 2];
        let mut u = [0.0f32; 4];
        let mut vt = [0.0f32; 4];

        unsafe {
            let result = oblas_sgesdd(
                OblasLayout::RowMajor,
                2,
                2,
                a.as_mut_ptr(),
                2,
                s.as_mut_ptr(),
                u.as_mut_ptr(),
                2,
                vt.as_mut_ptr(),
                2,
            );

            assert_eq!(result, 0);
            assert!((s[0] - 4.0).abs() < 1e-5);
            assert!((s[1] - 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dgesdd_tall() {
        // 3x2 matrix
        let mut a = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]; // Row major
        let mut s = [0.0f64; 2];
        let mut u = [0.0f64; 9]; // 3x3
        let mut vt = [0.0f64; 4]; // 2x2

        unsafe {
            let result = oblas_dgesdd(
                OblasLayout::RowMajor,
                3,
                2,
                a.as_mut_ptr(),
                2,
                s.as_mut_ptr(),
                u.as_mut_ptr(),
                3,
                vt.as_mut_ptr(),
                2,
            );

            assert_eq!(result, 0);

            // Singular values should be positive and sorted descending
            assert!(s[0] >= s[1]);
            assert!(s[0] > 0.0);
            assert!(s[1] > 0.0);
        }
    }

    #[test]
    fn test_dgesvd_thin_tall() {
        // 4x2 matrix - thin SVD should return U: 4x2, Vt: 2x2
        let mut a = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // Row major 4x2
        let mut s = [0.0f64; 2];
        let mut u = [0.0f64; 8]; // 4x2 (thin)
        let mut vt = [0.0f64; 4]; // 2x2

        unsafe {
            let result = oblas_dgesvd_thin(
                OblasLayout::RowMajor,
                4,
                2,
                a.as_mut_ptr(),
                2,
                s.as_mut_ptr(),
                u.as_mut_ptr(),
                2, // ldu = min(m,n) = 2
                vt.as_mut_ptr(),
                2,
            );

            assert_eq!(result, 0);
            assert!(s[0] >= s[1]);
            assert!(s[0] > 0.0);
        }
    }

    #[test]
    fn test_dgesvd_thin_wide() {
        // 2x4 matrix - thin SVD should return U: 2x2, Vt: 2x4
        let mut a = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // Row major 2x4
        let mut s = [0.0f64; 2];
        let mut u = [0.0f64; 4]; // 2x2 (thin)
        let mut vt = [0.0f64; 8]; // 2x4

        unsafe {
            let result = oblas_dgesvd_thin(
                OblasLayout::RowMajor,
                2,
                4,
                a.as_mut_ptr(),
                4,
                s.as_mut_ptr(),
                u.as_mut_ptr(),
                2,
                vt.as_mut_ptr(),
                4, // ldvt = n = 4
            );

            assert_eq!(result, 0);
            assert!(s[0] >= s[1]);
            assert!(s[0] > 0.0);
        }
    }

    #[test]
    fn test_sgesvd_thin_basic() {
        // 3x2 matrix
        let mut a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // Row major 3x2
        let mut s = [0.0f32; 2];
        let mut u = [0.0f32; 6]; // 3x2 (thin)
        let mut vt = [0.0f32; 4]; // 2x2

        unsafe {
            let result = oblas_sgesvd_thin(
                OblasLayout::RowMajor,
                3,
                2,
                a.as_mut_ptr(),
                2,
                s.as_mut_ptr(),
                u.as_mut_ptr(),
                2,
                vt.as_mut_ptr(),
                2,
            );

            assert_eq!(result, 0);
            assert!(s[0] >= s[1]);
        }
    }

    #[test]
    fn test_dgesdd_thin_tall() {
        // 4x2 matrix using divide-and-conquer
        let mut a = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // Row major 4x2
        let mut s = [0.0f64; 2];
        let mut u = [0.0f64; 8]; // 4x2 (thin)
        let mut vt = [0.0f64; 4]; // 2x2

        unsafe {
            let result = oblas_dgesdd_thin(
                OblasLayout::RowMajor,
                4,
                2,
                a.as_mut_ptr(),
                2,
                s.as_mut_ptr(),
                u.as_mut_ptr(),
                2,
                vt.as_mut_ptr(),
                2,
            );

            assert_eq!(result, 0);
            assert!(s[0] >= s[1]);
            assert!(s[0] > 0.0);
        }
    }

    #[test]
    fn test_sgesdd_thin_basic() {
        // 3x2 matrix using divide-and-conquer
        let mut a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // Row major 3x2
        let mut s = [0.0f32; 2];
        let mut u = [0.0f32; 6]; // 3x2 (thin)
        let mut vt = [0.0f32; 4]; // 2x2

        unsafe {
            let result = oblas_sgesdd_thin(
                OblasLayout::RowMajor,
                3,
                2,
                a.as_mut_ptr(),
                2,
                s.as_mut_ptr(),
                u.as_mut_ptr(),
                2,
                vt.as_mut_ptr(),
                2,
            );

            assert_eq!(result, 0);
            assert!(s[0] >= s[1]);
        }
    }

    #[test]
    fn test_zgesvd_basic() {
        // 2x2 complex matrix with real entries
        // [[3, 0], [0, 4]] should have singular values 4 and 3
        let mut a = [
            OblasComplex64 { re: 3.0, im: 0.0 },
            OblasComplex64 { re: 0.0, im: 0.0 },
            OblasComplex64 { re: 0.0, im: 0.0 },
            OblasComplex64 { re: 4.0, im: 0.0 },
        ];
        let mut s = [0.0f64; 2];
        let mut u = [OblasComplex64 { re: 0.0, im: 0.0 }; 4];
        let mut vh = [OblasComplex64 { re: 0.0, im: 0.0 }; 4];

        unsafe {
            let result = oblas_zgesvd(
                OblasLayout::RowMajor,
                2,
                2,
                a.as_mut_ptr(),
                2,
                s.as_mut_ptr(),
                u.as_mut_ptr(),
                2,
                vh.as_mut_ptr(),
                2,
            );

            assert_eq!(result, 0);
            // Singular values should be sorted descending
            assert!((s[0] - 4.0).abs() < 1e-10);
            assert!((s[1] - 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_zgesvd_complex_entries() {
        // 2x2 complex matrix
        let mut a = [
            OblasComplex64 { re: 1.0, im: 1.0 },
            OblasComplex64 { re: 2.0, im: 0.0 },
            OblasComplex64 { re: 0.0, im: 1.0 },
            OblasComplex64 { re: 3.0, im: -1.0 },
        ];
        let mut s = [0.0f64; 2];
        let mut u = [OblasComplex64 { re: 0.0, im: 0.0 }; 4];
        let mut vh = [OblasComplex64 { re: 0.0, im: 0.0 }; 4];

        unsafe {
            let result = oblas_zgesvd(
                OblasLayout::RowMajor,
                2,
                2,
                a.as_mut_ptr(),
                2,
                s.as_mut_ptr(),
                u.as_mut_ptr(),
                2,
                vh.as_mut_ptr(),
                2,
            );

            assert_eq!(result, 0);
            // Singular values should be positive and sorted descending
            assert!(s[0] >= s[1]);
            assert!(s[0] > 0.0);
            assert!(s[1] > 0.0);
        }
    }

    #[test]
    fn test_zgesvd_reconstruction() {
        // Verify that U * Σ * V^H reconstructs the original matrix
        let mut a = [
            OblasComplex64 { re: 1.0, im: 2.0 },
            OblasComplex64 { re: 3.0, im: 1.0 },
            OblasComplex64 { re: 2.0, im: -1.0 },
            OblasComplex64 { re: 4.0, im: 0.0 },
        ];
        let a_orig = a;
        let mut s = [0.0f64; 2];
        let mut u = [OblasComplex64 { re: 0.0, im: 0.0 }; 4];
        let mut vh = [OblasComplex64 { re: 0.0, im: 0.0 }; 4];

        unsafe {
            let result = oblas_zgesvd(
                OblasLayout::RowMajor,
                2,
                2,
                a.as_mut_ptr(),
                2,
                s.as_mut_ptr(),
                u.as_mut_ptr(),
                2,
                vh.as_mut_ptr(),
                2,
            );

            assert_eq!(result, 0);

            // Reconstruct: A = U * diag(s) * V^H
            let mut reconstructed = [OblasComplex64 { re: 0.0, im: 0.0 }; 4];
            for i in 0..2 {
                for j in 0..2 {
                    let mut sum_re = 0.0;
                    let mut sum_im = 0.0;
                    for k in 0..2 {
                        // A[i,j] = sum_k U[i,k] * s[k] * VH[k,j]
                        let u_ik = &u[i * 2 + k];
                        let vh_kj = &vh[k * 2 + j];
                        // (u_re + i*u_im) * s * (vh_re + i*vh_im)
                        let temp_re = u_ik.re * s[k];
                        let temp_im = u_ik.im * s[k];
                        sum_re += temp_re * vh_kj.re - temp_im * vh_kj.im;
                        sum_im += temp_re * vh_kj.im + temp_im * vh_kj.re;
                    }
                    reconstructed[i * 2 + j] = OblasComplex64 {
                        re: sum_re,
                        im: sum_im,
                    };
                }
            }

            // Check reconstruction error
            for i in 0..4 {
                assert!(
                    (a_orig[i].re - reconstructed[i].re).abs() < 1e-10,
                    "Real part mismatch at {}: {} vs {}",
                    i,
                    a_orig[i].re,
                    reconstructed[i].re
                );
                assert!(
                    (a_orig[i].im - reconstructed[i].im).abs() < 1e-10,
                    "Imag part mismatch at {}: {} vs {}",
                    i,
                    a_orig[i].im,
                    reconstructed[i].im
                );
            }
        }
    }

    #[test]
    fn test_cgesvd_basic() {
        // 2x2 complex matrix (single precision)
        let mut a = [
            OblasComplex32 { re: 3.0, im: 0.0 },
            OblasComplex32 { re: 0.0, im: 0.0 },
            OblasComplex32 { re: 0.0, im: 0.0 },
            OblasComplex32 { re: 4.0, im: 0.0 },
        ];
        let mut s = [0.0f32; 2];
        let mut u = [OblasComplex32 { re: 0.0, im: 0.0 }; 4];
        let mut vh = [OblasComplex32 { re: 0.0, im: 0.0 }; 4];

        unsafe {
            let result = oblas_cgesvd(
                OblasLayout::RowMajor,
                2,
                2,
                a.as_mut_ptr(),
                2,
                s.as_mut_ptr(),
                u.as_mut_ptr(),
                2,
                vh.as_mut_ptr(),
                2,
            );

            assert_eq!(result, 0);
            // Singular values should be sorted descending
            assert!((s[0] - 4.0).abs() < 1e-5);
            assert!((s[1] - 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_zgesvd_tall() {
        // 3x2 complex matrix
        let mut a = [
            OblasComplex64 { re: 1.0, im: 0.0 },
            OblasComplex64 { re: 2.0, im: 0.0 },
            OblasComplex64 { re: 3.0, im: 0.0 },
            OblasComplex64 { re: 4.0, im: 1.0 },
            OblasComplex64 { re: 5.0, im: -1.0 },
            OblasComplex64 { re: 6.0, im: 0.0 },
        ];
        let mut s = [0.0f64; 2];
        let mut u = [OblasComplex64 { re: 0.0, im: 0.0 }; 9];
        let mut vh = [OblasComplex64 { re: 0.0, im: 0.0 }; 4];

        unsafe {
            let result = oblas_zgesvd(
                OblasLayout::RowMajor,
                3,
                2,
                a.as_mut_ptr(),
                2,
                s.as_mut_ptr(),
                u.as_mut_ptr(),
                3,
                vh.as_mut_ptr(),
                2,
            );

            assert_eq!(result, 0);
            // Singular values should be positive and sorted descending
            assert!(s[0] >= s[1]);
            assert!(s[0] > 0.0);
        }
    }
}
