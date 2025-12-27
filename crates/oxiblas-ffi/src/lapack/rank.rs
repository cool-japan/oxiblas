//! LAPACK FFI - Matrix rank and nullity routines.

use crate::types::*;
use oxiblas_lapack::utils::{nullity, rank};
use oxiblas_matrix::Mat;
use std::ffi::c_int;

// =============================================================================
// SRANK - Matrix rank (single precision)
// =============================================================================

/// Computes the numerical rank of a matrix using SVD.
///
/// The rank is determined by counting singular values above the tolerance.
///
/// # Arguments
/// * `layout` - Matrix layout (row-major or column-major)
/// * `m` - Number of rows
/// * `n` - Number of columns
/// * `a` - The matrix (input, not modified)
/// * `lda` - Leading dimension of A
/// * `tol` - Tolerance for considering a singular value as zero (use negative for auto)
/// * `result` - Pointer to store the rank (as f32 for easy casting to int)
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * 1 if SVD failed
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `result` must point to valid storage for a f32
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_srank(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *const f32,
    lda: c_int,
    tol: f32,
    result: *mut c_int,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || result.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m_val = m as usize;
    let n_val = n as usize;
    let lda_val = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat
    let mut mat: Mat<f32> = Mat::zeros(m_val, n_val);
    for i in 0..m_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            mat[(i, j)] = *a.add(idx);
        }
    }

    let tolerance = if tol < 0.0 { None } else { Some(tol) };

    match rank(mat.as_ref(), tolerance) {
        Ok(r) => {
            *result = r as c_int;
            OblasReturn::Success as c_int
        }
        Err(_) => 1,
    }
}

// =============================================================================
// DRANK - Matrix rank (double precision)
// =============================================================================

/// Computes the numerical rank of a matrix using SVD.
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `result` must point to valid storage for a c_int
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_drank(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *const f64,
    lda: c_int,
    tol: f64,
    result: *mut c_int,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || result.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m_val = m as usize;
    let n_val = n as usize;
    let lda_val = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat
    let mut mat: Mat<f64> = Mat::zeros(m_val, n_val);
    for i in 0..m_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            mat[(i, j)] = *a.add(idx);
        }
    }

    let tolerance = if tol < 0.0 { None } else { Some(tol) };

    match rank(mat.as_ref(), tolerance) {
        Ok(r) => {
            *result = r as c_int;
            OblasReturn::Success as c_int
        }
        Err(_) => 1,
    }
}

// =============================================================================
// SNULLITY - Matrix nullity (single precision)
// =============================================================================

/// Computes the nullity (dimension of null space) of a matrix.
///
/// nullity(A) = n - rank(A) for an m×n matrix.
///
/// # Arguments
/// * `layout` - Matrix layout
/// * `m` - Number of rows
/// * `n` - Number of columns
/// * `a` - The matrix (input, not modified)
/// * `lda` - Leading dimension of A
/// * `tol` - Tolerance (negative for auto)
/// * `result` - Pointer to store the nullity
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `result` must point to valid storage for a c_int
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_snullity(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *const f32,
    lda: c_int,
    tol: f32,
    result: *mut c_int,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || result.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m_val = m as usize;
    let n_val = n as usize;
    let lda_val = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat
    let mut mat: Mat<f32> = Mat::zeros(m_val, n_val);
    for i in 0..m_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            mat[(i, j)] = *a.add(idx);
        }
    }

    let tolerance = if tol < 0.0 { None } else { Some(tol) };

    match nullity(mat.as_ref(), tolerance) {
        Ok(r) => {
            *result = r as c_int;
            OblasReturn::Success as c_int
        }
        Err(_) => 1,
    }
}

// =============================================================================
// DNULLITY - Matrix nullity (double precision)
// =============================================================================

/// Computes the nullity (dimension of null space) of a matrix.
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `result` must point to valid storage for a c_int
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dnullity(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *const f64,
    lda: c_int,
    tol: f64,
    result: *mut c_int,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || result.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m_val = m as usize;
    let n_val = n as usize;
    let lda_val = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat
    let mut mat: Mat<f64> = Mat::zeros(m_val, n_val);
    for i in 0..m_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            mat[(i, j)] = *a.add(idx);
        }
    }

    let tolerance = if tol < 0.0 { None } else { Some(tol) };

    match nullity(mat.as_ref(), tolerance) {
        Ok(r) => {
            *result = r as c_int;
            OblasReturn::Success as c_int
        }
        Err(_) => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drank_full() {
        // Full rank matrix [[1, 2], [3, 4]]
        let a = [1.0f64, 3.0, 2.0, 4.0]; // column-major
        let mut result = 0;

        unsafe {
            let ret = oblas_drank(
                OblasLayout::ColMajor,
                2,
                2,
                a.as_ptr(),
                2,
                -1.0,
                &mut result,
            );
            assert_eq!(ret, 0);
            assert_eq!(result, 2);
        }
    }

    #[test]
    fn test_drank_deficient() {
        // Rank-deficient matrix [[1, 2], [2, 4]] (rank 1)
        let a = [1.0f64, 2.0, 2.0, 4.0]; // column-major
        let mut result = 0;

        unsafe {
            let ret = oblas_drank(
                OblasLayout::ColMajor,
                2,
                2,
                a.as_ptr(),
                2,
                -1.0,
                &mut result,
            );
            assert_eq!(ret, 0);
            assert_eq!(result, 1);
        }
    }

    #[test]
    fn test_drank_identity() {
        let a = [1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; // column-major 3x3 identity
        let mut result = 0;

        unsafe {
            let ret = oblas_drank(
                OblasLayout::ColMajor,
                3,
                3,
                a.as_ptr(),
                3,
                -1.0,
                &mut result,
            );
            assert_eq!(ret, 0);
            assert_eq!(result, 3);
        }
    }

    #[test]
    fn test_drank_tall() {
        // 3×2 matrix with rank 2
        let a = [1.0f64, 3.0, 5.0, 2.0, 4.0, 6.0]; // column-major
        let mut result = 0;

        unsafe {
            let ret = oblas_drank(
                OblasLayout::ColMajor,
                3,
                2,
                a.as_ptr(),
                3,
                -1.0,
                &mut result,
            );
            assert_eq!(ret, 0);
            assert_eq!(result, 2);
        }
    }

    #[test]
    fn test_dnullity_full_rank() {
        // Full rank matrix, nullity = 0
        let a = [1.0f64, 3.0, 2.0, 4.0]; // column-major
        let mut result = 0;

        unsafe {
            let ret = oblas_dnullity(
                OblasLayout::ColMajor,
                2,
                2,
                a.as_ptr(),
                2,
                -1.0,
                &mut result,
            );
            assert_eq!(ret, 0);
            assert_eq!(result, 0);
        }
    }

    #[test]
    fn test_dnullity_rank_deficient() {
        // Rank 1 matrix, nullity = 2-1 = 1
        let a = [1.0f64, 2.0, 2.0, 4.0]; // column-major [[1, 2], [2, 4]]
        let mut result = 0;

        unsafe {
            let ret = oblas_dnullity(
                OblasLayout::ColMajor,
                2,
                2,
                a.as_ptr(),
                2,
                -1.0,
                &mut result,
            );
            assert_eq!(ret, 0);
            assert_eq!(result, 1);
        }
    }

    #[test]
    fn test_srank_full() {
        let a = [1.0f32, 3.0, 2.0, 4.0]; // column-major
        let mut result = 0;

        unsafe {
            let ret = oblas_srank(
                OblasLayout::ColMajor,
                2,
                2,
                a.as_ptr(),
                2,
                -1.0,
                &mut result,
            );
            assert_eq!(ret, 0);
            assert_eq!(result, 2);
        }
    }

    #[test]
    fn test_drank_with_custom_tolerance() {
        // Near-rank-deficient matrix
        let a = [1.0f64, 2.0 + 1e-10, 2.0, 4.0]; // column-major
        let mut result = 0;

        // With a loose tolerance, should detect as rank 1
        unsafe {
            let ret = oblas_drank(
                OblasLayout::ColMajor,
                2,
                2,
                a.as_ptr(),
                2,
                1e-6,
                &mut result,
            );
            assert_eq!(ret, 0);
            assert_eq!(result, 1);
        }

        // With a tight tolerance, should detect as rank 2
        unsafe {
            let ret = oblas_drank(
                OblasLayout::ColMajor,
                2,
                2,
                a.as_ptr(),
                2,
                1e-15,
                &mut result,
            );
            assert_eq!(ret, 0);
            assert_eq!(result, 2);
        }
    }
}
