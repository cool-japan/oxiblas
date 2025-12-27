//! LAPACK FFI - Matrix trace and nuclear norm routines.

use crate::types::*;
use oxiblas_lapack::utils::{norm_nuclear, trace};
use oxiblas_matrix::Mat;
use std::ffi::c_int;

// =============================================================================
// STRACE - Matrix trace (single precision)
// =============================================================================

/// Computes the trace of a square matrix.
///
/// tr(A) = Σ_i a_ii (sum of diagonal elements)
///
/// # Arguments
/// * `layout` - Matrix layout (row-major or column-major)
/// * `n` - Dimension of the square matrix
/// * `a` - The matrix (input, not modified)
/// * `lda` - Leading dimension of A
/// * `result` - Pointer to store the trace
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
///
/// # Safety
/// - `a` must point to a valid n x n matrix
/// - `result` must point to valid storage for a f32
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_strace(
    layout: OblasLayout,
    n: c_int,
    a: *const f32,
    lda: c_int,
    result: *mut f32,
) -> c_int {
    if n <= 0 || a.is_null() || result.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let lda_val = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat
    let mut mat: Mat<f32> = Mat::zeros(n_val, n_val);
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            mat[(i, j)] = *a.add(idx);
        }
    }

    *result = trace(mat.as_ref());
    OblasReturn::Success as c_int
}

// =============================================================================
// DTRACE - Matrix trace (double precision)
// =============================================================================

/// Computes the trace of a square matrix.
///
/// tr(A) = Σ_i a_ii (sum of diagonal elements)
///
/// # Safety
/// - `a` must point to a valid n x n matrix
/// - `result` must point to valid storage for a f64
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dtrace(
    layout: OblasLayout,
    n: c_int,
    a: *const f64,
    lda: c_int,
    result: *mut f64,
) -> c_int {
    if n <= 0 || a.is_null() || result.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let lda_val = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat
    let mut mat: Mat<f64> = Mat::zeros(n_val, n_val);
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            mat[(i, j)] = *a.add(idx);
        }
    }

    *result = trace(mat.as_ref());
    OblasReturn::Success as c_int
}

// =============================================================================
// SNORMNUC - Nuclear norm (single precision)
// =============================================================================

/// Computes the nuclear norm (trace norm / sum of singular values).
///
/// ||A||_* = Σ_i σ_i
///
/// Also known as the Schatten 1-norm.
///
/// # Arguments
/// * `layout` - Matrix layout
/// * `m` - Number of rows
/// * `n` - Number of columns
/// * `a` - The matrix (input, not modified)
/// * `lda` - Leading dimension of A
/// * `result` - Pointer to store the nuclear norm
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * 1 if SVD computation failed
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `result` must point to valid storage for a f32
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_snormnuc(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *const f32,
    lda: c_int,
    result: *mut f32,
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

    match norm_nuclear(mat.as_ref()) {
        Ok(nn) => {
            *result = nn;
            OblasReturn::Success as c_int
        }
        Err(_) => 1,
    }
}

// =============================================================================
// DNORMNUC - Nuclear norm (double precision)
// =============================================================================

/// Computes the nuclear norm (trace norm / sum of singular values).
///
/// ||A||_* = Σ_i σ_i
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `result` must point to valid storage for a f64
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dnormnuc(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *const f64,
    lda: c_int,
    result: *mut f64,
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

    match norm_nuclear(mat.as_ref()) {
        Ok(nn) => {
            *result = nn;
            OblasReturn::Success as c_int
        }
        Err(_) => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtrace_2x2() {
        // [[1, 2], [3, 4]] -> trace = 1 + 4 = 5
        let a = [1.0f64, 3.0, 2.0, 4.0]; // column-major
        let mut result = 0.0f64;

        unsafe {
            let ret = oblas_dtrace(OblasLayout::ColMajor, 2, a.as_ptr(), 2, &mut result);
            assert_eq!(ret, 0);
            assert!((result - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dtrace_identity() {
        let a = [1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; // column-major 3x3 identity
        let mut result = 0.0f64;

        unsafe {
            let ret = oblas_dtrace(OblasLayout::ColMajor, 3, a.as_ptr(), 3, &mut result);
            assert_eq!(ret, 0);
            assert!((result - 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dtrace_row_major() {
        // [[1, 2], [3, 4]] -> trace = 1 + 4 = 5
        let a = [1.0f64, 2.0, 3.0, 4.0]; // row-major
        let mut result = 0.0f64;

        unsafe {
            let ret = oblas_dtrace(OblasLayout::RowMajor, 2, a.as_ptr(), 2, &mut result);
            assert_eq!(ret, 0);
            assert!((result - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_strace_2x2() {
        let a = [1.0f32, 3.0, 2.0, 4.0]; // column-major
        let mut result = 0.0f32;

        unsafe {
            let ret = oblas_strace(OblasLayout::ColMajor, 2, a.as_ptr(), 2, &mut result);
            assert_eq!(ret, 0);
            assert!((result - 5.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dnormnuc_diagonal() {
        // [[3, 0], [0, 4]] -> nuclear norm = 3 + 4 = 7
        let a = [3.0f64, 0.0, 0.0, 4.0]; // column-major
        let mut result = 0.0f64;

        unsafe {
            let ret = oblas_dnormnuc(OblasLayout::ColMajor, 2, 2, a.as_ptr(), 2, &mut result);
            assert_eq!(ret, 0);
            assert!((result - 7.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dnormnuc_identity() {
        // 3x3 identity -> nuclear norm = 1 + 1 + 1 = 3
        let a = [1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; // column-major
        let mut result = 0.0f64;

        unsafe {
            let ret = oblas_dnormnuc(OblasLayout::ColMajor, 3, 3, a.as_ptr(), 3, &mut result);
            assert_eq!(ret, 0);
            assert!((result - 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dnormnuc_rank1() {
        // [[1, 2], [1, 2]] -> rank 1, only one nonzero singular value
        let a = [1.0f64, 1.0, 2.0, 2.0]; // column-major
        let mut result = 0.0f64;

        unsafe {
            let ret = oblas_dnormnuc(OblasLayout::ColMajor, 2, 2, a.as_ptr(), 2, &mut result);
            assert_eq!(ret, 0);
            // ||A||_F = sqrt(1+1+4+4) = sqrt(10) ≈ 3.162
            // For rank-1, nuclear norm = Frobenius norm
            assert!((result - 10.0f64.sqrt()).abs() < 1e-10);
        }
    }

    #[test]
    fn test_snormnuc_diagonal() {
        let a = [3.0f32, 0.0, 0.0, 4.0]; // column-major
        let mut result = 0.0f32;

        unsafe {
            let ret = oblas_snormnuc(OblasLayout::ColMajor, 2, 2, a.as_ptr(), 2, &mut result);
            assert_eq!(ret, 0);
            assert!((result - 7.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dnormnuc_rectangular() {
        // 2x3 matrix
        let a = [1.0f64, 4.0, 2.0, 5.0, 3.0, 6.0]; // column-major
        let mut result = 0.0f64;

        unsafe {
            let ret = oblas_dnormnuc(OblasLayout::ColMajor, 2, 3, a.as_ptr(), 2, &mut result);
            assert_eq!(ret, 0);
            // Nuclear norm should be positive
            assert!(result > 0.0);
        }
    }
}
