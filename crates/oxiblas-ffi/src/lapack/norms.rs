//! LAPACK FFI - Matrix norm routines.

use crate::types::*;
use oxiblas_lapack::utils::{norm_1, norm_2, norm_frobenius, norm_inf, norm_max};
use oxiblas_matrix::Mat;
use std::ffi::c_int;

// =============================================================================
// SLANGE - Matrix norm (single precision)
// =============================================================================

/// Returns the value of the one norm, or the Frobenius norm, or the infinity
/// norm, or the element of largest absolute value of a real matrix A.
///
/// # Arguments
/// * `norm` - Specifies the value to be returned:
///   - 'M' or 'm': max(abs(A(i,j)))
///   - '1' or 'O' or 'o': one norm (max column sum)
///   - 'I' or 'i': infinity norm (max row sum)
///   - 'F' or 'f' or 'E' or 'e': Frobenius norm
///
/// # Safety
/// - `a` must point to a valid m x n matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_slange(
    layout: OblasLayout,
    norm: u8,
    m: c_int,
    n: c_int,
    a: *const f32,
    lda: c_int,
) -> f32 {
    if m <= 0 || n <= 0 || a.is_null() {
        return 0.0;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat
    let mut mat: Mat<f32> = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat[(i, j)] = *a.add(idx);
        }
    }

    match norm {
        b'M' | b'm' => norm_max(mat.as_ref()),
        b'1' | b'O' | b'o' => norm_1(mat.as_ref()),
        b'I' | b'i' => norm_inf(mat.as_ref()),
        b'F' | b'f' | b'E' | b'e' => norm_frobenius(mat.as_ref()),
        _ => 0.0,
    }
}

// =============================================================================
// DLANGE - Matrix norm (double precision)
// =============================================================================

/// Returns the value of the one norm, or the Frobenius norm, or the infinity
/// norm, or the element of largest absolute value of a real matrix A.
///
/// # Arguments
/// * `norm` - Specifies the value to be returned:
///   - 'M' or 'm': max(abs(A(i,j)))
///   - '1' or 'O' or 'o': one norm (max column sum)
///   - 'I' or 'i': infinity norm (max row sum)
///   - 'F' or 'f' or 'E' or 'e': Frobenius norm
///
/// # Safety
/// - `a` must point to a valid m x n matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dlange(
    layout: OblasLayout,
    norm: u8,
    m: c_int,
    n: c_int,
    a: *const f64,
    lda: c_int,
) -> f64 {
    if m <= 0 || n <= 0 || a.is_null() {
        return 0.0;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat
    let mut mat: Mat<f64> = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat[(i, j)] = *a.add(idx);
        }
    }

    match norm {
        b'M' | b'm' => norm_max(mat.as_ref()),
        b'1' | b'O' | b'o' => norm_1(mat.as_ref()),
        b'I' | b'i' => norm_inf(mat.as_ref()),
        b'F' | b'f' | b'E' | b'e' => norm_frobenius(mat.as_ref()),
        _ => 0.0,
    }
}

// =============================================================================
// SLANSY - Symmetric matrix norm (single precision)
// =============================================================================

/// Returns the value of the one norm, or the Frobenius norm, or the infinity
/// norm, or the element of largest absolute value of a real symmetric matrix A.
///
/// # Safety
/// - `a` must point to a valid n x n symmetric matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_slansy(
    layout: OblasLayout,
    norm: u8,
    uplo: OblasUplo,
    n: c_int,
    a: *const f32,
    lda: c_int,
) -> f32 {
    if n <= 0 || a.is_null() {
        return 0.0;
    }

    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;

    // Convert to full symmetric Mat
    let mut mat: Mat<f32> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat[(i, j)] = *a.add(idx);
        }
    }

    // Symmetrize
    for i in 0..n {
        for j in (i + 1)..n {
            if upper {
                mat[(j, i)] = mat[(i, j)];
            } else {
                mat[(i, j)] = mat[(j, i)];
            }
        }
    }

    match norm {
        b'M' | b'm' => norm_max(mat.as_ref()),
        b'1' | b'O' | b'o' => norm_1(mat.as_ref()),
        b'I' | b'i' => norm_inf(mat.as_ref()),
        b'F' | b'f' | b'E' | b'e' => norm_frobenius(mat.as_ref()),
        _ => 0.0,
    }
}

// =============================================================================
// DLANSY - Symmetric matrix norm (double precision)
// =============================================================================

/// Returns the value of the one norm, or the Frobenius norm, or the infinity
/// norm, or the element of largest absolute value of a real symmetric matrix A.
///
/// # Safety
/// - `a` must point to a valid n x n symmetric matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dlansy(
    layout: OblasLayout,
    norm: u8,
    uplo: OblasUplo,
    n: c_int,
    a: *const f64,
    lda: c_int,
) -> f64 {
    if n <= 0 || a.is_null() {
        return 0.0;
    }

    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;

    // Convert to full symmetric Mat
    let mut mat: Mat<f64> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat[(i, j)] = *a.add(idx);
        }
    }

    // Symmetrize
    for i in 0..n {
        for j in (i + 1)..n {
            if upper {
                mat[(j, i)] = mat[(i, j)];
            } else {
                mat[(i, j)] = mat[(j, i)];
            }
        }
    }

    match norm {
        b'M' | b'm' => norm_max(mat.as_ref()),
        b'1' | b'O' | b'o' => norm_1(mat.as_ref()),
        b'I' | b'i' => norm_inf(mat.as_ref()),
        b'F' | b'f' | b'E' | b'e' => norm_frobenius(mat.as_ref()),
        _ => 0.0,
    }
}

// =============================================================================
// SLANTR - Triangular matrix norm (single precision)
// =============================================================================

/// Returns the value of the one norm, or the Frobenius norm, or the infinity
/// norm, or the element of largest absolute value of a triangular matrix A.
///
/// # Safety
/// - `a` must point to a valid m x n matrix (only triangular part is used)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_slantr(
    layout: OblasLayout,
    norm: u8,
    uplo: OblasUplo,
    diag: OblasDiag,
    m: c_int,
    n: c_int,
    a: *const f32,
    lda: c_int,
) -> f32 {
    if m <= 0 || n <= 0 || a.is_null() {
        return 0.0;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let unit_diag = diag == OblasDiag::Unit;

    // Convert to Mat (only triangular part)
    let mut mat: Mat<f32> = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            if upper {
                if j >= i {
                    if j == i && unit_diag {
                        mat[(i, j)] = 1.0;
                    } else {
                        mat[(i, j)] = *a.add(idx);
                    }
                }
            } else {
                if i >= j {
                    if i == j && unit_diag {
                        mat[(i, j)] = 1.0;
                    } else {
                        mat[(i, j)] = *a.add(idx);
                    }
                }
            }
        }
    }

    match norm {
        b'M' | b'm' => norm_max(mat.as_ref()),
        b'1' | b'O' | b'o' => norm_1(mat.as_ref()),
        b'I' | b'i' => norm_inf(mat.as_ref()),
        b'F' | b'f' | b'E' | b'e' => norm_frobenius(mat.as_ref()),
        _ => 0.0,
    }
}

// =============================================================================
// DLANTR - Triangular matrix norm (double precision)
// =============================================================================

/// Returns the value of the one norm, or the Frobenius norm, or the infinity
/// norm, or the element of largest absolute value of a triangular matrix A.
///
/// # Safety
/// - `a` must point to a valid m x n matrix (only triangular part is used)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dlantr(
    layout: OblasLayout,
    norm: u8,
    uplo: OblasUplo,
    diag: OblasDiag,
    m: c_int,
    n: c_int,
    a: *const f64,
    lda: c_int,
) -> f64 {
    if m <= 0 || n <= 0 || a.is_null() {
        return 0.0;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let unit_diag = diag == OblasDiag::Unit;

    // Convert to Mat (only triangular part)
    let mut mat: Mat<f64> = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            if upper {
                if j >= i {
                    if j == i && unit_diag {
                        mat[(i, j)] = 1.0;
                    } else {
                        mat[(i, j)] = *a.add(idx);
                    }
                }
            } else {
                if i >= j {
                    if i == j && unit_diag {
                        mat[(i, j)] = 1.0;
                    } else {
                        mat[(i, j)] = *a.add(idx);
                    }
                }
            }
        }
    }

    match norm {
        b'M' | b'm' => norm_max(mat.as_ref()),
        b'1' | b'O' | b'o' => norm_1(mat.as_ref()),
        b'I' | b'i' => norm_inf(mat.as_ref()),
        b'F' | b'f' | b'E' | b'e' => norm_frobenius(mat.as_ref()),
        _ => 0.0,
    }
}

// =============================================================================
// SNORM2 - Spectral norm (single precision)
// =============================================================================

/// Computes the 2-norm (spectral norm) of a matrix.
///
/// The spectral norm is the largest singular value: ||A||_2 = σ_max(A).
///
/// This is more expensive to compute than other norms as it requires SVD.
///
/// # Arguments
/// * `layout` - Matrix layout
/// * `m` - Number of rows
/// * `n` - Number of columns
/// * `a` - The matrix (input, not modified)
/// * `lda` - Leading dimension of A
/// * `result` - Pointer to store the spectral norm
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
pub unsafe extern "C" fn oblas_snorm2(
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

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat
    let mut mat: Mat<f32> = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat[(i, j)] = *a.add(idx);
        }
    }

    match norm_2(mat.as_ref()) {
        Ok(norm) => {
            *result = norm;
            OblasReturn::Success as c_int
        }
        Err(_) => 1,
    }
}

// =============================================================================
// DNORM2 - Spectral norm (double precision)
// =============================================================================

/// Computes the 2-norm (spectral norm) of a matrix.
///
/// The spectral norm is the largest singular value: ||A||_2 = σ_max(A).
///
/// This is more expensive to compute than other norms as it requires SVD.
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `result` must point to valid storage for a f64
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dnorm2(
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

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat
    let mut mat: Mat<f64> = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat[(i, j)] = *a.add(idx);
        }
    }

    match norm_2(mat.as_ref()) {
        Ok(norm) => {
            *result = norm;
            OblasReturn::Success as c_int
        }
        Err(_) => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dlange_frobenius() {
        // A = [[1, 2], [3, 4]] (column-major)
        // Frobenius norm = sqrt(1 + 4 + 9 + 16) = sqrt(30)
        let a = [1.0f64, 3.0, 2.0, 4.0];

        unsafe {
            let norm = oblas_dlange(OblasLayout::ColMajor, b'F', 2, 2, a.as_ptr(), 2);
            let expected = (30.0f64).sqrt();
            assert!(
                (norm - expected).abs() < 1e-10,
                "norm = {}, expected = {}",
                norm,
                expected
            );
        }
    }

    #[test]
    fn test_dlange_1norm() {
        // A = [[1, 2], [3, 4]] (column-major: [1, 3, 2, 4])
        // 1-norm = max column sum = max(|1|+|3|, |2|+|4|) = max(4, 6) = 6
        let a = [1.0f64, 3.0, 2.0, 4.0];

        unsafe {
            let norm = oblas_dlange(OblasLayout::ColMajor, b'1', 2, 2, a.as_ptr(), 2);
            assert!((norm - 6.0).abs() < 1e-10, "norm = {}", norm);
        }
    }

    #[test]
    fn test_dlange_inf_norm() {
        // A = [[1, 2], [3, 4]] (column-major)
        // Inf-norm = max row sum = max(|1|+|2|, |3|+|4|) = max(3, 7) = 7
        let a = [1.0f64, 3.0, 2.0, 4.0];

        unsafe {
            let norm = oblas_dlange(OblasLayout::ColMajor, b'I', 2, 2, a.as_ptr(), 2);
            assert!((norm - 7.0).abs() < 1e-10, "norm = {}", norm);
        }
    }

    #[test]
    fn test_dlange_max_norm() {
        // A = [[1, 2], [3, 4]]
        // Max norm = max(|1|, |2|, |3|, |4|) = 4
        let a = [1.0f64, 3.0, 2.0, 4.0];

        unsafe {
            let norm = oblas_dlange(OblasLayout::ColMajor, b'M', 2, 2, a.as_ptr(), 2);
            assert!((norm - 4.0).abs() < 1e-10, "norm = {}", norm);
        }
    }

    #[test]
    fn test_dlansy() {
        // Symmetric A = [[4, 2], [2, 5]] (column-major)
        // Frobenius norm = sqrt(16 + 4 + 4 + 25) = sqrt(49) = 7
        let a = [4.0f64, 2.0, 2.0, 5.0];

        unsafe {
            let norm = oblas_dlansy(
                OblasLayout::ColMajor,
                b'F',
                OblasUplo::Lower,
                2,
                a.as_ptr(),
                2,
            );
            assert!((norm - 7.0).abs() < 1e-10, "norm = {}", norm);
        }
    }

    #[test]
    fn test_dnorm2_diagonal() {
        // A = [[3, 0], [0, 4]] (column-major)
        // Spectral norm = max singular value = 4
        let a = [3.0f64, 0.0, 0.0, 4.0];
        let mut result = 0.0f64;

        unsafe {
            let ret = oblas_dnorm2(OblasLayout::ColMajor, 2, 2, a.as_ptr(), 2, &mut result);
            assert_eq!(ret, 0);
            assert!((result - 4.0).abs() < 1e-10, "norm = {}", result);
        }
    }

    #[test]
    fn test_dnorm2_identity() {
        // Identity matrix has spectral norm 1
        let a = [1.0f64, 0.0, 0.0, 1.0];
        let mut result = 0.0f64;

        unsafe {
            let ret = oblas_dnorm2(OblasLayout::ColMajor, 2, 2, a.as_ptr(), 2, &mut result);
            assert_eq!(ret, 0);
            assert!((result - 1.0).abs() < 1e-10, "norm = {}", result);
        }
    }

    #[test]
    fn test_snorm2_diagonal() {
        let a = [3.0f32, 0.0, 0.0, 4.0];
        let mut result = 0.0f32;

        unsafe {
            let ret = oblas_snorm2(OblasLayout::ColMajor, 2, 2, a.as_ptr(), 2, &mut result);
            assert_eq!(ret, 0);
            assert!((result - 4.0).abs() < 1e-5, "norm = {}", result);
        }
    }

    #[test]
    fn test_dnorm2_rectangular() {
        // Rectangular matrix - spectral norm is still well-defined
        // A = [[1, 2, 3]] row-major, but column-major: [1, 2, 3]
        let a = [1.0f64, 2.0, 3.0];
        let mut result = 0.0f64;

        unsafe {
            // 1x3 matrix in row-major
            let ret = oblas_dnorm2(OblasLayout::RowMajor, 1, 3, a.as_ptr(), 3, &mut result);
            assert_eq!(ret, 0);
            // For a single row, ||A||_2 = ||a||_2 = sqrt(1+4+9) = sqrt(14)
            assert!((result - 14.0f64.sqrt()).abs() < 1e-10, "norm = {}", result);
        }
    }
}
