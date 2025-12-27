//! LAPACK FFI - Determinant computation routines.

use crate::types::*;
use oxiblas_lapack::cholesky::Cholesky;
use oxiblas_lapack::lu::Lu;
use oxiblas_lapack::utils::det;
use oxiblas_matrix::Mat;
use std::ffi::c_int;

// =============================================================================
// SDET - Determinant (single precision)
// =============================================================================

/// Computes the determinant of a general square matrix A (single precision).
///
/// Uses LU decomposition internally.
///
/// # Arguments
/// * `n` - Order of the matrix A (n x n)
/// * `a` - The n x n matrix
/// * `lda` - Leading dimension of A
/// * `result` - Pointer to store the determinant value
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * 1 if matrix is singular
///
/// # Safety
/// - `a` must point to a valid n x n matrix
/// - `result` must point to valid storage for a f32
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sdet(
    layout: OblasLayout,
    n: c_int,
    a: *const f32,
    lda: c_int,
    result: *mut f32,
) -> c_int {
    if n <= 0 || a.is_null() || result.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat
    let mut mat: Mat<f32> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat[(i, j)] = *a.add(idx);
        }
    }

    // Compute determinant
    match det(mat.as_ref()) {
        Ok(d) => {
            *result = d;
            OblasReturn::Success as c_int
        }
        Err(_) => {
            *result = 0.0;
            OblasReturn::Singular as c_int
        }
    }
}

// =============================================================================
// DDET - Determinant (double precision)
// =============================================================================

/// Computes the determinant of a general square matrix A (double precision).
///
/// Uses LU decomposition internally.
///
/// # Arguments
/// * `n` - Order of the matrix A (n x n)
/// * `a` - The n x n matrix
/// * `lda` - Leading dimension of A
/// * `result` - Pointer to store the determinant value
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * 1 if matrix is singular
///
/// # Safety
/// - `a` must point to a valid n x n matrix
/// - `result` must point to valid storage for a f64
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ddet(
    layout: OblasLayout,
    n: c_int,
    a: *const f64,
    lda: c_int,
    result: *mut f64,
) -> c_int {
    if n <= 0 || a.is_null() || result.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat
    let mut mat: Mat<f64> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat[(i, j)] = *a.add(idx);
        }
    }

    // Compute determinant
    match det(mat.as_ref()) {
        Ok(d) => {
            *result = d;
            OblasReturn::Success as c_int
        }
        Err(_) => {
            *result = 0.0;
            OblasReturn::Singular as c_int
        }
    }
}

// =============================================================================
// SDETLU - Determinant from LU factors (single precision)
// =============================================================================

/// Computes the determinant from LU factors (single precision).
///
/// Given the LU factorization computed by oblas_sgetrf, this computes
/// the determinant as the product of diagonal elements times the sign
/// from the permutation.
///
/// # Arguments
/// * `n` - Order of the matrix A (n x n)
/// * `lu` - The LU factors from sgetrf
/// * `lda` - Leading dimension of A
/// * `ipiv` - The pivot indices from sgetrf (1-based)
/// * `result` - Pointer to store the determinant value
///
/// # Safety
/// - `lu` must point to a valid n x n matrix containing LU factors
/// - `ipiv` must point to n pivot indices
/// - `result` must point to valid storage for a f32
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sdetlu(
    layout: OblasLayout,
    n: c_int,
    lu: *const f32,
    lda: c_int,
    ipiv: *const c_int,
    result: *mut f32,
) -> c_int {
    if n <= 0 || lu.is_null() || ipiv.is_null() || result.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Product of diagonal elements
    let mut det_val: f32 = 1.0;
    for i in 0..n {
        let idx = if row_major { i * lda + i } else { i * lda + i };
        det_val *= *lu.add(idx);
    }

    // Count row swaps from pivot vector (ipiv is 1-based)
    let mut swap_count = 0;
    for i in 0..n {
        let pivot = *ipiv.add(i) as usize;
        if pivot != i + 1 {
            swap_count += 1;
        }
    }

    // Adjust sign based on permutation
    if swap_count % 2 == 1 {
        det_val = -det_val;
    }

    *result = det_val;
    OblasReturn::Success as c_int
}

// =============================================================================
// DDETLU - Determinant from LU factors (double precision)
// =============================================================================

/// Computes the determinant from LU factors (double precision).
///
/// Given the LU factorization computed by oblas_dgetrf, this computes
/// the determinant as the product of diagonal elements times the sign
/// from the permutation.
///
/// # Arguments
/// * `n` - Order of the matrix A (n x n)
/// * `lu` - The LU factors from dgetrf
/// * `lda` - Leading dimension of A
/// * `ipiv` - The pivot indices from dgetrf (1-based)
/// * `result` - Pointer to store the determinant value
///
/// # Safety
/// - `lu` must point to a valid n x n matrix containing LU factors
/// - `ipiv` must point to n pivot indices
/// - `result` must point to valid storage for a f64
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ddetlu(
    layout: OblasLayout,
    n: c_int,
    lu: *const f64,
    lda: c_int,
    ipiv: *const c_int,
    result: *mut f64,
) -> c_int {
    if n <= 0 || lu.is_null() || ipiv.is_null() || result.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Product of diagonal elements
    let mut det_val: f64 = 1.0;
    for i in 0..n {
        let idx = if row_major { i * lda + i } else { i * lda + i };
        det_val *= *lu.add(idx);
    }

    // Count row swaps from pivot vector (ipiv is 1-based)
    let mut swap_count = 0;
    for i in 0..n {
        let pivot = *ipiv.add(i) as usize;
        if pivot != i + 1 {
            swap_count += 1;
        }
    }

    // Adjust sign based on permutation
    if swap_count % 2 == 1 {
        det_val = -det_val;
    }

    *result = det_val;
    OblasReturn::Success as c_int
}

// =============================================================================
// SLOGDET - Log-determinant (single precision)
// =============================================================================

/// Computes the log-determinant of a general square matrix A (single precision).
///
/// Returns sign(det) and ln(|det|) separately for numerical stability with
/// large matrices.
///
/// # Arguments
/// * `n` - Order of the matrix A (n x n)
/// * `a` - The n x n matrix
/// * `lda` - Leading dimension of A
/// * `sign` - Pointer to store the sign of the determinant (-1, 0, or 1)
/// * `logabsdet` - Pointer to store ln(|det(A)|)
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * 1 if matrix is singular (sign=0, logabsdet=-inf)
///
/// # Safety
/// - `a` must point to a valid n x n matrix
/// - `sign` and `logabsdet` must point to valid storage
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_slogdet(
    layout: OblasLayout,
    n: c_int,
    a: *const f32,
    lda: c_int,
    sign: *mut f32,
    logabsdet: *mut f32,
) -> c_int {
    if n <= 0 || a.is_null() || sign.is_null() || logabsdet.is_null() {
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

    // Compute determinant
    match det(mat.as_ref()) {
        Ok(d) => {
            if d == 0.0 {
                *sign = 0.0;
                *logabsdet = f32::NEG_INFINITY;
                OblasReturn::Singular as c_int
            } else {
                *sign = if d > 0.0 { 1.0 } else { -1.0 };
                *logabsdet = d.abs().ln();
                OblasReturn::Success as c_int
            }
        }
        Err(_) => {
            *sign = 0.0;
            *logabsdet = f32::NEG_INFINITY;
            OblasReturn::Singular as c_int
        }
    }
}

// =============================================================================
// DLOGDET - Log-determinant (double precision)
// =============================================================================

/// Computes the log-determinant of a general square matrix A (double precision).
///
/// Returns sign(det) and ln(|det|) separately for numerical stability with
/// large matrices.
///
/// # Arguments
/// * `n` - Order of the matrix A (n x n)
/// * `a` - The n x n matrix
/// * `lda` - Leading dimension of A
/// * `sign` - Pointer to store the sign of the determinant (-1, 0, or 1)
/// * `logabsdet` - Pointer to store ln(|det(A)|)
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * 1 if matrix is singular (sign=0, logabsdet=-inf)
///
/// # Safety
/// - `a` must point to a valid n x n matrix
/// - `sign` and `logabsdet` must point to valid storage
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dlogdet(
    layout: OblasLayout,
    n: c_int,
    a: *const f64,
    lda: c_int,
    sign: *mut f64,
    logabsdet: *mut f64,
) -> c_int {
    if n <= 0 || a.is_null() || sign.is_null() || logabsdet.is_null() {
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

    // Compute determinant
    match det(mat.as_ref()) {
        Ok(d) => {
            if d == 0.0 {
                *sign = 0.0;
                *logabsdet = f64::NEG_INFINITY;
                OblasReturn::Singular as c_int
            } else {
                *sign = if d > 0.0 { 1.0 } else { -1.0 };
                *logabsdet = d.abs().ln();
                OblasReturn::Success as c_int
            }
        }
        Err(_) => {
            *sign = 0.0;
            *logabsdet = f64::NEG_INFINITY;
            OblasReturn::Singular as c_int
        }
    }
}

// =============================================================================
// SLOGDET_CHOL - Log-determinant via Cholesky (single precision)
// =============================================================================

/// Computes the log-determinant of a symmetric positive definite matrix A using
/// Cholesky factorization (single precision).
///
/// For SPD matrices: log(det(A)) = 2 * sum(log(L\[i,i\])) where A = L*L^T.
/// This is more efficient and numerically stable than LU-based methods for SPD matrices.
///
/// # Arguments
/// * `layout` - Matrix layout
/// * `n` - Order of the matrix A (n x n)
/// * `a` - The n x n symmetric positive definite matrix
/// * `lda` - Leading dimension of A
/// * `logdet` - Pointer to store log(det(A))
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * 2 if matrix is not positive definite
///
/// # Safety
/// - `a` must point to a valid n x n symmetric positive definite matrix
/// - `logdet` must point to valid storage for a f32
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_slogdet_chol(
    layout: OblasLayout,
    n: c_int,
    a: *const f32,
    lda: c_int,
    logdet: *mut f32,
) -> c_int {
    if n <= 0 || a.is_null() || logdet.is_null() {
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

    // Compute Cholesky and log-determinant
    match Cholesky::compute(mat.as_ref()) {
        Ok(chol) => {
            *logdet = chol.log_determinant();
            OblasReturn::Success as c_int
        }
        Err(_) => {
            *logdet = f32::NEG_INFINITY;
            OblasReturn::NotConverged as c_int
        }
    }
}

// =============================================================================
// DLOGDET_CHOL - Log-determinant via Cholesky (double precision)
// =============================================================================

/// Computes the log-determinant of a symmetric positive definite matrix A using
/// Cholesky factorization (double precision).
///
/// For SPD matrices: log(det(A)) = 2 * sum(log(L\[i,i\])) where A = L*L^T.
/// This is more efficient and numerically stable than LU-based methods for SPD matrices.
///
/// # Arguments
/// * `layout` - Matrix layout
/// * `n` - Order of the matrix A (n x n)
/// * `a` - The n x n symmetric positive definite matrix
/// * `lda` - Leading dimension of A
/// * `logdet` - Pointer to store log(det(A))
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * 2 if matrix is not positive definite
///
/// # Safety
/// - `a` must point to a valid n x n symmetric positive definite matrix
/// - `logdet` must point to valid storage for a f64
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dlogdet_chol(
    layout: OblasLayout,
    n: c_int,
    a: *const f64,
    lda: c_int,
    logdet: *mut f64,
) -> c_int {
    if n <= 0 || a.is_null() || logdet.is_null() {
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

    // Compute Cholesky and log-determinant
    match Cholesky::compute(mat.as_ref()) {
        Ok(chol) => {
            *logdet = chol.log_determinant();
            OblasReturn::Success as c_int
        }
        Err(_) => {
            *logdet = f64::NEG_INFINITY;
            OblasReturn::NotConverged as c_int
        }
    }
}

// =============================================================================
// SDET_CHOL - Determinant via Cholesky (single precision)
// =============================================================================

/// Computes the determinant of a symmetric positive definite matrix A using
/// Cholesky factorization (single precision).
///
/// For SPD matrices: det(A) = product(L\[i,i\])^2 where A = L*L^T.
///
/// # Arguments
/// * `layout` - Matrix layout
/// * `n` - Order of the matrix A (n x n)
/// * `a` - The n x n symmetric positive definite matrix
/// * `lda` - Leading dimension of A
/// * `result` - Pointer to store det(A)
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * 2 if matrix is not positive definite
///
/// # Safety
/// - `a` must point to a valid n x n symmetric positive definite matrix
/// - `result` must point to valid storage for a f32
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sdet_chol(
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

    // Compute Cholesky and determinant
    match Cholesky::compute(mat.as_ref()) {
        Ok(chol) => {
            *result = chol.determinant();
            OblasReturn::Success as c_int
        }
        Err(_) => {
            *result = 0.0;
            OblasReturn::NotConverged as c_int
        }
    }
}

// =============================================================================
// DDET_CHOL - Determinant via Cholesky (double precision)
// =============================================================================

/// Computes the determinant of a symmetric positive definite matrix A using
/// Cholesky factorization (double precision).
///
/// For SPD matrices: det(A) = product(L\[i,i\])^2 where A = L*L^T.
///
/// # Arguments
/// * `layout` - Matrix layout
/// * `n` - Order of the matrix A (n x n)
/// * `a` - The n x n symmetric positive definite matrix
/// * `lda` - Leading dimension of A
/// * `result` - Pointer to store det(A)
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * 2 if matrix is not positive definite
///
/// # Safety
/// - `a` must point to a valid n x n symmetric positive definite matrix
/// - `result` must point to valid storage for a f64
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ddet_chol(
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

    // Compute Cholesky and determinant
    match Cholesky::compute(mat.as_ref()) {
        Ok(chol) => {
            *result = chol.determinant();
            OblasReturn::Success as c_int
        }
        Err(_) => {
            *result = 0.0;
            OblasReturn::NotConverged as c_int
        }
    }
}

// =============================================================================
// SABSDET - Absolute determinant (single precision)
// =============================================================================

/// Computes the absolute value of the determinant |det(A)| (single precision).
///
/// This is numerically more stable for matrices with negative determinants.
///
/// # Arguments
/// * `layout` - Matrix layout
/// * `n` - Order of the square matrix A (n x n)
/// * `a` - The n x n matrix (input, not modified)
/// * `lda` - Leading dimension of A
/// * `result` - Pointer to store |det(A)|
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * 1 if matrix is singular
///
/// # Safety
/// - `a` must point to a valid n x n matrix
/// - `result` must point to valid storage for a f32
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sabsdet(
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

    // Compute LU and determinant
    match Lu::compute(mat.as_ref()) {
        Ok(lu) => {
            let det = lu.determinant();
            *result = det.abs();
            OblasReturn::Success as c_int
        }
        Err(_) => {
            *result = 0.0;
            OblasReturn::Singular as c_int
        }
    }
}

// =============================================================================
// DABSDET - Absolute determinant (double precision)
// =============================================================================

/// Computes the absolute value of the determinant |det(A)| (double precision).
///
/// This is numerically more stable for matrices with negative determinants.
///
/// # Arguments
/// * `layout` - Matrix layout
/// * `n` - Order of the square matrix A (n x n)
/// * `a` - The n x n matrix (input, not modified)
/// * `lda` - Leading dimension of A
/// * `result` - Pointer to store |det(A)|
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * 1 if matrix is singular
///
/// # Safety
/// - `a` must point to a valid n x n matrix
/// - `result` must point to valid storage for a f64
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dabsdet(
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

    // Compute LU and determinant
    match Lu::compute(mat.as_ref()) {
        Ok(lu) => {
            let det = lu.determinant();
            *result = det.abs();
            OblasReturn::Success as c_int
        }
        Err(_) => {
            *result = 0.0;
            OblasReturn::Singular as c_int
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lapack::factorization::oblas_dgetrf;

    #[test]
    fn test_ddet_2x2() {
        // A = [[4, 7], [2, 6]] (column-major)
        // det = 4*6 - 7*2 = 10
        let a = [4.0f64, 2.0, 7.0, 6.0];
        let mut result = 0.0f64;

        unsafe {
            let ret = oblas_ddet(OblasLayout::ColMajor, 2, a.as_ptr(), 2, &mut result);
            assert_eq!(ret, 0);
            assert!((result - 10.0).abs() < 1e-10, "det = {}", result);
        }
    }

    #[test]
    fn test_ddet_3x3() {
        // A = [[1, 2, 3], [4, 5, 6], [7, 8, 10]] (column-major)
        // det = -3
        let a = [1.0f64, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0];
        let mut result = 0.0f64;

        unsafe {
            let ret = oblas_ddet(OblasLayout::ColMajor, 3, a.as_ptr(), 3, &mut result);
            assert_eq!(ret, 0);
            assert!((result - (-3.0)).abs() < 1e-10, "det = {}", result);
        }
    }

    #[test]
    fn test_ddet_singular() {
        // Singular matrix: rows are linearly dependent
        let a = [1.0f64, 2.0, 2.0, 4.0]; // column-major [[1, 2], [2, 4]]
        let mut result = 0.0f64;

        unsafe {
            let ret = oblas_ddet(OblasLayout::ColMajor, 2, a.as_ptr(), 2, &mut result);
            assert_eq!(ret, OblasReturn::Singular as c_int);
        }
    }

    #[test]
    fn test_ddet_identity() {
        // Identity matrix: det = 1
        let a = [1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut result = 0.0f64;

        unsafe {
            let ret = oblas_ddet(OblasLayout::ColMajor, 3, a.as_ptr(), 3, &mut result);
            assert_eq!(ret, 0);
            assert!((result - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_ddetlu() {
        // A = [[4, 7], [2, 6]] (column-major), det = 10
        let mut a = [4.0f64, 2.0, 7.0, 6.0];
        let mut ipiv = [0i32; 2];
        let mut result = 0.0f64;

        unsafe {
            // First compute LU
            let ret = oblas_dgetrf(
                OblasLayout::ColMajor,
                2,
                2,
                a.as_mut_ptr(),
                2,
                ipiv.as_mut_ptr(),
            );
            assert_eq!(ret, 0);

            // Then compute determinant from LU
            let ret = oblas_ddetlu(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                ipiv.as_ptr(),
                &mut result,
            );
            assert_eq!(ret, 0);
            assert!((result - 10.0).abs() < 1e-10, "det = {}", result);
        }
    }

    #[test]
    fn test_dlogdet() {
        // A = [[4, 7], [2, 6]] (column-major), det = 10
        let a = [4.0f64, 2.0, 7.0, 6.0];
        let mut sign = 0.0f64;
        let mut logabsdet = 0.0f64;

        unsafe {
            let ret = oblas_dlogdet(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                &mut sign,
                &mut logabsdet,
            );
            assert_eq!(ret, 0);
            assert!((sign - 1.0).abs() < 1e-10); // positive determinant
            assert!((logabsdet - 10.0f64.ln()).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dlogdet_negative() {
        // A = [[0, 1], [1, 0]] (column-major), det = -1
        let a = [0.0f64, 1.0, 1.0, 0.0];
        let mut sign = 0.0f64;
        let mut logabsdet = 0.0f64;

        unsafe {
            let ret = oblas_dlogdet(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                &mut sign,
                &mut logabsdet,
            );
            assert_eq!(ret, 0);
            assert!((sign - (-1.0)).abs() < 1e-10); // negative determinant
            assert!(logabsdet.abs() < 1e-10); // ln(1) = 0
        }
    }

    #[test]
    fn test_sdet_2x2() {
        let a = [4.0f32, 2.0, 7.0, 6.0];
        let mut result = 0.0f32;

        unsafe {
            let ret = oblas_sdet(OblasLayout::ColMajor, 2, a.as_ptr(), 2, &mut result);
            assert_eq!(ret, 0);
            assert!((result - 10.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dlogdet_chol() {
        // A = [[4, 2], [2, 5]] (positive definite, column-major)
        // det = 4*5 - 2*2 = 16
        // log(det) = log(16)
        let a = [4.0f64, 2.0, 2.0, 5.0];
        let mut logdet = 0.0f64;

        unsafe {
            let ret = oblas_dlogdet_chol(OblasLayout::ColMajor, 2, a.as_ptr(), 2, &mut logdet);
            assert_eq!(ret, 0);
            assert!((logdet - 16.0f64.ln()).abs() < 1e-10, "logdet = {}", logdet);
        }
    }

    #[test]
    fn test_dlogdet_chol_identity() {
        // Identity matrix: det = 1, log(det) = 0
        let a = [1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut logdet = 0.0f64;

        unsafe {
            let ret = oblas_dlogdet_chol(OblasLayout::ColMajor, 3, a.as_ptr(), 3, &mut logdet);
            assert_eq!(ret, 0);
            assert!(logdet.abs() < 1e-10);
        }
    }

    #[test]
    fn test_dlogdet_chol_not_posdef() {
        // A = [[1, 2], [2, 1]] (indefinite, not positive definite)
        let a = [1.0f64, 2.0, 2.0, 1.0];
        let mut logdet = 0.0f64;

        unsafe {
            let ret = oblas_dlogdet_chol(OblasLayout::ColMajor, 2, a.as_ptr(), 2, &mut logdet);
            // Should return error because matrix is not positive definite
            assert_eq!(ret, OblasReturn::NotConverged as c_int);
        }
    }

    #[test]
    fn test_slogdet_chol() {
        // A = [[4, 2], [2, 5]] (positive definite)
        let a = [4.0f32, 2.0, 2.0, 5.0];
        let mut logdet = 0.0f32;

        unsafe {
            let ret = oblas_slogdet_chol(OblasLayout::ColMajor, 2, a.as_ptr(), 2, &mut logdet);
            assert_eq!(ret, 0);
            assert!((logdet - 16.0f32.ln()).abs() < 1e-5);
        }
    }

    #[test]
    fn test_ddet_chol() {
        // A = [[4, 2], [2, 5]] (positive definite)
        // det = 4*5 - 2*2 = 16
        let a = [4.0f64, 2.0, 2.0, 5.0];
        let mut result = 0.0f64;

        unsafe {
            let ret = oblas_ddet_chol(OblasLayout::ColMajor, 2, a.as_ptr(), 2, &mut result);
            assert_eq!(ret, 0);
            assert!((result - 16.0).abs() < 1e-10, "det = {}", result);
        }
    }

    #[test]
    fn test_sdet_chol() {
        // A = [[4, 2], [2, 5]] (positive definite)
        let a = [4.0f32, 2.0, 2.0, 5.0];
        let mut result = 0.0f32;

        unsafe {
            let ret = oblas_sdet_chol(OblasLayout::ColMajor, 2, a.as_ptr(), 2, &mut result);
            assert_eq!(ret, 0);
            assert!((result - 16.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_ddet_chol_3x3() {
        // A = [[4, 2, 1], [2, 5, 2], [1, 2, 6]] (positive definite)
        // det = 4*(30-4) - 2*(12-2) + 1*(4-5) = 104 - 20 - 1 = 83
        let a = [4.0f64, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 6.0];
        let mut result = 0.0f64;

        unsafe {
            let ret = oblas_ddet_chol(OblasLayout::ColMajor, 3, a.as_ptr(), 3, &mut result);
            assert_eq!(ret, 0);
            assert!((result - 83.0).abs() < 1e-10, "det = {}", result);
        }
    }

    #[test]
    fn test_dabsdet_positive() {
        // A = [[4, 7], [2, 6]] (column-major)
        // det = 10, |det| = 10
        let a = [4.0f64, 2.0, 7.0, 6.0];
        let mut result = 0.0f64;

        unsafe {
            let ret = oblas_dabsdet(OblasLayout::ColMajor, 2, a.as_ptr(), 2, &mut result);
            assert_eq!(ret, 0);
            assert!((result - 10.0).abs() < 1e-10, "|det| = {}", result);
        }
    }

    #[test]
    fn test_dabsdet_negative() {
        // A = [[0, 1], [1, 0]] (column-major)
        // det = -1, |det| = 1
        let a = [0.0f64, 1.0, 1.0, 0.0];
        let mut result = 0.0f64;

        unsafe {
            let ret = oblas_dabsdet(OblasLayout::ColMajor, 2, a.as_ptr(), 2, &mut result);
            assert_eq!(ret, 0);
            assert!((result - 1.0).abs() < 1e-10, "|det| = {}", result);
        }
    }

    #[test]
    fn test_dabsdet_3x3_negative() {
        // A = [[1, 2, 3], [4, 5, 6], [7, 8, 10]] (column-major)
        // det = -3, |det| = 3
        let a = [1.0f64, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0];
        let mut result = 0.0f64;

        unsafe {
            let ret = oblas_dabsdet(OblasLayout::ColMajor, 3, a.as_ptr(), 3, &mut result);
            assert_eq!(ret, 0);
            assert!((result - 3.0).abs() < 1e-10, "|det| = {}", result);
        }
    }

    #[test]
    fn test_sabsdet_positive() {
        let a = [4.0f32, 2.0, 7.0, 6.0];
        let mut result = 0.0f32;

        unsafe {
            let ret = oblas_sabsdet(OblasLayout::ColMajor, 2, a.as_ptr(), 2, &mut result);
            assert_eq!(ret, 0);
            assert!((result - 10.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_sabsdet_negative() {
        let a = [0.0f32, 1.0, 1.0, 0.0];
        let mut result = 0.0f32;

        unsafe {
            let ret = oblas_sabsdet(OblasLayout::ColMajor, 2, a.as_ptr(), 2, &mut result);
            assert_eq!(ret, 0);
            assert!((result - 1.0).abs() < 1e-5);
        }
    }
}
