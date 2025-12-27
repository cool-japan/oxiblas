//! LAPACK FFI - LDL^T factorization routines.
//!
//! LDL^T decomposition for symmetric matrices (not necessarily positive definite).
//! A = LDL^T where L is unit lower triangular and D is diagonal.

use crate::types::*;
use oxiblas_lapack::cholesky::Ldlt;
use oxiblas_matrix::Mat;
use std::ffi::c_int;

// =============================================================================
// SSYTRF - LDL^T factorization (single precision)
// =============================================================================

/// Computes the LDL^T factorization of a symmetric matrix.
///
/// A = LDL^T where L is unit lower triangular and D is diagonal.
/// Works for symmetric indefinite matrices.
///
/// # Arguments
/// * `layout` - Matrix layout
/// * `uplo` - 'L' for lower triangle, 'U' for upper triangle (only lower used)
/// * `n` - Dimension of matrix A
/// * `a` - On input: symmetric matrix A. On output: L (strictly lower) and D (diagonal)
/// * `lda` - Leading dimension of A
/// * `d` - Array to store diagonal D (length n)
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * positive value k if matrix is singular at index k-1
///
/// # Safety
/// - `a` must point to a valid n × n matrix
/// - `d` must point to an array of length n
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ssytrf(
    layout: OblasLayout,
    _uplo: u8, // Currently only lower triangle is used
    n: c_int,
    a: *mut f32,
    lda: c_int,
    d: *mut f32,
) -> c_int {
    if n <= 0 || a.is_null() || d.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let lda_val = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert A to Mat
    let mut mat_a: Mat<f32> = Mat::zeros(n_val, n_val);
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    // Compute LDLT factorization
    let ldlt = match Ldlt::compute(mat_a.as_ref()) {
        Ok(f) => f,
        Err(oxiblas_lapack::cholesky::LdltError::Singular { index }) => {
            return (index + 1) as c_int;
        }
        Err(_) => return OblasReturn::InvalidArg as c_int,
    };

    // Get L factor and D diagonal
    let l = ldlt.l_factor();
    let d_diag = ldlt.d_diagonal();

    // Copy L to A (lower triangular, with implicit unit diagonal)
    for i in 0..n_val {
        for j in 0..=i {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            *a.add(idx) = l[(i, j)];
        }
    }

    // Copy D diagonal
    for (i, &val) in d_diag.iter().enumerate() {
        *d.add(i) = val;
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DSYTRF - LDL^T factorization (double precision)
// =============================================================================

/// Computes the LDL^T factorization of a symmetric matrix (double precision).
///
/// # Safety
/// - `a` must point to a valid n × n matrix
/// - `d` must point to an array of length n
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dsytrf(
    layout: OblasLayout,
    _uplo: u8,
    n: c_int,
    a: *mut f64,
    lda: c_int,
    d: *mut f64,
) -> c_int {
    if n <= 0 || a.is_null() || d.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let lda_val = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert A to Mat
    let mut mat_a: Mat<f64> = Mat::zeros(n_val, n_val);
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    // Compute LDLT factorization
    let ldlt = match Ldlt::compute(mat_a.as_ref()) {
        Ok(f) => f,
        Err(oxiblas_lapack::cholesky::LdltError::Singular { index }) => {
            return (index + 1) as c_int;
        }
        Err(_) => return OblasReturn::InvalidArg as c_int,
    };

    // Get L factor and D diagonal
    let l = ldlt.l_factor();
    let d_diag = ldlt.d_diagonal();

    // Copy L to A (lower triangular)
    for i in 0..n_val {
        for j in 0..=i {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            *a.add(idx) = l[(i, j)];
        }
    }

    // Copy D diagonal
    for (i, &val) in d_diag.iter().enumerate() {
        *d.add(i) = val;
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// SSYTRS - Solve from LDL^T factors (single precision)
// =============================================================================

/// Solves A*X = B using LDL^T factorization.
///
/// # Arguments
/// * `layout` - Matrix layout
/// * `uplo` - 'L' for lower triangle
/// * `n` - Dimension of matrix A
/// * `nrhs` - Number of right-hand side columns
/// * `a` - The LDL^T factors from oblas_ssytrf
/// * `lda` - Leading dimension of A
/// * `d` - The diagonal D from oblas_ssytrf
/// * `b` - On input: right-hand side B. On output: solution X
/// * `ldb` - Leading dimension of B
///
/// # Safety
/// - `a` must point to a valid n × n matrix containing LDL^T factors
/// - `d` must point to the diagonal array from factorization
/// - `b` must point to an n × nrhs matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ssytrs(
    layout: OblasLayout,
    _uplo: u8,
    n: c_int,
    nrhs: c_int,
    a: *const f32,
    lda: c_int,
    d: *const f32,
    b: *mut f32,
    ldb: c_int,
) -> c_int {
    if n <= 0 || nrhs <= 0 || a.is_null() || d.is_null() || b.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let nrhs_val = nrhs as usize;
    let lda_val = lda as usize;
    let ldb_val = ldb as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Reconstruct L matrix
    let mut l_mat: Mat<f32> = Mat::zeros(n_val, n_val);
    for i in 0..n_val {
        for j in 0..=i {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            l_mat[(i, j)] = *a.add(idx);
        }
    }

    // Read B
    let mut b_mat: Mat<f32> = Mat::zeros(n_val, nrhs_val);
    for i in 0..n_val {
        for j in 0..nrhs_val {
            let idx = if row_major {
                i * ldb_val + j
            } else {
                j * ldb_val + i
            };
            b_mat[(i, j)] = *b.add(idx);
        }
    }

    // Solve L y = b (forward substitution)
    for k in 0..nrhs_val {
        for i in 0..n_val {
            let mut sum = b_mat[(i, k)];
            for j in 0..i {
                sum -= l_mat[(i, j)] * b_mat[(j, k)];
            }
            b_mat[(i, k)] = sum / l_mat[(i, i)];
        }
    }

    // Solve D z = y
    for k in 0..nrhs_val {
        for i in 0..n_val {
            let d_ii = *d.add(i);
            if d_ii.abs() < f32::EPSILON {
                return 1; // Singular
            }
            b_mat[(i, k)] /= d_ii;
        }
    }

    // Solve L^T x = z (backward substitution)
    for k in 0..nrhs_val {
        for i in (0..n_val).rev() {
            let mut sum = b_mat[(i, k)];
            for j in (i + 1)..n_val {
                sum -= l_mat[(j, i)] * b_mat[(j, k)];
            }
            b_mat[(i, k)] = sum / l_mat[(i, i)];
        }
    }

    // Copy result back to B
    for i in 0..n_val {
        for j in 0..nrhs_val {
            let idx = if row_major {
                i * ldb_val + j
            } else {
                j * ldb_val + i
            };
            *b.add(idx) = b_mat[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DSYTRS - Solve from LDL^T factors (double precision)
// =============================================================================

/// Solves A*X = B using LDL^T factorization (double precision).
///
/// # Safety
/// - `a` must point to a valid n × n matrix containing LDL^T factors
/// - `d` must point to the diagonal array from factorization
/// - `b` must point to an n × nrhs matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dsytrs(
    layout: OblasLayout,
    _uplo: u8,
    n: c_int,
    nrhs: c_int,
    a: *const f64,
    lda: c_int,
    d: *const f64,
    b: *mut f64,
    ldb: c_int,
) -> c_int {
    if n <= 0 || nrhs <= 0 || a.is_null() || d.is_null() || b.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let nrhs_val = nrhs as usize;
    let lda_val = lda as usize;
    let ldb_val = ldb as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Reconstruct L matrix
    let mut l_mat: Mat<f64> = Mat::zeros(n_val, n_val);
    for i in 0..n_val {
        for j in 0..=i {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            l_mat[(i, j)] = *a.add(idx);
        }
    }

    // Read B
    let mut b_mat: Mat<f64> = Mat::zeros(n_val, nrhs_val);
    for i in 0..n_val {
        for j in 0..nrhs_val {
            let idx = if row_major {
                i * ldb_val + j
            } else {
                j * ldb_val + i
            };
            b_mat[(i, j)] = *b.add(idx);
        }
    }

    // Solve L y = b (forward substitution)
    for k in 0..nrhs_val {
        for i in 0..n_val {
            let mut sum = b_mat[(i, k)];
            for j in 0..i {
                sum -= l_mat[(i, j)] * b_mat[(j, k)];
            }
            b_mat[(i, k)] = sum / l_mat[(i, i)];
        }
    }

    // Solve D z = y
    for k in 0..nrhs_val {
        for i in 0..n_val {
            let d_ii = *d.add(i);
            if d_ii.abs() < f64::EPSILON {
                return 1; // Singular
            }
            b_mat[(i, k)] /= d_ii;
        }
    }

    // Solve L^T x = z (backward substitution)
    for k in 0..nrhs_val {
        for i in (0..n_val).rev() {
            let mut sum = b_mat[(i, k)];
            for j in (i + 1)..n_val {
                sum -= l_mat[(j, i)] * b_mat[(j, k)];
            }
            b_mat[(i, k)] = sum / l_mat[(i, i)];
        }
    }

    // Copy result back to B
    for i in 0..n_val {
        for j in 0..nrhs_val {
            let idx = if row_major {
                i * ldb_val + j
            } else {
                j * ldb_val + i
            };
            *b.add(idx) = b_mat[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// SINERTIA - Matrix inertia (single precision)
// =============================================================================

/// Computes the matrix inertia from LDL^T factorization.
///
/// The inertia (n+, n-, n0) counts the number of positive, negative, and zero
/// eigenvalues of a symmetric matrix.
///
/// # Arguments
/// * `layout` - Matrix layout
/// * `n` - Dimension of matrix A
/// * `a` - Symmetric matrix A (will compute LDL^T internally)
/// * `lda` - Leading dimension of A
/// * `n_positive` - Output: number of positive eigenvalues
/// * `n_negative` - Output: number of negative eigenvalues
/// * `n_zero` - Output: number of zero eigenvalues
///
/// # Safety
/// - `a` must point to a valid n × n symmetric matrix
/// - Output pointers must be valid
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sinertia(
    layout: OblasLayout,
    n: c_int,
    a: *const f32,
    lda: c_int,
    n_positive: *mut c_int,
    n_negative: *mut c_int,
    n_zero: *mut c_int,
) -> c_int {
    if n <= 0 || a.is_null() || n_positive.is_null() || n_negative.is_null() || n_zero.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let lda_val = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert A to Mat
    let mut mat_a: Mat<f32> = Mat::zeros(n_val, n_val);
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    // Compute LDLT and get inertia
    match Ldlt::compute(mat_a.as_ref()) {
        Ok(ldlt) => {
            let (pos, neg, zero) = ldlt.inertia();
            *n_positive = pos as c_int;
            *n_negative = neg as c_int;
            *n_zero = zero as c_int;
            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::InvalidArg as c_int,
    }
}

// =============================================================================
// DINERTIA - Matrix inertia (double precision)
// =============================================================================

/// Computes the matrix inertia from LDL^T factorization.
///
/// # Safety
/// - `a` must point to a valid n × n symmetric matrix
/// - Output pointers must be valid
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dinertia(
    layout: OblasLayout,
    n: c_int,
    a: *const f64,
    lda: c_int,
    n_positive: *mut c_int,
    n_negative: *mut c_int,
    n_zero: *mut c_int,
) -> c_int {
    if n <= 0 || a.is_null() || n_positive.is_null() || n_negative.is_null() || n_zero.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let lda_val = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert A to Mat
    let mut mat_a: Mat<f64> = Mat::zeros(n_val, n_val);
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    // Compute LDLT and get inertia
    match Ldlt::compute(mat_a.as_ref()) {
        Ok(ldlt) => {
            let (pos, neg, zero) = ldlt.inertia();
            *n_positive = pos as c_int;
            *n_negative = neg as c_int;
            *n_zero = zero as c_int;
            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::InvalidArg as c_int,
    }
}

// =============================================================================
// SISPOSDEF - Check positive definiteness (single precision)
// =============================================================================

/// Checks if a symmetric matrix is positive definite.
///
/// # Arguments
/// * `layout` - Matrix layout
/// * `n` - Dimension of matrix A
/// * `a` - Symmetric matrix A
/// * `lda` - Leading dimension of A
/// * `is_posdef` - Output: 1 if positive definite, 0 otherwise
///
/// # Safety
/// - `a` must point to a valid n × n symmetric matrix
/// - `is_posdef` must be a valid pointer
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sisposdef(
    layout: OblasLayout,
    n: c_int,
    a: *const f32,
    lda: c_int,
    is_posdef: *mut c_int,
) -> c_int {
    if n <= 0 || a.is_null() || is_posdef.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let lda_val = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert A to Mat
    let mut mat_a: Mat<f32> = Mat::zeros(n_val, n_val);
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    match Ldlt::compute(mat_a.as_ref()) {
        Ok(ldlt) => {
            *is_posdef = if ldlt.is_positive_definite() { 1 } else { 0 };
            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::InvalidArg as c_int,
    }
}

// =============================================================================
// DISPOSDEF - Check positive definiteness (double precision)
// =============================================================================

/// Checks if a symmetric matrix is positive definite.
///
/// # Safety
/// - `a` must point to a valid n × n symmetric matrix
/// - `is_posdef` must be a valid pointer
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_disposdef(
    layout: OblasLayout,
    n: c_int,
    a: *const f64,
    lda: c_int,
    is_posdef: *mut c_int,
) -> c_int {
    if n <= 0 || a.is_null() || is_posdef.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let lda_val = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert A to Mat
    let mut mat_a: Mat<f64> = Mat::zeros(n_val, n_val);
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    match Ldlt::compute(mat_a.as_ref()) {
        Ok(ldlt) => {
            *is_posdef = if ldlt.is_positive_definite() { 1 } else { 0 };
            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::InvalidArg as c_int,
    }
}

// =============================================================================
// SISNEGDEF - Check negative definiteness (single precision)
// =============================================================================

/// Checks if a symmetric matrix is negative definite.
///
/// # Arguments
/// * `layout` - Matrix layout
/// * `n` - Dimension of matrix A
/// * `a` - Symmetric matrix A
/// * `lda` - Leading dimension of A
/// * `is_negdef` - Output: 1 if negative definite, 0 otherwise
///
/// # Safety
/// - `a` must point to a valid n × n symmetric matrix
/// - `is_negdef` must be a valid pointer
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sisnegdef(
    layout: OblasLayout,
    n: c_int,
    a: *const f32,
    lda: c_int,
    is_negdef: *mut c_int,
) -> c_int {
    if n <= 0 || a.is_null() || is_negdef.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let lda_val = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert A to Mat
    let mut mat_a: Mat<f32> = Mat::zeros(n_val, n_val);
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    match Ldlt::compute(mat_a.as_ref()) {
        Ok(ldlt) => {
            *is_negdef = if ldlt.is_negative_definite() { 1 } else { 0 };
            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::InvalidArg as c_int,
    }
}

// =============================================================================
// DISNEGDEF - Check negative definiteness (double precision)
// =============================================================================

/// Checks if a symmetric matrix is negative definite.
///
/// # Safety
/// - `a` must point to a valid n × n symmetric matrix
/// - `is_negdef` must be a valid pointer
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_disnegdef(
    layout: OblasLayout,
    n: c_int,
    a: *const f64,
    lda: c_int,
    is_negdef: *mut c_int,
) -> c_int {
    if n <= 0 || a.is_null() || is_negdef.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let lda_val = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert A to Mat
    let mut mat_a: Mat<f64> = Mat::zeros(n_val, n_val);
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    match Ldlt::compute(mat_a.as_ref()) {
        Ok(ldlt) => {
            *is_negdef = if ldlt.is_negative_definite() { 1 } else { 0 };
            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::InvalidArg as c_int,
    }
}

// =============================================================================
// SLOGABSDET_LDLT - Log absolute determinant via LDLT (single precision)
// =============================================================================

/// Computes log(|det(A)|) and sign(det(A)) using LDL^T factorization.
///
/// For symmetric matrices, returns the log absolute determinant and sign.
/// This is more numerically stable than computing the determinant directly.
///
/// # Arguments
/// * `layout` - Matrix layout
/// * `n` - Dimension of matrix A
/// * `a` - Symmetric matrix A
/// * `lda` - Leading dimension of A
/// * `logabsdet` - Output: log(|det(A)|)
/// * `sign` - Output: sign of determinant (-1, 0, or 1)
///
/// # Safety
/// - `a` must point to a valid n × n symmetric matrix
/// - Output pointers must be valid
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_slogabsdet_ldlt(
    layout: OblasLayout,
    n: c_int,
    a: *const f32,
    lda: c_int,
    logabsdet: *mut f32,
    sign: *mut c_int,
) -> c_int {
    if n <= 0 || a.is_null() || logabsdet.is_null() || sign.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let lda_val = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert A to Mat
    let mut mat_a: Mat<f32> = Mat::zeros(n_val, n_val);
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    match Ldlt::compute(mat_a.as_ref()) {
        Ok(ldlt) => {
            let (log_abs, s) = ldlt.log_abs_determinant();
            *logabsdet = log_abs;
            *sign = s;
            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::InvalidArg as c_int,
    }
}

// =============================================================================
// DLOGABSDET_LDLT - Log absolute determinant via LDLT (double precision)
// =============================================================================

/// Computes log(|det(A)|) and sign(det(A)) using LDL^T factorization.
///
/// # Safety
/// - `a` must point to a valid n × n symmetric matrix
/// - Output pointers must be valid
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dlogabsdet_ldlt(
    layout: OblasLayout,
    n: c_int,
    a: *const f64,
    lda: c_int,
    logabsdet: *mut f64,
    sign: *mut c_int,
) -> c_int {
    if n <= 0 || a.is_null() || logabsdet.is_null() || sign.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let lda_val = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert A to Mat
    let mut mat_a: Mat<f64> = Mat::zeros(n_val, n_val);
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    match Ldlt::compute(mat_a.as_ref()) {
        Ok(ldlt) => {
            let (log_abs, s) = ldlt.log_abs_determinant();
            *logabsdet = log_abs;
            *sign = s;
            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::InvalidArg as c_int,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dsytrf_positive_definite() {
        // A = [[4, 2], [2, 5]] (symmetric positive definite)
        let mut a = [4.0f64, 2.0, 2.0, 5.0]; // column-major
        let mut d = [0.0f64; 2];

        unsafe {
            let ret = oblas_dsytrf(
                OblasLayout::ColMajor,
                b'L',
                2,
                a.as_mut_ptr(),
                2,
                d.as_mut_ptr(),
            );
            assert_eq!(ret, 0);

            // D should have positive values for positive definite
            assert!(d[0] > 0.0);
            assert!(d[1] > 0.0);
        }
    }

    #[test]
    fn test_dsytrf_indefinite() {
        // A = [[1, 2], [2, 1]] (symmetric indefinite)
        let mut a = [1.0f64, 2.0, 2.0, 1.0]; // column-major
        let mut d = [0.0f64; 2];

        unsafe {
            let ret = oblas_dsytrf(
                OblasLayout::ColMajor,
                b'L',
                2,
                a.as_mut_ptr(),
                2,
                d.as_mut_ptr(),
            );
            assert_eq!(ret, 0);

            // D should have both positive and negative values for indefinite
            // eigenvalues are 3 and -1
            let prod = d[0] * d[1];
            assert!(prod < 0.0); // Different signs
        }
    }

    #[test]
    fn test_ssytrf_basic() {
        let mut a = [4.0f32, 2.0, 2.0, 5.0]; // column-major
        let mut d = [0.0f32; 2];

        unsafe {
            let ret = oblas_ssytrf(
                OblasLayout::ColMajor,
                b'L',
                2,
                a.as_mut_ptr(),
                2,
                d.as_mut_ptr(),
            );
            assert_eq!(ret, 0);
            assert!(d[0] > 0.0);
            assert!(d[1] > 0.0);
        }
    }

    #[test]
    fn test_dsytrf_3x3() {
        // A = [[4, 2, 1], [2, 5, 2], [1, 2, 6]] (symmetric positive definite)
        let mut a = [4.0f64, 2.0, 1.0, 2.0, 5.0, 2.0, 1.0, 2.0, 6.0]; // column-major
        let mut d = [0.0f64; 3];

        unsafe {
            let ret = oblas_dsytrf(
                OblasLayout::ColMajor,
                b'L',
                3,
                a.as_mut_ptr(),
                3,
                d.as_mut_ptr(),
            );
            assert_eq!(ret, 0);

            // All D values should be positive
            for &val in &d {
                assert!(val > 0.0);
            }
        }
    }

    #[test]
    fn test_dinertia_positive_definite() {
        // A = [[4, 2], [2, 5]] (symmetric positive definite)
        // eigenvalues are 3 and 6, so (2, 0, 0)
        let a = [4.0f64, 2.0, 2.0, 5.0]; // column-major
        let mut n_pos = 0i32;
        let mut n_neg = 0i32;
        let mut n_zero = 0i32;

        unsafe {
            let ret = oblas_dinertia(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                &mut n_pos,
                &mut n_neg,
                &mut n_zero,
            );
            assert_eq!(ret, 0);
            assert_eq!(n_pos, 2); // Both eigenvalues positive
            assert_eq!(n_neg, 0);
            assert_eq!(n_zero, 0);
        }
    }

    #[test]
    fn test_dinertia_indefinite() {
        // A = [[1, 2], [2, 1]] (symmetric indefinite)
        // eigenvalues are 3 and -1, so (1, 1, 0)
        let a = [1.0f64, 2.0, 2.0, 1.0]; // column-major
        let mut n_pos = 0i32;
        let mut n_neg = 0i32;
        let mut n_zero = 0i32;

        unsafe {
            let ret = oblas_dinertia(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                &mut n_pos,
                &mut n_neg,
                &mut n_zero,
            );
            assert_eq!(ret, 0);
            assert_eq!(n_pos, 1); // One positive eigenvalue
            assert_eq!(n_neg, 1); // One negative eigenvalue
            assert_eq!(n_zero, 0);
        }
    }

    #[test]
    fn test_dinertia_negative_definite() {
        // A = [[-4, -2], [-2, -5]] (symmetric negative definite)
        let a = [-4.0f64, -2.0, -2.0, -5.0]; // column-major
        let mut n_pos = 0i32;
        let mut n_neg = 0i32;
        let mut n_zero = 0i32;

        unsafe {
            let ret = oblas_dinertia(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                &mut n_pos,
                &mut n_neg,
                &mut n_zero,
            );
            assert_eq!(ret, 0);
            assert_eq!(n_pos, 0);
            assert_eq!(n_neg, 2); // Both eigenvalues negative
            assert_eq!(n_zero, 0);
        }
    }

    #[test]
    fn test_sinertia_basic() {
        // A = [[4, 2], [2, 5]] (positive definite)
        let a = [4.0f32, 2.0, 2.0, 5.0]; // column-major
        let mut n_pos = 0i32;
        let mut n_neg = 0i32;
        let mut n_zero = 0i32;

        unsafe {
            let ret = oblas_sinertia(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                &mut n_pos,
                &mut n_neg,
                &mut n_zero,
            );
            assert_eq!(ret, 0);
            assert_eq!(n_pos, 2);
            assert_eq!(n_neg, 0);
            assert_eq!(n_zero, 0);
        }
    }

    #[test]
    fn test_disposdef_positive_definite() {
        // A = [[4, 2], [2, 5]] (positive definite)
        let a = [4.0f64, 2.0, 2.0, 5.0]; // column-major
        let mut is_posdef = 0i32;

        unsafe {
            let ret = oblas_disposdef(OblasLayout::ColMajor, 2, a.as_ptr(), 2, &mut is_posdef);
            assert_eq!(ret, 0);
            assert_eq!(is_posdef, 1); // Should be positive definite
        }
    }

    #[test]
    fn test_disposdef_indefinite() {
        // A = [[1, 2], [2, 1]] (indefinite)
        let a = [1.0f64, 2.0, 2.0, 1.0]; // column-major
        let mut is_posdef = 0i32;

        unsafe {
            let ret = oblas_disposdef(OblasLayout::ColMajor, 2, a.as_ptr(), 2, &mut is_posdef);
            assert_eq!(ret, 0);
            assert_eq!(is_posdef, 0); // Not positive definite
        }
    }

    #[test]
    fn test_sisposdef_basic() {
        // A = [[4, 2], [2, 5]] (positive definite)
        let a = [4.0f32, 2.0, 2.0, 5.0]; // column-major
        let mut is_posdef = 0i32;

        unsafe {
            let ret = oblas_sisposdef(OblasLayout::ColMajor, 2, a.as_ptr(), 2, &mut is_posdef);
            assert_eq!(ret, 0);
            assert_eq!(is_posdef, 1);
        }
    }

    #[test]
    fn test_dinertia_3x3() {
        // A = [[2, 1, 0], [1, -1, 1], [0, 1, 2]] (mixed eigenvalues)
        let a = [2.0f64, 1.0, 0.0, 1.0, -1.0, 1.0, 0.0, 1.0, 2.0]; // column-major
        let mut n_pos = 0i32;
        let mut n_neg = 0i32;
        let mut n_zero = 0i32;

        unsafe {
            let ret = oblas_dinertia(
                OblasLayout::ColMajor,
                3,
                a.as_ptr(),
                3,
                &mut n_pos,
                &mut n_neg,
                &mut n_zero,
            );
            assert_eq!(ret, 0);
            // Should have 2 positive and 1 negative eigenvalue
            assert_eq!(n_pos + n_neg + n_zero, 3);
        }
    }

    #[test]
    fn test_disnegdef_negative_definite() {
        // A = [[-4, -2], [-2, -5]] (negative definite)
        let a = [-4.0f64, -2.0, -2.0, -5.0]; // column-major
        let mut is_negdef = 0i32;

        unsafe {
            let ret = oblas_disnegdef(OblasLayout::ColMajor, 2, a.as_ptr(), 2, &mut is_negdef);
            assert_eq!(ret, 0);
            assert_eq!(is_negdef, 1); // Should be negative definite
        }
    }

    #[test]
    fn test_disnegdef_positive_definite() {
        // A = [[4, 2], [2, 5]] (positive definite, not negative definite)
        let a = [4.0f64, 2.0, 2.0, 5.0]; // column-major
        let mut is_negdef = 0i32;

        unsafe {
            let ret = oblas_disnegdef(OblasLayout::ColMajor, 2, a.as_ptr(), 2, &mut is_negdef);
            assert_eq!(ret, 0);
            assert_eq!(is_negdef, 0); // Not negative definite
        }
    }

    #[test]
    fn test_sisnegdef_negative_definite() {
        // A = [[-4, -2], [-2, -5]] (negative definite)
        let a = [-4.0f32, -2.0, -2.0, -5.0]; // column-major
        let mut is_negdef = 0i32;

        unsafe {
            let ret = oblas_sisnegdef(OblasLayout::ColMajor, 2, a.as_ptr(), 2, &mut is_negdef);
            assert_eq!(ret, 0);
            assert_eq!(is_negdef, 1);
        }
    }

    #[test]
    fn test_dlogabsdet_ldlt_positive_definite() {
        // A = [[4, 2], [2, 5]] (positive definite)
        // det = 4*5 - 2*2 = 16, log(16) ≈ 2.77, sign = 1
        let a = [4.0f64, 2.0, 2.0, 5.0]; // column-major
        let mut logabsdet = 0.0f64;
        let mut sign = 0i32;

        unsafe {
            let ret = oblas_dlogabsdet_ldlt(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                &mut logabsdet,
                &mut sign,
            );
            assert_eq!(ret, 0);
            assert!((logabsdet - 16.0f64.ln()).abs() < 1e-10);
            assert_eq!(sign, 1); // Positive determinant
        }
    }

    #[test]
    fn test_dlogabsdet_ldlt_negative_definite() {
        // A = [[-4, -2], [-2, -5]] (negative definite)
        // det = (-4)*(-5) - (-2)*(-2) = 20 - 4 = 16, log(16), sign = 1
        let a = [-4.0f64, -2.0, -2.0, -5.0]; // column-major
        let mut logabsdet = 0.0f64;
        let mut sign = 0i32;

        unsafe {
            let ret = oblas_dlogabsdet_ldlt(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                &mut logabsdet,
                &mut sign,
            );
            assert_eq!(ret, 0);
            assert!((logabsdet - 16.0f64.ln()).abs() < 1e-10);
            assert_eq!(sign, 1); // Positive determinant (both eigenvalues negative, product positive)
        }
    }

    #[test]
    fn test_dlogabsdet_ldlt_indefinite() {
        // A = [[1, 2], [2, 1]] (indefinite)
        // det = 1 - 4 = -3, log(3), sign = -1
        let a = [1.0f64, 2.0, 2.0, 1.0]; // column-major
        let mut logabsdet = 0.0f64;
        let mut sign = 0i32;

        unsafe {
            let ret = oblas_dlogabsdet_ldlt(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                &mut logabsdet,
                &mut sign,
            );
            assert_eq!(ret, 0);
            assert!((logabsdet - 3.0f64.ln()).abs() < 1e-10);
            assert_eq!(sign, -1); // Negative determinant
        }
    }

    #[test]
    fn test_slogabsdet_ldlt_basic() {
        // A = [[4, 2], [2, 5]] (positive definite)
        let a = [4.0f32, 2.0, 2.0, 5.0]; // column-major
        let mut logabsdet = 0.0f32;
        let mut sign = 0i32;

        unsafe {
            let ret = oblas_slogabsdet_ldlt(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                &mut logabsdet,
                &mut sign,
            );
            assert_eq!(ret, 0);
            assert!((logabsdet - 16.0f32.ln()).abs() < 1e-5);
            assert_eq!(sign, 1);
        }
    }
}
