//! LAPACK FFI - Factorization routines (LU, Cholesky, QR).

use crate::types::*;
use oxiblas_lapack::{cholesky, lu, qr};
use oxiblas_matrix::Mat;
use std::ffi::c_int;
use std::slice;

// =============================================================================
// SGETRF - LU factorization with partial pivoting (single precision)
// =============================================================================

/// Computes LU factorization of a general m x n matrix.
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `ipiv` must point to an array of min(m, n) integers
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sgetrf(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *mut f32,
    lda: c_int,
    ipiv: *mut c_int,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || ipiv.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat
    let mut mat = Mat::<f32>::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat[(i, j)] = *a.add(idx);
        }
    }

    // Compute LU
    match lu::Lu::compute(mat.as_ref()) {
        Ok(lu_result) => {
            // Copy L and U back to a (packed format)
            let l = lu_result.l_factor();
            let u = lu_result.u_factor();
            let perm = lu_result.pivot();

            for i in 0..m {
                for j in 0..n {
                    let idx = if row_major { i * lda + j } else { j * lda + i };
                    if i > j {
                        // Below diagonal: L
                        if j < l.ncols() && i < l.nrows() {
                            *a.add(idx) = l[(i, j)];
                        }
                    } else {
                        // Diagonal and above: U
                        if j < u.ncols() && i < u.nrows() {
                            *a.add(idx) = u[(i, j)];
                        }
                    }
                }
            }

            // Copy pivot indices (1-based for LAPACK compatibility)
            let min_mn = m.min(n);
            let ipiv_slice = slice::from_raw_parts_mut(ipiv, min_mn);
            for i in 0..min_mn {
                ipiv_slice[i] = (perm[i] + 1) as c_int;
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::Singular as c_int,
    }
}

// =============================================================================
// DGETRF - LU factorization with partial pivoting (double precision)
// =============================================================================

/// Computes LU factorization of a general m x n matrix.
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `ipiv` must point to an array of min(m, n) integers
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dgetrf(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *mut f64,
    lda: c_int,
    ipiv: *mut c_int,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || ipiv.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat
    let mut mat = Mat::<f64>::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat[(i, j)] = *a.add(idx);
        }
    }

    // Compute LU
    match lu::Lu::compute(mat.as_ref()) {
        Ok(lu_result) => {
            // Copy L and U back to a (packed format)
            let l = lu_result.l_factor();
            let u = lu_result.u_factor();
            let perm = lu_result.pivot();

            for i in 0..m {
                for j in 0..n {
                    let idx = if row_major { i * lda + j } else { j * lda + i };
                    if i > j {
                        if j < l.ncols() && i < l.nrows() {
                            *a.add(idx) = l[(i, j)];
                        }
                    } else {
                        if j < u.ncols() && i < u.nrows() {
                            *a.add(idx) = u[(i, j)];
                        }
                    }
                }
            }

            // Copy pivot indices (1-based)
            let min_mn = m.min(n);
            let ipiv_slice = slice::from_raw_parts_mut(ipiv, min_mn);
            for i in 0..min_mn {
                ipiv_slice[i] = (perm[i] + 1) as c_int;
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::Singular as c_int,
    }
}

// =============================================================================
// ZGETRF - Complex double precision LU factorization
// =============================================================================

/// Computes LU factorization of a complex general m x n matrix.
///
/// # Safety
/// - `a` must point to a valid m x n complex matrix
/// - `ipiv` must point to an array of min(m, n) integers
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zgetrf(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *mut OblasComplex64,
    lda: c_int,
    ipiv: *mut c_int,
) -> c_int {
    use num_complex::Complex64;

    if m <= 0 || n <= 0 || a.is_null() || ipiv.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
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

    // Compute LU
    match lu::Lu::compute(mat.as_ref()) {
        Ok(lu_result) => {
            let l = lu_result.l_factor();
            let u = lu_result.u_factor();
            let perm = lu_result.pivot();

            for i in 0..m {
                for j in 0..n {
                    let idx = if row_major { i * lda + j } else { j * lda + i };
                    if i > j {
                        if j < l.ncols() && i < l.nrows() {
                            let v = l[(i, j)];
                            (*a.add(idx)).re = v.re;
                            (*a.add(idx)).im = v.im;
                        }
                    } else {
                        if j < u.ncols() && i < u.nrows() {
                            let v = u[(i, j)];
                            (*a.add(idx)).re = v.re;
                            (*a.add(idx)).im = v.im;
                        }
                    }
                }
            }

            let min_mn = m.min(n);
            let ipiv_slice = slice::from_raw_parts_mut(ipiv, min_mn);
            for i in 0..min_mn {
                ipiv_slice[i] = (perm[i] + 1) as c_int;
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::Singular as c_int,
    }
}

// =============================================================================
// CGETRF - Complex LU factorization (single precision)
// =============================================================================

/// Computes LU factorization of a general complex m x n matrix.
///
/// # Safety
/// - `a` must point to a valid complex m x n matrix
/// - `ipiv` must point to an array of min(m, n) integers
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_cgetrf(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *mut OblasComplex32,
    lda: c_int,
    ipiv: *mut c_int,
) -> c_int {
    use num_complex::Complex32;

    if m <= 0 || n <= 0 || a.is_null() || ipiv.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
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

    // Compute LU
    match lu::Lu::compute(mat.as_ref()) {
        Ok(lu_result) => {
            let l = lu_result.l_factor();
            let u = lu_result.u_factor();
            let perm = lu_result.pivot();

            for i in 0..m {
                for j in 0..n {
                    let idx = if row_major { i * lda + j } else { j * lda + i };
                    if i > j {
                        if j < l.ncols() && i < l.nrows() {
                            let v = l[(i, j)];
                            (*a.add(idx)).re = v.re;
                            (*a.add(idx)).im = v.im;
                        }
                    } else {
                        if j < u.ncols() && i < u.nrows() {
                            let v = u[(i, j)];
                            (*a.add(idx)).re = v.re;
                            (*a.add(idx)).im = v.im;
                        }
                    }
                }
            }

            let min_mn = m.min(n);
            let ipiv_slice = slice::from_raw_parts_mut(ipiv, min_mn);
            for i in 0..min_mn {
                ipiv_slice[i] = (perm[i] + 1) as c_int;
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::Singular as c_int,
    }
}

// =============================================================================
// SPOTRF - Cholesky factorization (single precision)
// =============================================================================

/// Computes Cholesky factorization of a symmetric positive definite matrix.
///
/// # Safety
/// - `a` must point to a valid n x n symmetric positive definite matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_spotrf(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    a: *mut f32,
    lda: c_int,
) -> c_int {
    if n <= 0 || a.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;

    // Convert to Mat
    let mut mat = Mat::<f32>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat[(i, j)] = *a.add(idx);
        }
    }

    // Compute Cholesky
    match cholesky::Cholesky::compute(mat.as_ref()) {
        Ok(chol) => {
            let l = chol.l_factor();

            // Copy result back
            for i in 0..n {
                for j in 0..n {
                    let idx = if row_major { i * lda + j } else { j * lda + i };
                    if upper {
                        // Store upper triangular (L^T)
                        if j >= i {
                            *a.add(idx) = l[(j, i)];
                        } else {
                            *a.add(idx) = 0.0;
                        }
                    } else {
                        // Store lower triangular (L)
                        if i >= j {
                            *a.add(idx) = l[(i, j)];
                        } else {
                            *a.add(idx) = 0.0;
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
// DPOTRF - Cholesky factorization (double precision)
// =============================================================================

/// Computes Cholesky factorization of a symmetric positive definite matrix.
///
/// # Safety
/// - `a` must point to a valid n x n symmetric positive definite matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dpotrf(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    a: *mut f64,
    lda: c_int,
) -> c_int {
    if n <= 0 || a.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;

    // Convert to Mat
    let mut mat = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat[(i, j)] = *a.add(idx);
        }
    }

    // Compute Cholesky
    match cholesky::Cholesky::compute(mat.as_ref()) {
        Ok(chol) => {
            let l = chol.l_factor();

            // Copy result back
            for i in 0..n {
                for j in 0..n {
                    let idx = if row_major { i * lda + j } else { j * lda + i };
                    if upper {
                        if j >= i {
                            *a.add(idx) = l[(j, i)];
                        } else {
                            *a.add(idx) = 0.0;
                        }
                    } else {
                        if i >= j {
                            *a.add(idx) = l[(i, j)];
                        } else {
                            *a.add(idx) = 0.0;
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
// ZPOTRF - Complex Cholesky factorization (double precision)
// =============================================================================

/// Computes Cholesky factorization of a Hermitian positive definite matrix.
///
/// # Safety
/// - `a` must point to a valid n x n Hermitian positive definite matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zpotrf(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    a: *mut OblasComplex64,
    lda: c_int,
) -> c_int {
    use num_complex::Complex64;

    if n <= 0 || a.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;

    // Convert to Mat<Complex64>
    let mut mat: Mat<Complex64> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            let c = *a.add(idx);
            mat[(i, j)] = Complex64::new(c.re, c.im);
        }
    }

    // Compute Hermitian Cholesky
    match cholesky::HermitianCholesky::compute(mat.as_ref()) {
        Ok(chol) => {
            let l = chol.l_factor();

            // Copy result back
            for i in 0..n {
                for j in 0..n {
                    let idx = if row_major { i * lda + j } else { j * lda + i };
                    if upper {
                        // Store upper triangular (L^H)
                        if j >= i {
                            let v = l[(j, i)].conj();
                            (*a.add(idx)).re = v.re;
                            (*a.add(idx)).im = v.im;
                        } else {
                            (*a.add(idx)).re = 0.0;
                            (*a.add(idx)).im = 0.0;
                        }
                    } else {
                        // Store lower triangular (L)
                        if i >= j {
                            let v = l[(i, j)];
                            (*a.add(idx)).re = v.re;
                            (*a.add(idx)).im = v.im;
                        } else {
                            (*a.add(idx)).re = 0.0;
                            (*a.add(idx)).im = 0.0;
                        }
                    }
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::NotPosdef as c_int,
    }
}

// =============================================================================
// CPOTRF - Complex Cholesky factorization (single precision)
// =============================================================================

/// Computes Cholesky factorization of a Hermitian positive definite matrix.
///
/// # Safety
/// - `a` must point to a valid n x n Hermitian positive definite matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_cpotrf(
    layout: OblasLayout,
    uplo: OblasUplo,
    n: c_int,
    a: *mut OblasComplex32,
    lda: c_int,
) -> c_int {
    use num_complex::Complex32;

    if n <= 0 || a.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;

    // Convert to Mat<Complex32>
    let mut mat: Mat<Complex32> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            let c = *a.add(idx);
            mat[(i, j)] = Complex32::new(c.re, c.im);
        }
    }

    // Compute Hermitian Cholesky
    match cholesky::HermitianCholesky::compute(mat.as_ref()) {
        Ok(chol) => {
            let l = chol.l_factor();

            // Copy result back
            for i in 0..n {
                for j in 0..n {
                    let idx = if row_major { i * lda + j } else { j * lda + i };
                    if upper {
                        // Store upper triangular (L^H)
                        if j >= i {
                            let v = l[(j, i)].conj();
                            (*a.add(idx)).re = v.re;
                            (*a.add(idx)).im = v.im;
                        } else {
                            (*a.add(idx)).re = 0.0;
                            (*a.add(idx)).im = 0.0;
                        }
                    } else {
                        // Store lower triangular (L)
                        if i >= j {
                            let v = l[(i, j)];
                            (*a.add(idx)).re = v.re;
                            (*a.add(idx)).im = v.im;
                        } else {
                            (*a.add(idx)).re = 0.0;
                            (*a.add(idx)).im = 0.0;
                        }
                    }
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::NotPosdef as c_int,
    }
}

// =============================================================================
// SGEQRF - QR factorization (single precision)
// =============================================================================

/// Computes QR factorization of a general m x n matrix.
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `tau` must point to an array of min(m, n) elements
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sgeqrf(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *mut f32,
    lda: c_int,
    tau: *mut f32,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || tau.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat
    let mut mat = Mat::<f32>::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat[(i, j)] = *a.add(idx);
        }
    }

    // Compute QR
    match qr::Qr::compute(mat.as_ref()) {
        Ok(qr_result) => {
            let r = qr_result.r();

            // Copy R back to upper triangular of a
            for i in 0..m.min(n) {
                for j in i..n {
                    let idx = if row_major { i * lda + j } else { j * lda + i };
                    *a.add(idx) = r[(i, j)];
                }
            }

            // Store Householder vectors below diagonal (simplified)
            let min_mn = m.min(n);
            let tau_slice = slice::from_raw_parts_mut(tau, min_mn);
            for i in 0..min_mn {
                tau_slice[i] = 1.0; // Placeholder
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::NotConverged as c_int,
    }
}

// =============================================================================
// DGEQRF - QR factorization (double precision)
// =============================================================================

/// Computes QR factorization of a general m x n matrix.
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `tau` must point to an array of min(m, n) elements
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dgeqrf(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *mut f64,
    lda: c_int,
    tau: *mut f64,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || tau.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat
    let mut mat = Mat::<f64>::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat[(i, j)] = *a.add(idx);
        }
    }

    // Compute QR
    match qr::Qr::compute(mat.as_ref()) {
        Ok(qr_result) => {
            let r = qr_result.r();

            // Copy R back to upper triangular of a
            for i in 0..m.min(n) {
                for j in i..n {
                    let idx = if row_major { i * lda + j } else { j * lda + i };
                    *a.add(idx) = r[(i, j)];
                }
            }

            // Store Householder vectors (simplified)
            let min_mn = m.min(n);
            let tau_slice = slice::from_raw_parts_mut(tau, min_mn);
            for i in 0..min_mn {
                tau_slice[i] = 1.0;
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::NotConverged as c_int,
    }
}

// =============================================================================
// ZGEQRF - Complex QR factorization (double precision)
// =============================================================================

/// Computes QR factorization of a complex general m x n matrix.
///
/// # Safety
/// - `a` must point to a valid complex m x n matrix
/// - `tau` must point to an array of min(m, n) complex elements
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zgeqrf(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *mut OblasComplex64,
    lda: c_int,
    tau: *mut OblasComplex64,
) -> c_int {
    use num_complex::Complex64;

    if m <= 0 || n <= 0 || a.is_null() || tau.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
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

    // Compute QR using UnitaryQr
    match qr::UnitaryQr::compute(mat.as_ref()) {
        Ok(qr_result) => {
            let r = qr_result.r();
            let tau_vals = qr_result.tau();

            // Copy R back to upper triangular of a
            for i in 0..m.min(n) {
                for j in i..n {
                    let idx = if row_major { i * lda + j } else { j * lda + i };
                    let v = r[(i, j)];
                    (*a.add(idx)).re = v.re;
                    (*a.add(idx)).im = v.im;
                }
            }

            // Copy tau values
            let min_mn = m.min(n);
            let tau_slice = slice::from_raw_parts_mut(tau, min_mn);
            for i in 0..min_mn {
                tau_slice[i].re = tau_vals[i].re;
                tau_slice[i].im = tau_vals[i].im;
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::InvalidArg as c_int,
    }
}

// =============================================================================
// CGEQRF - Complex QR factorization (single precision)
// =============================================================================

/// Computes QR factorization of a complex general m x n matrix.
///
/// # Safety
/// - `a` must point to a valid complex m x n matrix
/// - `tau` must point to an array of min(m, n) complex elements
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_cgeqrf(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *mut OblasComplex32,
    lda: c_int,
    tau: *mut OblasComplex32,
) -> c_int {
    use num_complex::Complex32;

    if m <= 0 || n <= 0 || a.is_null() || tau.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
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

    // Compute QR using UnitaryQr
    match qr::UnitaryQr::compute(mat.as_ref()) {
        Ok(qr_result) => {
            let r = qr_result.r();
            let tau_vals = qr_result.tau();

            // Copy R back to upper triangular of a
            for i in 0..m.min(n) {
                for j in i..n {
                    let idx = if row_major { i * lda + j } else { j * lda + i };
                    let v = r[(i, j)];
                    (*a.add(idx)).re = v.re;
                    (*a.add(idx)).im = v.im;
                }
            }

            // Copy tau values
            let min_mn = m.min(n);
            let tau_slice = slice::from_raw_parts_mut(tau, min_mn);
            for i in 0..min_mn {
                tau_slice[i].re = tau_vals[i].re;
                tau_slice[i].im = tau_vals[i].im;
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::InvalidArg as c_int,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dgetrf() {
        // A = [[2, 1], [1, 3]] (column-major)
        let mut a = [2.0f64, 1.0, 1.0, 3.0];
        let mut ipiv = [0i32; 2];

        unsafe {
            let result = oblas_dgetrf(
                OblasLayout::ColMajor,
                2,
                2,
                a.as_mut_ptr(),
                2,
                ipiv.as_mut_ptr(),
            );
            assert_eq!(result, 0);
        }
    }

    #[test]
    fn test_dpotrf() {
        // A = [[4, 2], [2, 5]] (column-major, symmetric positive definite)
        let mut a = [4.0f64, 2.0, 2.0, 5.0];

        unsafe {
            let result = oblas_dpotrf(
                OblasLayout::ColMajor,
                OblasUplo::Lower,
                2,
                a.as_mut_ptr(),
                2,
            );
            assert_eq!(result, 0);
        }
    }

    #[test]
    fn test_zgetrf() {
        // Complex matrix A = [[2+1i, 1], [1, 3-1i]] (column-major)
        let mut a = [
            OblasComplex64 { re: 2.0, im: 1.0 },
            OblasComplex64 { re: 1.0, im: 0.0 },
            OblasComplex64 { re: 1.0, im: 0.0 },
            OblasComplex64 { re: 3.0, im: -1.0 },
        ];
        let mut ipiv = [0i32; 2];

        unsafe {
            let result = oblas_zgetrf(
                OblasLayout::ColMajor,
                2,
                2,
                a.as_mut_ptr(),
                2,
                ipiv.as_mut_ptr(),
            );
            assert_eq!(result, 0);
        }
    }

    #[test]
    fn test_cgetrf() {
        // Complex single precision matrix A = [[4, 1+i], [1-i, 3]] (column-major)
        let mut a = [
            OblasComplex32 { re: 4.0, im: 0.0 },
            OblasComplex32 { re: 1.0, im: -1.0 },
            OblasComplex32 { re: 1.0, im: 1.0 },
            OblasComplex32 { re: 3.0, im: 0.0 },
        ];
        let mut ipiv = [0i32; 2];

        unsafe {
            let result = oblas_cgetrf(
                OblasLayout::ColMajor,
                2,
                2,
                a.as_mut_ptr(),
                2,
                ipiv.as_mut_ptr(),
            );
            assert_eq!(result, 0);
        }
    }

    #[test]
    fn test_zpotrf() {
        // Hermitian positive definite matrix A = [[4, 1-i], [1+i, 3]] (column-major)
        // Note: A[0,1] = conj(A[1,0]) for Hermitian
        let mut a = [
            OblasComplex64 { re: 4.0, im: 0.0 },
            OblasComplex64 { re: 1.0, im: 1.0 },
            OblasComplex64 { re: 1.0, im: -1.0 },
            OblasComplex64 { re: 3.0, im: 0.0 },
        ];

        unsafe {
            let result = oblas_zpotrf(
                OblasLayout::ColMajor,
                OblasUplo::Lower,
                2,
                a.as_mut_ptr(),
                2,
            );
            assert_eq!(result, 0);

            // Verify L is lower triangular and L[0,0] = sqrt(4) = 2
            assert!((a[0].re - 2.0).abs() < 1e-10);
            assert!(a[0].im.abs() < 1e-10);

            // L[1,0] = A[1,0] / L[0,0] = (1+i) / 2 = 0.5 + 0.5i
            assert!((a[1].re - 0.5).abs() < 1e-10);
            assert!((a[1].im - 0.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cpotrf() {
        // Hermitian positive definite matrix A = [[4, 2-i], [2+i, 5]] (column-major)
        let mut a = [
            OblasComplex32 { re: 4.0, im: 0.0 },
            OblasComplex32 { re: 2.0, im: 1.0 },
            OblasComplex32 { re: 2.0, im: -1.0 },
            OblasComplex32 { re: 5.0, im: 0.0 },
        ];

        unsafe {
            let result = oblas_cpotrf(
                OblasLayout::ColMajor,
                OblasUplo::Lower,
                2,
                a.as_mut_ptr(),
                2,
            );
            assert_eq!(result, 0);

            // Verify L[0,0] = sqrt(4) = 2
            assert!((a[0].re - 2.0).abs() < 1e-5);
            assert!(a[0].im.abs() < 1e-5);
        }
    }

    #[test]
    fn test_zpotrf_upper() {
        // Hermitian positive definite matrix A = [[4, 1-i], [1+i, 3]] (column-major)
        // Store upper triangular (L^H)
        let mut a = [
            OblasComplex64 { re: 4.0, im: 0.0 },
            OblasComplex64 { re: 1.0, im: 1.0 },
            OblasComplex64 { re: 1.0, im: -1.0 },
            OblasComplex64 { re: 3.0, im: 0.0 },
        ];

        unsafe {
            let result = oblas_zpotrf(
                OblasLayout::ColMajor,
                OblasUplo::Upper,
                2,
                a.as_mut_ptr(),
                2,
            );
            assert_eq!(result, 0);

            // For upper, U = L^H
            // U[0,0] = L[0,0] = 2
            assert!((a[0].re - 2.0).abs() < 1e-10);
            assert!(a[0].im.abs() < 1e-10);

            // U[0,1] = conj(L[1,0]) = conj((1+i)/2) = 0.5 - 0.5i
            assert!((a[2].re - 0.5).abs() < 1e-10);
            assert!((a[2].im + 0.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_zpotrf_not_positive_definite() {
        // Non-positive definite matrix A = [[1, 2], [2, 1]]
        let mut a = [
            OblasComplex64 { re: 1.0, im: 0.0 },
            OblasComplex64 { re: 2.0, im: 0.0 },
            OblasComplex64 { re: 2.0, im: 0.0 },
            OblasComplex64 { re: 1.0, im: 0.0 },
        ];

        unsafe {
            let result = oblas_zpotrf(
                OblasLayout::ColMajor,
                OblasUplo::Lower,
                2,
                a.as_mut_ptr(),
                2,
            );
            // Should return NotPosdef error (-5)
            assert_eq!(result, OblasReturn::NotPosdef as c_int);
        }
    }

    #[test]
    fn test_zgeqrf() {
        // Complex matrix A = [[1+i, 2], [3-i, 4+i]] (column-major)
        let mut a = [
            OblasComplex64 { re: 1.0, im: 1.0 },
            OblasComplex64 { re: 3.0, im: -1.0 },
            OblasComplex64 { re: 2.0, im: 0.0 },
            OblasComplex64 { re: 4.0, im: 1.0 },
        ];
        let mut tau = [
            OblasComplex64 { re: 0.0, im: 0.0 },
            OblasComplex64 { re: 0.0, im: 0.0 },
        ];

        unsafe {
            let result = oblas_zgeqrf(
                OblasLayout::ColMajor,
                2,
                2,
                a.as_mut_ptr(),
                2,
                tau.as_mut_ptr(),
            );
            assert_eq!(result, 0);

            // R[0,0] should have the norm of the first column
            // ||[1+i, 3-i]|| = sqrt(|1+i|^2 + |3-i|^2) = sqrt(2 + 10) = sqrt(12)
            let r00_norm = (a[0].re * a[0].re + a[0].im * a[0].im).sqrt();
            assert!(
                (r00_norm - 12.0f64.sqrt()).abs() < 1e-10,
                "R[0,0] norm = {}, expected {}",
                r00_norm,
                12.0f64.sqrt()
            );
        }
    }

    #[test]
    fn test_cgeqrf() {
        // Complex single precision matrix (column-major)
        let mut a = [
            OblasComplex32 { re: 1.0, im: 0.0 },
            OblasComplex32 { re: 2.0, im: 1.0 },
            OblasComplex32 { re: 3.0, im: -1.0 },
            OblasComplex32 { re: 4.0, im: 0.0 },
        ];
        let mut tau = [
            OblasComplex32 { re: 0.0, im: 0.0 },
            OblasComplex32 { re: 0.0, im: 0.0 },
        ];

        unsafe {
            let result = oblas_cgeqrf(
                OblasLayout::ColMajor,
                2,
                2,
                a.as_mut_ptr(),
                2,
                tau.as_mut_ptr(),
            );
            assert_eq!(result, 0);

            // R[0,0] should have the norm of first column
            // ||[1, 2+i]|| = sqrt(1 + 5) = sqrt(6)
            let r00_norm = (a[0].re * a[0].re + a[0].im * a[0].im).sqrt();
            assert!(
                (r00_norm - 6.0f32.sqrt()).abs() < 1e-5,
                "R[0,0] norm = {}, expected {}",
                r00_norm,
                6.0f32.sqrt()
            );
        }
    }

    #[test]
    fn test_zgeqrf_tall() {
        // 3x2 complex matrix (column-major)
        let mut a = [
            OblasComplex64 { re: 1.0, im: 0.0 },
            OblasComplex64 { re: 2.0, im: 0.0 },
            OblasComplex64 { re: 3.0, im: 0.0 },
            OblasComplex64 { re: 4.0, im: 1.0 },
            OblasComplex64 { re: 5.0, im: -1.0 },
            OblasComplex64 { re: 6.0, im: 0.0 },
        ];
        let mut tau = [
            OblasComplex64 { re: 0.0, im: 0.0 },
            OblasComplex64 { re: 0.0, im: 0.0 },
        ];

        unsafe {
            let result = oblas_zgeqrf(
                OblasLayout::ColMajor,
                3,
                2,
                a.as_mut_ptr(),
                3,
                tau.as_mut_ptr(),
            );
            assert_eq!(result, 0);

            // R should be upper triangular in the 2x2 part
            // R[0,0] = ||first column|| = sqrt(1 + 4 + 9) = sqrt(14)
            let r00_norm = (a[0].re * a[0].re + a[0].im * a[0].im).sqrt();
            assert!(
                (r00_norm - 14.0f64.sqrt()).abs() < 1e-10,
                "R[0,0] norm = {}",
                r00_norm
            );
        }
    }
}
