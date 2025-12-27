//! LAPACK FFI - Eigenvalue decomposition routines.

use crate::types::*;
use num_complex::{Complex32, Complex64};
use oxiblas_lapack::evd;
use oxiblas_matrix::Mat;
use std::ffi::c_int;
use std::slice;

// =============================================================================
// SSYEV - Symmetric eigenvalue decomposition (single precision)
// =============================================================================

/// Computes eigenvalues and optionally eigenvectors of a symmetric matrix.
///
/// # Safety
/// - `a` must point to a valid n x n symmetric matrix
/// - `w` must point to an array of n elements for eigenvalues
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ssyev(
    layout: OblasLayout,
    jobz: u8, // 'N' = eigenvalues only, 'V' = eigenvalues and eigenvectors
    uplo: OblasUplo,
    n: c_int,
    a: *mut f32,
    lda: c_int,
    w: *mut f32,
) -> c_int {
    if n <= 0 || a.is_null() || w.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let compute_eigenvectors = jobz == b'V';

    // Convert to full symmetric Mat
    let mut mat: Mat<f32> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat[(i, j)] = *a.add(idx);
        }
    }

    // Symmetrize the matrix
    for i in 0..n {
        for j in (i + 1)..n {
            if upper {
                mat[(j, i)] = mat[(i, j)];
            } else {
                mat[(i, j)] = mat[(j, i)];
            }
        }
    }

    // Compute symmetric eigenvalue decomposition
    match evd::SymmetricEvd::compute(mat.as_ref()) {
        Ok(evd_result) => {
            let eigenvalues = evd_result.eigenvalues();
            let w_slice = slice::from_raw_parts_mut(w, n);
            for i in 0..n {
                w_slice[i] = eigenvalues[i];
            }

            // Copy eigenvectors back to a if requested
            if compute_eigenvectors {
                let eigenvectors = evd_result.eigenvectors();
                for i in 0..n {
                    for j in 0..n {
                        let idx = if row_major { i * lda + j } else { j * lda + i };
                        *a.add(idx) = eigenvectors[(i, j)];
                    }
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::NotConverged as c_int,
    }
}

// =============================================================================
// DSYEV - Symmetric eigenvalue decomposition (double precision)
// =============================================================================

/// Computes eigenvalues and optionally eigenvectors of a symmetric matrix.
///
/// # Safety
/// - `a` must point to a valid n x n symmetric matrix
/// - `w` must point to an array of n elements for eigenvalues
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dsyev(
    layout: OblasLayout,
    jobz: u8, // 'N' = eigenvalues only, 'V' = eigenvalues and eigenvectors
    uplo: OblasUplo,
    n: c_int,
    a: *mut f64,
    lda: c_int,
    w: *mut f64,
) -> c_int {
    if n <= 0 || a.is_null() || w.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let compute_eigenvectors = jobz == b'V';

    // Convert to full symmetric Mat
    let mut mat: Mat<f64> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat[(i, j)] = *a.add(idx);
        }
    }

    // Symmetrize the matrix
    for i in 0..n {
        for j in (i + 1)..n {
            if upper {
                mat[(j, i)] = mat[(i, j)];
            } else {
                mat[(i, j)] = mat[(j, i)];
            }
        }
    }

    // Compute symmetric eigenvalue decomposition
    match evd::SymmetricEvd::compute(mat.as_ref()) {
        Ok(evd_result) => {
            let eigenvalues = evd_result.eigenvalues();
            let w_slice = slice::from_raw_parts_mut(w, n);
            for i in 0..n {
                w_slice[i] = eigenvalues[i];
            }

            // Copy eigenvectors back to a if requested
            if compute_eigenvectors {
                let eigenvectors = evd_result.eigenvectors();
                for i in 0..n {
                    for j in 0..n {
                        let idx = if row_major { i * lda + j } else { j * lda + i };
                        *a.add(idx) = eigenvectors[(i, j)];
                    }
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::NotConverged as c_int,
    }
}

// =============================================================================
// SGEEV - General eigenvalue decomposition (single precision)
// =============================================================================

/// Computes eigenvalues and optionally eigenvectors of a general matrix.
///
/// # Safety
/// - `a` must point to a valid n x n matrix
/// - `wr` and `wi` must point to arrays of n elements for real and imaginary parts
/// - `vl` and `vr` must point to n x n matrices (or NULL if not needed)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sgeev(
    layout: OblasLayout,
    jobvl: u8, // 'N' = don't compute left eigenvectors, 'V' = compute
    jobvr: u8, // 'N' = don't compute right eigenvectors, 'V' = compute
    n: c_int,
    a: *mut f32,
    lda: c_int,
    wr: *mut f32, // Real parts of eigenvalues
    wi: *mut f32, // Imaginary parts of eigenvalues
    vl: *mut f32, // Left eigenvectors (n x n)
    ldvl: c_int,
    vr: *mut f32, // Right eigenvectors (n x n)
    ldvr: c_int,
) -> c_int {
    if n <= 0 || a.is_null() || wr.is_null() || wi.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let lda = lda as usize;
    let ldvl = ldvl as usize;
    let ldvr = ldvr as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let compute_vl = jobvl == b'V';
    let compute_vr = jobvr == b'V';

    // Convert to Mat
    let mut mat: Mat<f32> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat[(i, j)] = *a.add(idx);
        }
    }

    // Use compute_full if left eigenvectors are requested, otherwise use compute
    let evd_result = if compute_vl {
        evd::GeneralEvd::compute_full(mat.as_ref())
    } else {
        evd::GeneralEvd::compute(mat.as_ref())
    };

    match evd_result {
        Ok(evd_result) => {
            let eigenvalues = evd_result.eigenvalues();
            let wr_slice = slice::from_raw_parts_mut(wr, n);
            let wi_slice = slice::from_raw_parts_mut(wi, n);

            for i in 0..n {
                wr_slice[i] = eigenvalues[i].real;
                wi_slice[i] = eigenvalues[i].imag;
            }

            // Copy left eigenvectors if requested
            if compute_vl && !vl.is_null() {
                if let Some(left_eigenvectors) = evd_result.left_eigenvectors_real() {
                    for i in 0..n {
                        for j in 0..n {
                            let idx = if row_major {
                                i * ldvl + j
                            } else {
                                j * ldvl + i
                            };
                            *vl.add(idx) = left_eigenvectors[(i, j)];
                        }
                    }
                }
            }

            // Copy right eigenvectors if requested
            if compute_vr && !vr.is_null() {
                if let Some(eigenvectors) = evd_result.eigenvectors_real() {
                    for i in 0..n {
                        for j in 0..n {
                            let idx = if row_major {
                                i * ldvr + j
                            } else {
                                j * ldvr + i
                            };
                            *vr.add(idx) = eigenvectors[(i, j)];
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
// DGEEV - General eigenvalue decomposition (double precision)
// =============================================================================

/// Computes eigenvalues and optionally eigenvectors of a general matrix.
///
/// # Safety
/// - `a` must point to a valid n x n matrix
/// - `wr` and `wi` must point to arrays of n elements for real and imaginary parts
/// - `vl` and `vr` must point to n x n matrices (or NULL if not needed)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dgeev(
    layout: OblasLayout,
    jobvl: u8, // 'N' = don't compute left eigenvectors, 'V' = compute
    jobvr: u8, // 'N' = don't compute right eigenvectors, 'V' = compute
    n: c_int,
    a: *mut f64,
    lda: c_int,
    wr: *mut f64, // Real parts of eigenvalues
    wi: *mut f64, // Imaginary parts of eigenvalues
    vl: *mut f64, // Left eigenvectors (n x n)
    ldvl: c_int,
    vr: *mut f64, // Right eigenvectors (n x n)
    ldvr: c_int,
) -> c_int {
    if n <= 0 || a.is_null() || wr.is_null() || wi.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let lda = lda as usize;
    let ldvl = ldvl as usize;
    let ldvr = ldvr as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let compute_vl = jobvl == b'V';
    let compute_vr = jobvr == b'V';

    // Convert to Mat
    let mut mat: Mat<f64> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat[(i, j)] = *a.add(idx);
        }
    }

    // Use compute_full if left eigenvectors are requested, otherwise use compute
    let evd_result = if compute_vl {
        evd::GeneralEvd::compute_full(mat.as_ref())
    } else {
        evd::GeneralEvd::compute(mat.as_ref())
    };

    match evd_result {
        Ok(evd_result) => {
            let eigenvalues = evd_result.eigenvalues();
            let wr_slice = slice::from_raw_parts_mut(wr, n);
            let wi_slice = slice::from_raw_parts_mut(wi, n);

            for i in 0..n {
                wr_slice[i] = eigenvalues[i].real;
                wi_slice[i] = eigenvalues[i].imag;
            }

            // Copy left eigenvectors if requested
            if compute_vl && !vl.is_null() {
                if let Some(left_eigenvectors) = evd_result.left_eigenvectors_real() {
                    for i in 0..n {
                        for j in 0..n {
                            let idx = if row_major {
                                i * ldvl + j
                            } else {
                                j * ldvl + i
                            };
                            *vl.add(idx) = left_eigenvectors[(i, j)];
                        }
                    }
                }
            }

            // Copy right eigenvectors if requested
            if compute_vr && !vr.is_null() {
                if let Some(eigenvectors) = evd_result.eigenvectors_real() {
                    // LAPACK stores complex conjugate pairs in adjacent columns
                    // eigenvectors_real() returns the real parts
                    for i in 0..n {
                        for j in 0..n {
                            let idx = if row_major {
                                i * ldvr + j
                            } else {
                                j * ldvr + i
                            };
                            *vr.add(idx) = eigenvectors[(i, j)];
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
// SSYEVD - Symmetric eigenvalue decomposition using divide-and-conquer (single precision)
// =============================================================================

/// Computes eigenvalues and optionally eigenvectors of a symmetric matrix
/// using the divide-and-conquer algorithm.
///
/// # Safety
/// - `a` must point to a valid n x n symmetric matrix
/// - `w` must point to an array of n elements for eigenvalues
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ssyevd(
    layout: OblasLayout,
    jobz: u8, // 'N' = eigenvalues only, 'V' = eigenvalues and eigenvectors
    uplo: OblasUplo,
    n: c_int,
    a: *mut f32,
    lda: c_int,
    w: *mut f32,
) -> c_int {
    if n <= 0 || a.is_null() || w.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let compute_eigenvectors = jobz == b'V';

    // Convert to full symmetric Mat
    let mut mat: Mat<f32> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat[(i, j)] = *a.add(idx);
        }
    }

    // Symmetrize the matrix
    for i in 0..n {
        for j in (i + 1)..n {
            if upper {
                mat[(j, i)] = mat[(i, j)];
            } else {
                mat[(i, j)] = mat[(j, i)];
            }
        }
    }

    // Compute symmetric eigenvalue decomposition using divide-and-conquer
    match evd::SymmetricEvdDc::compute(mat.as_ref()) {
        Ok(evd_result) => {
            let eigenvalues = evd_result.eigenvalues();
            let w_slice = slice::from_raw_parts_mut(w, n);
            for i in 0..n {
                w_slice[i] = eigenvalues[i];
            }

            // Copy eigenvectors back to a if requested
            if compute_eigenvectors {
                let eigenvectors = evd_result.eigenvectors();
                for i in 0..n {
                    for j in 0..n {
                        let idx = if row_major { i * lda + j } else { j * lda + i };
                        *a.add(idx) = eigenvectors[(i, j)];
                    }
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::NotConverged as c_int,
    }
}

// =============================================================================
// DSYEVD - Symmetric eigenvalue decomposition using divide-and-conquer (double precision)
// =============================================================================

/// Computes eigenvalues and optionally eigenvectors of a symmetric matrix
/// using the divide-and-conquer algorithm.
///
/// # Safety
/// - `a` must point to a valid n x n symmetric matrix
/// - `w` must point to an array of n elements for eigenvalues
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dsyevd(
    layout: OblasLayout,
    jobz: u8, // 'N' = eigenvalues only, 'V' = eigenvalues and eigenvectors
    uplo: OblasUplo,
    n: c_int,
    a: *mut f64,
    lda: c_int,
    w: *mut f64,
) -> c_int {
    if n <= 0 || a.is_null() || w.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let compute_eigenvectors = jobz == b'V';

    // Convert to full symmetric Mat
    let mut mat: Mat<f64> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat[(i, j)] = *a.add(idx);
        }
    }

    // Symmetrize the matrix
    for i in 0..n {
        for j in (i + 1)..n {
            if upper {
                mat[(j, i)] = mat[(i, j)];
            } else {
                mat[(i, j)] = mat[(j, i)];
            }
        }
    }

    // Compute symmetric eigenvalue decomposition using divide-and-conquer
    match evd::SymmetricEvdDc::compute(mat.as_ref()) {
        Ok(evd_result) => {
            let eigenvalues = evd_result.eigenvalues();
            let w_slice = slice::from_raw_parts_mut(w, n);
            for i in 0..n {
                w_slice[i] = eigenvalues[i];
            }

            // Copy eigenvectors back to a if requested
            if compute_eigenvectors {
                let eigenvectors = evd_result.eigenvectors();
                for i in 0..n {
                    for j in 0..n {
                        let idx = if row_major { i * lda + j } else { j * lda + i };
                        *a.add(idx) = eigenvectors[(i, j)];
                    }
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::NotConverged as c_int,
    }
}

// =============================================================================
// ZHEEV - Hermitian eigenvalue decomposition (double precision complex)
// =============================================================================

/// Computes eigenvalues and optionally eigenvectors of a Hermitian matrix.
///
/// For a Hermitian matrix (A = A^H), all eigenvalues are real and eigenvectors
/// are unitary.
///
/// # Arguments
/// * `layout` - Matrix layout (row or column major)
/// * `jobz` - 'N' = eigenvalues only, 'V' = eigenvalues and eigenvectors
/// * `uplo` - Use 'U'pper or 'L'ower triangle of A
/// * `n` - Order of the matrix
/// * `a` - On entry: Hermitian matrix. On exit: eigenvectors if jobz='V'
/// * `lda` - Leading dimension of A
/// * `w` - Output array of n real eigenvalues in ascending order
///
/// # Safety
/// - `a` must point to a valid n x n Hermitian complex matrix
/// - `w` must point to an array of n real elements for eigenvalues
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zheev(
    layout: OblasLayout,
    jobz: u8, // 'N' = eigenvalues only, 'V' = eigenvalues and eigenvectors
    uplo: OblasUplo,
    n: c_int,
    a: *mut OblasComplex64,
    lda: c_int,
    w: *mut f64, // Eigenvalues are real
) -> c_int {
    if n <= 0 || a.is_null() || w.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let compute_eigenvectors = jobz == b'V';

    // Convert to full Hermitian Mat
    let mut mat: Mat<Complex64> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            let c = *a.add(idx);
            mat[(i, j)] = Complex64::new(c.re, c.im);
        }
    }

    // Make the matrix Hermitian using the specified triangle
    for i in 0..n {
        for j in (i + 1)..n {
            if upper {
                mat[(j, i)] = mat[(i, j)].conj();
            } else {
                mat[(i, j)] = mat[(j, i)].conj();
            }
        }
        // Ensure diagonal is real
        mat[(i, i)] = Complex64::new(mat[(i, i)].re, 0.0);
    }

    // Compute Hermitian eigenvalue decomposition
    match evd::HermitianEvd::compute(mat.as_ref()) {
        Ok(evd_result) => {
            let eigenvalues = evd_result.eigenvalues();
            let w_slice = slice::from_raw_parts_mut(w, n);
            for i in 0..n {
                w_slice[i] = eigenvalues[i];
            }

            // Copy eigenvectors back to a if requested
            if compute_eigenvectors {
                let eigenvectors = evd_result.eigenvectors();
                for i in 0..n {
                    for j in 0..n {
                        let idx = if row_major { i * lda + j } else { j * lda + i };
                        let c = eigenvectors[(i, j)];
                        *a.add(idx) = OblasComplex64 { re: c.re, im: c.im };
                    }
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::NotConverged as c_int,
    }
}

// =============================================================================
// ZGEEV - Complex general eigenvalue decomposition (double precision)
// =============================================================================

/// Computes eigenvalues and optionally eigenvectors of a general complex matrix.
///
/// For complex matrices, eigenvalues are directly complex (stored in array w).
///
/// # Arguments
/// * `layout` - Matrix layout (row or column major)
/// * `jobvl` - 'N' = don't compute left eigenvectors, 'V' = compute
/// * `jobvr` - 'N' = don't compute right eigenvectors, 'V' = compute
/// * `n` - Order of the matrix
/// * `a` - On entry: general complex matrix
/// * `lda` - Leading dimension of A
/// * `w` - Output array of n complex eigenvalues
/// * `vl` - Left eigenvectors (n x n) or NULL
/// * `ldvl` - Leading dimension of VL
/// * `vr` - Right eigenvectors (n x n) or NULL
/// * `ldvr` - Leading dimension of VR
///
/// # Safety
/// - `a` must point to a valid n x n complex matrix
/// - `w` must point to an array of n complex elements for eigenvalues
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zgeev(
    layout: OblasLayout,
    jobvl: u8, // 'N' = don't compute left eigenvectors, 'V' = compute
    jobvr: u8, // 'N' = don't compute right eigenvectors, 'V' = compute
    n: c_int,
    a: *mut OblasComplex64,
    lda: c_int,
    w: *mut OblasComplex64,  // Complex eigenvalues
    vl: *mut OblasComplex64, // Left eigenvectors (n x n)
    ldvl: c_int,
    vr: *mut OblasComplex64, // Right eigenvectors (n x n)
    ldvr: c_int,
) -> c_int {
    if n <= 0 || a.is_null() || w.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let lda = lda as usize;
    let ldvl = ldvl as usize;
    let ldvr = ldvr as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let compute_vl = jobvl == b'V';
    let compute_vr = jobvr == b'V';

    // Convert to Mat
    let mut mat: Mat<Complex64> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            let c = *a.add(idx);
            mat[(i, j)] = Complex64::new(c.re, c.im);
        }
    }

    // Use compute_full if left eigenvectors are requested, otherwise use compute
    let evd_result = if compute_vl {
        evd::ComplexGeneralEvd::compute_full(mat.as_ref())
    } else if compute_vr {
        evd::ComplexGeneralEvd::compute(mat.as_ref())
    } else {
        evd::ComplexGeneralEvd::eigenvalues_only(mat.as_ref())
    };

    match evd_result {
        Ok(evd_result) => {
            let eigenvalues = evd_result.eigenvalues();
            let w_slice = slice::from_raw_parts_mut(w, n);

            for i in 0..n {
                w_slice[i] = OblasComplex64 {
                    re: eigenvalues[i].re,
                    im: eigenvalues[i].im,
                };
            }

            // Copy left eigenvectors if requested
            if compute_vl && !vl.is_null() {
                if let Some(left_eigenvectors) = evd_result.left_eigenvectors() {
                    for i in 0..n {
                        for j in 0..n {
                            let idx = if row_major {
                                i * ldvl + j
                            } else {
                                j * ldvl + i
                            };
                            let c = left_eigenvectors[(i, j)];
                            *vl.add(idx) = OblasComplex64 { re: c.re, im: c.im };
                        }
                    }
                }
            }

            // Copy right eigenvectors if requested
            if compute_vr && !vr.is_null() {
                if let Some(eigenvectors) = evd_result.eigenvectors() {
                    for i in 0..n {
                        for j in 0..n {
                            let idx = if row_major {
                                i * ldvr + j
                            } else {
                                j * ldvr + i
                            };
                            let c = eigenvectors[(i, j)];
                            *vr.add(idx) = OblasComplex64 { re: c.re, im: c.im };
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
// CGEEV - Complex general eigenvalue decomposition (single precision)
// =============================================================================

/// Computes eigenvalues and optionally eigenvectors of a general complex matrix.
///
/// Single precision version of complex general EVD.
///
/// # Safety
/// - `a` must point to a valid n x n complex matrix
/// - `w` must point to an array of n complex elements for eigenvalues
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_cgeev(
    layout: OblasLayout,
    jobvl: u8, // 'N' = don't compute left eigenvectors, 'V' = compute
    jobvr: u8, // 'N' = don't compute right eigenvectors, 'V' = compute
    n: c_int,
    a: *mut OblasComplex32,
    lda: c_int,
    w: *mut OblasComplex32,  // Complex eigenvalues
    vl: *mut OblasComplex32, // Left eigenvectors (n x n)
    ldvl: c_int,
    vr: *mut OblasComplex32, // Right eigenvectors (n x n)
    ldvr: c_int,
) -> c_int {
    if n <= 0 || a.is_null() || w.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let lda = lda as usize;
    let ldvl = ldvl as usize;
    let ldvr = ldvr as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let compute_vl = jobvl == b'V';
    let compute_vr = jobvr == b'V';

    // Convert to Mat
    let mut mat: Mat<Complex32> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            let c = *a.add(idx);
            mat[(i, j)] = Complex32::new(c.re, c.im);
        }
    }

    // Use compute_full if left eigenvectors are requested, otherwise use compute
    let evd_result = if compute_vl {
        evd::ComplexGeneralEvd::compute_full(mat.as_ref())
    } else if compute_vr {
        evd::ComplexGeneralEvd::compute(mat.as_ref())
    } else {
        evd::ComplexGeneralEvd::eigenvalues_only(mat.as_ref())
    };

    match evd_result {
        Ok(evd_result) => {
            let eigenvalues = evd_result.eigenvalues();
            let w_slice = slice::from_raw_parts_mut(w, n);

            for i in 0..n {
                w_slice[i] = OblasComplex32 {
                    re: eigenvalues[i].re,
                    im: eigenvalues[i].im,
                };
            }

            // Copy left eigenvectors if requested
            if compute_vl && !vl.is_null() {
                if let Some(left_eigenvectors) = evd_result.left_eigenvectors() {
                    for i in 0..n {
                        for j in 0..n {
                            let idx = if row_major {
                                i * ldvl + j
                            } else {
                                j * ldvl + i
                            };
                            let c = left_eigenvectors[(i, j)];
                            *vl.add(idx) = OblasComplex32 { re: c.re, im: c.im };
                        }
                    }
                }
            }

            // Copy right eigenvectors if requested
            if compute_vr && !vr.is_null() {
                if let Some(eigenvectors) = evd_result.eigenvectors() {
                    for i in 0..n {
                        for j in 0..n {
                            let idx = if row_major {
                                i * ldvr + j
                            } else {
                                j * ldvr + i
                            };
                            let c = eigenvectors[(i, j)];
                            *vr.add(idx) = OblasComplex32 { re: c.re, im: c.im };
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
// CHEEV - Hermitian eigenvalue decomposition (single precision complex)
// =============================================================================

/// Computes eigenvalues and optionally eigenvectors of a Hermitian matrix.
///
/// Single precision version of Hermitian EVD.
///
/// # Safety
/// - `a` must point to a valid n x n Hermitian complex matrix
/// - `w` must point to an array of n real elements for eigenvalues
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_cheev(
    layout: OblasLayout,
    jobz: u8, // 'N' = eigenvalues only, 'V' = eigenvalues and eigenvectors
    uplo: OblasUplo,
    n: c_int,
    a: *mut OblasComplex32,
    lda: c_int,
    w: *mut f32, // Eigenvalues are real
) -> c_int {
    if n <= 0 || a.is_null() || w.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let compute_eigenvectors = jobz == b'V';

    // Convert to full Hermitian Mat
    let mut mat: Mat<Complex32> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            let c = *a.add(idx);
            mat[(i, j)] = Complex32::new(c.re, c.im);
        }
    }

    // Make the matrix Hermitian using the specified triangle
    for i in 0..n {
        for j in (i + 1)..n {
            if upper {
                mat[(j, i)] = mat[(i, j)].conj();
            } else {
                mat[(i, j)] = mat[(j, i)].conj();
            }
        }
        // Ensure diagonal is real
        mat[(i, i)] = Complex32::new(mat[(i, i)].re, 0.0);
    }

    // Compute Hermitian eigenvalue decomposition
    match evd::HermitianEvd::compute(mat.as_ref()) {
        Ok(evd_result) => {
            let eigenvalues = evd_result.eigenvalues();
            let w_slice = slice::from_raw_parts_mut(w, n);
            for i in 0..n {
                w_slice[i] = eigenvalues[i];
            }

            // Copy eigenvectors back to a if requested
            if compute_eigenvectors {
                let eigenvectors = evd_result.eigenvectors();
                for i in 0..n {
                    for j in 0..n {
                        let idx = if row_major { i * lda + j } else { j * lda + i };
                        let c = eigenvectors[(i, j)];
                        *a.add(idx) = OblasComplex32 { re: c.re, im: c.im };
                    }
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::NotConverged as c_int,
    }
}

// =============================================================================
// ZHEEVD - Hermitian eigenvalue decomposition using divide-and-conquer (double precision)
// =============================================================================

/// Computes eigenvalues and optionally eigenvectors of a Hermitian matrix
/// using the divide-and-conquer algorithm.
///
/// For a Hermitian matrix (A = A^H), all eigenvalues are real and eigenvectors
/// are unitary. The divide-and-conquer algorithm is typically faster than
/// the standard QR iteration for larger matrices.
///
/// # Arguments
/// * `layout` - Matrix layout (row or column major)
/// * `jobz` - 'N' = eigenvalues only, 'V' = eigenvalues and eigenvectors
/// * `uplo` - Use 'U'pper or 'L'ower triangle of A
/// * `n` - Order of the matrix
/// * `a` - On entry: Hermitian matrix. On exit: eigenvectors if jobz='V'
/// * `lda` - Leading dimension of A
/// * `w` - Output array of n real eigenvalues in ascending order
///
/// # Safety
/// - `a` must point to a valid n x n Hermitian complex matrix
/// - `w` must point to an array of n real elements for eigenvalues
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zheevd(
    layout: OblasLayout,
    jobz: u8, // 'N' = eigenvalues only, 'V' = eigenvalues and eigenvectors
    uplo: OblasUplo,
    n: c_int,
    a: *mut OblasComplex64,
    lda: c_int,
    w: *mut f64, // Eigenvalues are real
) -> c_int {
    if n <= 0 || a.is_null() || w.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let compute_eigenvectors = jobz == b'V';

    // Convert to full Hermitian Mat
    let mut mat: Mat<Complex64> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            let c = *a.add(idx);
            mat[(i, j)] = Complex64::new(c.re, c.im);
        }
    }

    // Make the matrix Hermitian using the specified triangle
    for i in 0..n {
        for j in (i + 1)..n {
            if upper {
                mat[(j, i)] = mat[(i, j)].conj();
            } else {
                mat[(i, j)] = mat[(j, i)].conj();
            }
        }
        // Ensure diagonal is real
        mat[(i, i)] = Complex64::new(mat[(i, i)].re, 0.0);
    }

    // Compute Hermitian eigenvalue decomposition using divide-and-conquer
    match evd::HermitianEvdDc::compute(mat.as_ref()) {
        Ok(evd_result) => {
            let eigenvalues = evd_result.eigenvalues();
            let w_slice = slice::from_raw_parts_mut(w, n);
            for i in 0..n {
                w_slice[i] = eigenvalues[i];
            }

            // Copy eigenvectors back to a if requested
            if compute_eigenvectors {
                let eigenvectors = evd_result.eigenvectors();
                for i in 0..n {
                    for j in 0..n {
                        let idx = if row_major { i * lda + j } else { j * lda + i };
                        let c = eigenvectors[(i, j)];
                        *a.add(idx) = OblasComplex64 { re: c.re, im: c.im };
                    }
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::NotConverged as c_int,
    }
}

// =============================================================================
// CHEEVD - Hermitian eigenvalue decomposition using divide-and-conquer (single precision)
// =============================================================================

/// Computes eigenvalues and optionally eigenvectors of a Hermitian matrix
/// using the divide-and-conquer algorithm.
///
/// Single precision version of Hermitian D&C EVD.
///
/// # Safety
/// - `a` must point to a valid n x n Hermitian complex matrix
/// - `w` must point to an array of n real elements for eigenvalues
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_cheevd(
    layout: OblasLayout,
    jobz: u8, // 'N' = eigenvalues only, 'V' = eigenvalues and eigenvectors
    uplo: OblasUplo,
    n: c_int,
    a: *mut OblasComplex32,
    lda: c_int,
    w: *mut f32, // Eigenvalues are real
) -> c_int {
    if n <= 0 || a.is_null() || w.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let upper = uplo == OblasUplo::Upper;
    let compute_eigenvectors = jobz == b'V';

    // Convert to full Hermitian Mat
    let mut mat: Mat<Complex32> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            let c = *a.add(idx);
            mat[(i, j)] = Complex32::new(c.re, c.im);
        }
    }

    // Make the matrix Hermitian using the specified triangle
    for i in 0..n {
        for j in (i + 1)..n {
            if upper {
                mat[(j, i)] = mat[(i, j)].conj();
            } else {
                mat[(i, j)] = mat[(j, i)].conj();
            }
        }
        // Ensure diagonal is real
        mat[(i, i)] = Complex32::new(mat[(i, i)].re, 0.0);
    }

    // Compute Hermitian eigenvalue decomposition using divide-and-conquer
    match evd::HermitianEvdDc::compute(mat.as_ref()) {
        Ok(evd_result) => {
            let eigenvalues = evd_result.eigenvalues();
            let w_slice = slice::from_raw_parts_mut(w, n);
            for i in 0..n {
                w_slice[i] = eigenvalues[i];
            }

            // Copy eigenvectors back to a if requested
            if compute_eigenvectors {
                let eigenvectors = evd_result.eigenvectors();
                for i in 0..n {
                    for j in 0..n {
                        let idx = if row_major { i * lda + j } else { j * lda + i };
                        let c = eigenvectors[(i, j)];
                        *a.add(idx) = OblasComplex32 { re: c.re, im: c.im };
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
    fn test_dsyev() {
        // Symmetric matrix A = [[2, 1], [1, 2]] (column-major)
        // Eigenvalues should be 1 and 3
        let mut a = [2.0f64, 1.0, 1.0, 2.0];
        let mut w = [0.0f64; 2];

        unsafe {
            let result = oblas_dsyev(
                OblasLayout::ColMajor,
                b'N',
                OblasUplo::Upper,
                2,
                a.as_mut_ptr(),
                2,
                w.as_mut_ptr(),
            );
            assert_eq!(result, 0);

            // Eigenvalues should be 1 and 3
            w.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert!((w[0] - 1.0).abs() < 1e-8);
            assert!((w[1] - 3.0).abs() < 1e-8);
        }
    }

    #[test]
    fn test_dgeev() {
        // Matrix A = [[1, 2], [0, 3]] (column-major, upper triangular)
        // Eigenvalues are 1 and 3 (diagonal elements)
        let mut a = [1.0f64, 0.0, 2.0, 3.0];
        let mut wr = [0.0f64; 2];
        let mut wi = [0.0f64; 2];

        unsafe {
            let result = oblas_dgeev(
                OblasLayout::ColMajor,
                b'N',
                b'N',
                2,
                a.as_mut_ptr(),
                2,
                wr.as_mut_ptr(),
                wi.as_mut_ptr(),
                std::ptr::null_mut(),
                2,
                std::ptr::null_mut(),
                2,
            );
            assert_eq!(result, 0);

            // Eigenvalues should be 1 and 3 with zero imaginary parts
            let mut eigenvalues: Vec<f64> = wr.iter().copied().collect();
            eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert!((eigenvalues[0] - 1.0).abs() < 1e-8);
            assert!((eigenvalues[1] - 3.0).abs() < 1e-8);
            assert!(wi[0].abs() < 1e-8);
            assert!(wi[1].abs() < 1e-8);
        }
    }

    #[test]
    fn test_dsyevd() {
        // Symmetric matrix A = [[2, 1], [1, 2]] (column-major)
        // Eigenvalues should be 1 and 3
        let mut a = [2.0f64, 1.0, 1.0, 2.0];
        let mut w = [0.0f64; 2];

        unsafe {
            let result = oblas_dsyevd(
                OblasLayout::ColMajor,
                b'N',
                OblasUplo::Upper,
                2,
                a.as_mut_ptr(),
                2,
                w.as_mut_ptr(),
            );
            assert_eq!(result, 0);

            // Eigenvalues should be 1 and 3
            w.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert!((w[0] - 1.0).abs() < 1e-8);
            assert!((w[1] - 3.0).abs() < 1e-8);
        }
    }

    #[test]
    fn test_dsyevd_with_eigenvectors() {
        // Symmetric matrix A = [[4, 1], [1, 3]] (column-major)
        let mut a = [4.0f64, 1.0, 1.0, 3.0];
        let mut w = [0.0f64; 2];

        unsafe {
            let result = oblas_dsyevd(
                OblasLayout::ColMajor,
                b'V', // Compute eigenvectors
                OblasUplo::Upper,
                2,
                a.as_mut_ptr(),
                2,
                w.as_mut_ptr(),
            );
            assert_eq!(result, 0);

            // Eigenvalues should be in ascending order
            assert!(w[0] < w[1]);

            // Verify eigenvectors are orthonormal
            // V = [v1, v2] stored column-major: [v1[0], v1[1], v2[0], v2[1]]
            let v11 = a[0];
            let v21 = a[1];
            let v12 = a[2];
            let v22 = a[3];

            // v1^T * v1 = 1
            let dot11 = v11 * v11 + v21 * v21;
            assert!((dot11 - 1.0).abs() < 1e-8, "v1 not normalized: {}", dot11);

            // v2^T * v2 = 1
            let dot22 = v12 * v12 + v22 * v22;
            assert!((dot22 - 1.0).abs() < 1e-8, "v2 not normalized: {}", dot22);

            // v1^T * v2 = 0
            let dot12 = v11 * v12 + v21 * v22;
            assert!(dot12.abs() < 1e-8, "v1, v2 not orthogonal: {}", dot12);
        }
    }

    #[test]
    fn test_ssyevd() {
        // Single precision version
        let mut a = [2.0f32, 1.0, 1.0, 2.0];
        let mut w = [0.0f32; 2];

        unsafe {
            let result = oblas_ssyevd(
                OblasLayout::ColMajor,
                b'N',
                OblasUplo::Upper,
                2,
                a.as_mut_ptr(),
                2,
                w.as_mut_ptr(),
            );
            assert_eq!(result, 0);

            // Eigenvalues should be 1 and 3
            w.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert!((w[0] - 1.0).abs() < 1e-5);
            assert!((w[1] - 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dsyevd_3x3() {
        // 3x3 symmetric matrix (column-major)
        let mut a = [
            4.0f64, 1.0, 1.0, // column 1
            1.0, 3.0, 2.0, // column 2
            1.0, 2.0, 3.0, // column 3
        ];
        let mut w = [0.0f64; 3];

        unsafe {
            let result = oblas_dsyevd(
                OblasLayout::ColMajor,
                b'V',
                OblasUplo::Upper,
                3,
                a.as_mut_ptr(),
                3,
                w.as_mut_ptr(),
            );
            assert_eq!(result, 0);

            // Eigenvalues should be in ascending order
            assert!(w[0] <= w[1] && w[1] <= w[2]);
        }
    }

    #[test]
    fn test_zheev_real_symmetric() {
        // Hermitian matrix that's actually real symmetric
        // [[2, 1], [1, 2]] has eigenvalues 1 and 3
        let mut a = [
            OblasComplex64 { re: 2.0, im: 0.0 },
            OblasComplex64 { re: 1.0, im: 0.0 },
            OblasComplex64 { re: 1.0, im: 0.0 },
            OblasComplex64 { re: 2.0, im: 0.0 },
        ];
        let mut w = [0.0f64; 2];

        unsafe {
            let result = oblas_zheev(
                OblasLayout::ColMajor,
                b'N',
                OblasUplo::Upper,
                2,
                a.as_mut_ptr(),
                2,
                w.as_mut_ptr(),
            );
            assert_eq!(result, 0);

            // Eigenvalues should be 1 and 3
            assert!((w[0] - 1.0).abs() < 1e-8);
            assert!((w[1] - 3.0).abs() < 1e-8);
        }
    }

    #[test]
    fn test_zheev_complex() {
        // Hermitian matrix with complex off-diagonal
        // [[2, 1+i], [1-i, 3]] has eigenvalues 1 and 4
        let mut a = [
            OblasComplex64 { re: 2.0, im: 0.0 },
            OblasComplex64 { re: 1.0, im: -1.0 }, // (1,0) position in col-major
            OblasComplex64 { re: 1.0, im: 1.0 },  // (0,1) position in col-major
            OblasComplex64 { re: 3.0, im: 0.0 },
        ];
        let mut w = [0.0f64; 2];

        unsafe {
            let result = oblas_zheev(
                OblasLayout::ColMajor,
                b'N',
                OblasUplo::Upper,
                2,
                a.as_mut_ptr(),
                2,
                w.as_mut_ptr(),
            );
            assert_eq!(result, 0);

            // Eigenvalues should be 1 and 4
            assert!((w[0] - 1.0).abs() < 1e-8, "First eigenvalue: {}", w[0]);
            assert!((w[1] - 4.0).abs() < 1e-8, "Second eigenvalue: {}", w[1]);
        }
    }

    #[test]
    fn test_zheev_with_eigenvectors() {
        // Hermitian matrix with eigenvector computation
        let mut a = [
            OblasComplex64 { re: 2.0, im: 0.0 },
            OblasComplex64 { re: 1.0, im: 0.0 },
            OblasComplex64 { re: 1.0, im: 0.0 },
            OblasComplex64 { re: 2.0, im: 0.0 },
        ];
        let mut w = [0.0f64; 2];

        unsafe {
            let result = oblas_zheev(
                OblasLayout::ColMajor,
                b'V', // Compute eigenvectors
                OblasUplo::Upper,
                2,
                a.as_mut_ptr(),
                2,
                w.as_mut_ptr(),
            );
            assert_eq!(result, 0);

            // Eigenvalues should be 1 and 3
            assert!((w[0] - 1.0).abs() < 1e-8);
            assert!((w[1] - 3.0).abs() < 1e-8);

            // Verify eigenvectors are unitary (U^H * U = I)
            // a now contains eigenvectors in column-major order
            let u00 = &a[0];
            let u10 = &a[1];
            let u01 = &a[2];
            let u11 = &a[3];

            // First column norm
            let norm1_sq = u00.re * u00.re + u00.im * u00.im + u10.re * u10.re + u10.im * u10.im;
            assert!(
                (norm1_sq - 1.0).abs() < 1e-8,
                "First eigenvector not normalized: {}",
                norm1_sq
            );

            // Second column norm
            let norm2_sq = u01.re * u01.re + u01.im * u01.im + u11.re * u11.re + u11.im * u11.im;
            assert!(
                (norm2_sq - 1.0).abs() < 1e-8,
                "Second eigenvector not normalized: {}",
                norm2_sq
            );
        }
    }

    #[test]
    fn test_cheev_basic() {
        // Single precision Hermitian EVD
        let mut a = [
            OblasComplex32 { re: 2.0, im: 0.0 },
            OblasComplex32 { re: 1.0, im: 0.0 },
            OblasComplex32 { re: 1.0, im: 0.0 },
            OblasComplex32 { re: 2.0, im: 0.0 },
        ];
        let mut w = [0.0f32; 2];

        unsafe {
            let result = oblas_cheev(
                OblasLayout::ColMajor,
                b'N',
                OblasUplo::Upper,
                2,
                a.as_mut_ptr(),
                2,
                w.as_mut_ptr(),
            );
            assert_eq!(result, 0);

            // Eigenvalues should be 1 and 3
            assert!((w[0] - 1.0).abs() < 1e-5);
            assert!((w[1] - 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_zheev_row_major() {
        // Test row-major layout
        // [[2, 1+i], [1-i, 3]] in row-major
        let mut a = [
            OblasComplex64 { re: 2.0, im: 0.0 },
            OblasComplex64 { re: 1.0, im: 1.0 }, // (0,1) position in row-major
            OblasComplex64 { re: 1.0, im: -1.0 }, // (1,0) position in row-major
            OblasComplex64 { re: 3.0, im: 0.0 },
        ];
        let mut w = [0.0f64; 2];

        unsafe {
            let result = oblas_zheev(
                OblasLayout::RowMajor,
                b'N',
                OblasUplo::Upper,
                2,
                a.as_mut_ptr(),
                2,
                w.as_mut_ptr(),
            );
            assert_eq!(result, 0);

            // Eigenvalues should be 1 and 4
            assert!((w[0] - 1.0).abs() < 1e-8, "First eigenvalue: {}", w[0]);
            assert!((w[1] - 4.0).abs() < 1e-8, "Second eigenvalue: {}", w[1]);
        }
    }

    #[test]
    fn test_zgeev_diagonal() {
        // Diagonal complex matrix [[1+i, 0], [0, 2-i]]
        // Eigenvalues are diagonal elements
        let mut a = [
            OblasComplex64 { re: 1.0, im: 1.0 },
            OblasComplex64 { re: 0.0, im: 0.0 },
            OblasComplex64 { re: 0.0, im: 0.0 },
            OblasComplex64 { re: 2.0, im: -1.0 },
        ];
        let mut w = [OblasComplex64 { re: 0.0, im: 0.0 }; 2];

        unsafe {
            let result = oblas_zgeev(
                OblasLayout::ColMajor,
                b'N',
                b'N',
                2,
                a.as_mut_ptr(),
                2,
                w.as_mut_ptr(),
                std::ptr::null_mut(),
                2,
                std::ptr::null_mut(),
                2,
            );
            assert_eq!(result, 0);

            // Check eigenvalues (diagonal matrix has diagonal elements as eigenvalues)
            let mut eigenvalues: Vec<(f64, f64)> = w.iter().map(|c| (c.re, c.im)).collect();
            eigenvalues.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            assert!((eigenvalues[0].0 - 1.0).abs() < 1e-8);
            assert!((eigenvalues[0].1 - 1.0).abs() < 1e-8);
            assert!((eigenvalues[1].0 - 2.0).abs() < 1e-8);
            assert!((eigenvalues[1].1 - (-1.0)).abs() < 1e-8);
        }
    }

    #[test]
    fn test_zgeev_trace() {
        // Complex matrix - sum of eigenvalues equals trace
        let mut a = [
            OblasComplex64 { re: 1.0, im: 2.0 },
            OblasComplex64 { re: 3.0, im: -1.0 },
            OblasComplex64 { re: 2.0, im: 1.0 },
            OblasComplex64 { re: 4.0, im: -2.0 },
        ];
        let mut w = [OblasComplex64 { re: 0.0, im: 0.0 }; 2];

        // Trace = (1+2i) + (4-2i) = 5+0i
        let trace_re = 1.0 + 4.0;
        let trace_im = 2.0 + (-2.0);

        unsafe {
            let result = oblas_zgeev(
                OblasLayout::ColMajor,
                b'N',
                b'N',
                2,
                a.as_mut_ptr(),
                2,
                w.as_mut_ptr(),
                std::ptr::null_mut(),
                2,
                std::ptr::null_mut(),
                2,
            );
            assert_eq!(result, 0);

            // Sum of eigenvalues should equal trace
            let sum_re = w[0].re + w[1].re;
            let sum_im = w[0].im + w[1].im;

            assert!(
                (sum_re - trace_re).abs() < 1e-8,
                "Trace mismatch: {} vs {}",
                sum_re,
                trace_re
            );
            assert!(
                (sum_im - trace_im).abs() < 1e-8,
                "Trace mismatch: {} vs {}",
                sum_im,
                trace_im
            );
        }
    }

    #[test]
    fn test_zgeev_with_eigenvectors() {
        // Diagonal matrix - eigenvectors should be basis vectors
        let mut a = [
            OblasComplex64 { re: 1.0, im: 0.0 },
            OblasComplex64 { re: 0.0, im: 0.0 },
            OblasComplex64 { re: 0.0, im: 0.0 },
            OblasComplex64 { re: 2.0, im: 0.0 },
        ];
        let mut w = [OblasComplex64 { re: 0.0, im: 0.0 }; 2];
        let mut vr = [OblasComplex64 { re: 0.0, im: 0.0 }; 4];

        unsafe {
            let result = oblas_zgeev(
                OblasLayout::ColMajor,
                b'N',
                b'V', // Compute right eigenvectors
                2,
                a.as_mut_ptr(),
                2,
                w.as_mut_ptr(),
                std::ptr::null_mut(),
                2,
                vr.as_mut_ptr(),
                2,
            );
            assert_eq!(result, 0);

            // Check eigenvalues
            let mut eigenvalues: Vec<f64> = w.iter().map(|c| c.re).collect();
            eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert!((eigenvalues[0] - 1.0).abs() < 1e-8);
            assert!((eigenvalues[1] - 2.0).abs() < 1e-8);
        }
    }

    #[test]
    fn test_cgeev_basic() {
        // Single precision complex general EVD
        let mut a = [
            OblasComplex32 { re: 1.0, im: 0.0 },
            OblasComplex32 { re: 0.0, im: 0.0 },
            OblasComplex32 { re: 0.0, im: 0.0 },
            OblasComplex32 { re: 2.0, im: 0.0 },
        ];
        let mut w = [OblasComplex32 { re: 0.0, im: 0.0 }; 2];

        unsafe {
            let result = oblas_cgeev(
                OblasLayout::ColMajor,
                b'N',
                b'N',
                2,
                a.as_mut_ptr(),
                2,
                w.as_mut_ptr(),
                std::ptr::null_mut(),
                2,
                std::ptr::null_mut(),
                2,
            );
            assert_eq!(result, 0);

            // Check eigenvalues
            let mut eigenvalues: Vec<f32> = w.iter().map(|c| c.re).collect();
            eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert!((eigenvalues[0] - 1.0).abs() < 1e-5);
            assert!((eigenvalues[1] - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_zgeev_row_major() {
        // Row-major layout test
        let mut a = [
            OblasComplex64 { re: 1.0, im: 0.0 },
            OblasComplex64 { re: 0.0, im: 0.0 },
            OblasComplex64 { re: 0.0, im: 0.0 },
            OblasComplex64 { re: 2.0, im: 0.0 },
        ];
        let mut w = [OblasComplex64 { re: 0.0, im: 0.0 }; 2];

        unsafe {
            let result = oblas_zgeev(
                OblasLayout::RowMajor,
                b'N',
                b'N',
                2,
                a.as_mut_ptr(),
                2,
                w.as_mut_ptr(),
                std::ptr::null_mut(),
                2,
                std::ptr::null_mut(),
                2,
            );
            assert_eq!(result, 0);

            // Check eigenvalues
            let mut eigenvalues: Vec<f64> = w.iter().map(|c| c.re).collect();
            eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert!((eigenvalues[0] - 1.0).abs() < 1e-8);
            assert!((eigenvalues[1] - 2.0).abs() < 1e-8);
        }
    }

    // =========================================================================
    // ZHEEVD/CHEEVD - Hermitian D&C EVD Tests
    // =========================================================================

    #[test]
    fn test_zheevd_real_symmetric() {
        // Hermitian matrix that's actually real symmetric
        // [[2, 1], [1, 2]] has eigenvalues 1 and 3
        let mut a = [
            OblasComplex64 { re: 2.0, im: 0.0 },
            OblasComplex64 { re: 1.0, im: 0.0 },
            OblasComplex64 { re: 1.0, im: 0.0 },
            OblasComplex64 { re: 2.0, im: 0.0 },
        ];
        let mut w = [0.0f64; 2];

        unsafe {
            let result = oblas_zheevd(
                OblasLayout::ColMajor,
                b'N',
                OblasUplo::Upper,
                2,
                a.as_mut_ptr(),
                2,
                w.as_mut_ptr(),
            );
            assert_eq!(result, 0);

            // Eigenvalues should be 1 and 3
            assert!((w[0] - 1.0).abs() < 1e-8, "First eigenvalue: {}", w[0]);
            assert!((w[1] - 3.0).abs() < 1e-8, "Second eigenvalue: {}", w[1]);
        }
    }

    #[test]
    fn test_zheevd_complex() {
        // Hermitian matrix with complex off-diagonal
        // [[2, 1+i], [1-i, 3]] has eigenvalues 1 and 4
        let mut a = [
            OblasComplex64 { re: 2.0, im: 0.0 },
            OblasComplex64 { re: 1.0, im: -1.0 }, // (1,0) position in col-major
            OblasComplex64 { re: 1.0, im: 1.0 },  // (0,1) position in col-major
            OblasComplex64 { re: 3.0, im: 0.0 },
        ];
        let mut w = [0.0f64; 2];

        unsafe {
            let result = oblas_zheevd(
                OblasLayout::ColMajor,
                b'N',
                OblasUplo::Upper,
                2,
                a.as_mut_ptr(),
                2,
                w.as_mut_ptr(),
            );
            assert_eq!(result, 0);

            // Eigenvalues should be 1 and 4
            assert!((w[0] - 1.0).abs() < 1e-8, "First eigenvalue: {}", w[0]);
            assert!((w[1] - 4.0).abs() < 1e-8, "Second eigenvalue: {}", w[1]);
        }
    }

    #[test]
    fn test_zheevd_with_eigenvectors() {
        // Hermitian matrix with eigenvector computation
        let mut a = [
            OblasComplex64 { re: 2.0, im: 0.0 },
            OblasComplex64 { re: 1.0, im: 0.0 },
            OblasComplex64 { re: 1.0, im: 0.0 },
            OblasComplex64 { re: 2.0, im: 0.0 },
        ];
        let mut w = [0.0f64; 2];

        unsafe {
            let result = oblas_zheevd(
                OblasLayout::ColMajor,
                b'V', // Compute eigenvectors
                OblasUplo::Upper,
                2,
                a.as_mut_ptr(),
                2,
                w.as_mut_ptr(),
            );
            assert_eq!(result, 0);

            // Eigenvalues should be 1 and 3
            assert!((w[0] - 1.0).abs() < 1e-8);
            assert!((w[1] - 3.0).abs() < 1e-8);

            // Verify eigenvectors are unitary (U^H * U = I)
            let u00 = &a[0];
            let u10 = &a[1];
            let u01 = &a[2];
            let u11 = &a[3];

            // First column norm
            let norm1_sq = u00.re * u00.re + u00.im * u00.im + u10.re * u10.re + u10.im * u10.im;
            assert!(
                (norm1_sq - 1.0).abs() < 1e-8,
                "First eigenvector not normalized: {}",
                norm1_sq
            );

            // Second column norm
            let norm2_sq = u01.re * u01.re + u01.im * u01.im + u11.re * u11.re + u11.im * u11.im;
            assert!(
                (norm2_sq - 1.0).abs() < 1e-8,
                "Second eigenvector not normalized: {}",
                norm2_sq
            );
        }
    }

    #[test]
    fn test_zheevd_3x3() {
        // 3x3 Hermitian matrix (column-major)
        let mut a = [
            OblasComplex64 { re: 4.0, im: 0.0 },
            OblasComplex64 { re: 1.0, im: -1.0 },
            OblasComplex64 { re: 1.0, im: 0.0 },
            OblasComplex64 { re: 1.0, im: 1.0 },
            OblasComplex64 { re: 3.0, im: 0.0 },
            OblasComplex64 { re: 2.0, im: -1.0 },
            OblasComplex64 { re: 1.0, im: 0.0 },
            OblasComplex64 { re: 2.0, im: 1.0 },
            OblasComplex64 { re: 3.0, im: 0.0 },
        ];
        let mut w = [0.0f64; 3];

        unsafe {
            let result = oblas_zheevd(
                OblasLayout::ColMajor,
                b'V',
                OblasUplo::Upper,
                3,
                a.as_mut_ptr(),
                3,
                w.as_mut_ptr(),
            );
            assert_eq!(result, 0);

            // Eigenvalues should be in ascending order
            assert!(w[0] <= w[1] && w[1] <= w[2]);
        }
    }

    #[test]
    fn test_cheevd_basic() {
        // Single precision Hermitian D&C EVD
        let mut a = [
            OblasComplex32 { re: 2.0, im: 0.0 },
            OblasComplex32 { re: 1.0, im: 0.0 },
            OblasComplex32 { re: 1.0, im: 0.0 },
            OblasComplex32 { re: 2.0, im: 0.0 },
        ];
        let mut w = [0.0f32; 2];

        unsafe {
            let result = oblas_cheevd(
                OblasLayout::ColMajor,
                b'N',
                OblasUplo::Upper,
                2,
                a.as_mut_ptr(),
                2,
                w.as_mut_ptr(),
            );
            assert_eq!(result, 0);

            // Eigenvalues should be 1 and 3
            assert!((w[0] - 1.0).abs() < 1e-5);
            assert!((w[1] - 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_zheevd_row_major() {
        // Test row-major layout
        // [[2, 1+i], [1-i, 3]] in row-major
        let mut a = [
            OblasComplex64 { re: 2.0, im: 0.0 },
            OblasComplex64 { re: 1.0, im: 1.0 }, // (0,1) position in row-major
            OblasComplex64 { re: 1.0, im: -1.0 }, // (1,0) position in row-major
            OblasComplex64 { re: 3.0, im: 0.0 },
        ];
        let mut w = [0.0f64; 2];

        unsafe {
            let result = oblas_zheevd(
                OblasLayout::RowMajor,
                b'N',
                OblasUplo::Upper,
                2,
                a.as_mut_ptr(),
                2,
                w.as_mut_ptr(),
            );
            assert_eq!(result, 0);

            // Eigenvalues should be 1 and 4
            assert!((w[0] - 1.0).abs() < 1e-8, "First eigenvalue: {}", w[0]);
            assert!((w[1] - 4.0).abs() < 1e-8, "Second eigenvalue: {}", w[1]);
        }
    }

    #[test]
    fn test_cheevd_complex() {
        // Single precision complex Hermitian with off-diagonal
        let mut a = [
            OblasComplex32 { re: 2.0, im: 0.0 },
            OblasComplex32 { re: 1.0, im: -1.0 },
            OblasComplex32 { re: 1.0, im: 1.0 },
            OblasComplex32 { re: 3.0, im: 0.0 },
        ];
        let mut w = [0.0f32; 2];

        unsafe {
            let result = oblas_cheevd(
                OblasLayout::ColMajor,
                b'N',
                OblasUplo::Upper,
                2,
                a.as_mut_ptr(),
                2,
                w.as_mut_ptr(),
            );
            assert_eq!(result, 0);

            // Eigenvalues should be 1 and 4
            assert!((w[0] - 1.0).abs() < 1e-5, "First eigenvalue: {}", w[0]);
            assert!((w[1] - 4.0).abs() < 1e-5, "Second eigenvalue: {}", w[1]);
        }
    }
}
