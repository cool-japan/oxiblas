//! LAPACK FFI - Generalized Eigenvalue Decomposition routines.
//!
//! Solves the generalized eigenvalue problem: A * x = λ * B * x
//!
//! - `oblas_ssygv`, `oblas_dsygv` - Symmetric generalized EVD (B positive definite)
//! - `oblas_sggev`, `oblas_dggev` - General generalized EVD

use crate::types::*;
use oxiblas_lapack::evd::{GeneralizedEvd, SymmetricGeneralizedEvd};
use oxiblas_matrix::Mat;
use std::ffi::c_int;

// =============================================================================
// SSYGV - Symmetric generalized eigenvalue problem (single precision)
// =============================================================================

/// Computes the generalized eigenvalues and eigenvectors for symmetric matrices.
///
/// Solves A * x = λ * B * x where A is symmetric and B is symmetric positive definite.
///
/// # Arguments
/// * `layout` - Matrix layout
/// * `n` - Dimension of matrices A and B
/// * `a` - Symmetric matrix A (lower triangle used)
/// * `lda` - Leading dimension of A
/// * `b` - Symmetric positive definite matrix B (lower triangle used)
/// * `ldb` - Leading dimension of B
/// * `w` - Output: eigenvalues in ascending order (length n)
/// * `v` - Output: eigenvectors (columns) (n × n)
/// * `ldv` - Leading dimension of V
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * 1 if B is not positive definite
/// * 2 if EVD did not converge
///
/// # Safety
/// - `a`, `b` must point to valid n × n matrices
/// - `w` must point to array of length n
/// - `v` must point to pre-allocated n × n storage
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ssygv(
    layout: OblasLayout,
    n: c_int,
    a: *const f32,
    lda: c_int,
    b: *const f32,
    ldb: c_int,
    w: *mut f32,
    v: *mut f32,
    ldv: c_int,
) -> c_int {
    if n <= 0 || a.is_null() || b.is_null() || w.is_null() || v.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let lda_val = lda as usize;
    let ldb_val = ldb as usize;
    let ldv_val = ldv as usize;
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

    // Convert B to Mat
    let mut mat_b: Mat<f32> = Mat::zeros(n_val, n_val);
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * ldb_val + j
            } else {
                j * ldb_val + i
            };
            mat_b[(i, j)] = *b.add(idx);
        }
    }

    // Compute generalized EVD
    let gevd = match SymmetricGeneralizedEvd::compute(mat_a.as_ref(), mat_b.as_ref()) {
        Ok(g) => g,
        Err(oxiblas_lapack::evd::GeneralizedEvdError::BNotPositiveDefinite) => return 1,
        Err(oxiblas_lapack::evd::GeneralizedEvdError::NotConverged) => return 2,
        Err(_) => return OblasReturn::InvalidArg as c_int,
    };

    // Copy eigenvalues
    let eigenvalues = gevd.eigenvalues();
    for (i, &val) in eigenvalues.iter().enumerate() {
        *w.add(i) = val;
    }

    // Copy eigenvectors
    let eigenvectors = gevd.eigenvectors();
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * ldv_val + j
            } else {
                j * ldv_val + i
            };
            *v.add(idx) = eigenvectors[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DSYGV - Symmetric generalized eigenvalue problem (double precision)
// =============================================================================

/// Computes the generalized eigenvalues and eigenvectors for symmetric matrices (double).
///
/// # Safety
/// - `a`, `b` must point to valid n × n matrices
/// - `w` must point to array of length n
/// - `v` must point to pre-allocated n × n storage
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dsygv(
    layout: OblasLayout,
    n: c_int,
    a: *const f64,
    lda: c_int,
    b: *const f64,
    ldb: c_int,
    w: *mut f64,
    v: *mut f64,
    ldv: c_int,
) -> c_int {
    if n <= 0 || a.is_null() || b.is_null() || w.is_null() || v.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let lda_val = lda as usize;
    let ldb_val = ldb as usize;
    let ldv_val = ldv as usize;
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

    // Convert B to Mat
    let mut mat_b: Mat<f64> = Mat::zeros(n_val, n_val);
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * ldb_val + j
            } else {
                j * ldb_val + i
            };
            mat_b[(i, j)] = *b.add(idx);
        }
    }

    // Compute generalized EVD
    let gevd = match SymmetricGeneralizedEvd::compute(mat_a.as_ref(), mat_b.as_ref()) {
        Ok(g) => g,
        Err(oxiblas_lapack::evd::GeneralizedEvdError::BNotPositiveDefinite) => return 1,
        Err(oxiblas_lapack::evd::GeneralizedEvdError::NotConverged) => return 2,
        Err(_) => return OblasReturn::InvalidArg as c_int,
    };

    // Copy eigenvalues
    let eigenvalues = gevd.eigenvalues();
    for (i, &val) in eigenvalues.iter().enumerate() {
        *w.add(i) = val;
    }

    // Copy eigenvectors
    let eigenvectors = gevd.eigenvectors();
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * ldv_val + j
            } else {
                j * ldv_val + i
            };
            *v.add(idx) = eigenvectors[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// SGGEV - General generalized eigenvalue problem (single precision)
// =============================================================================

/// Computes generalized eigenvalues for general matrices.
///
/// Solves A * x = λ * B * x for general (non-symmetric) A and B.
///
/// # Arguments
/// * `layout` - Matrix layout
/// * `n` - Dimension of matrices
/// * `a` - General matrix A
/// * `lda` - Leading dimension of A
/// * `b` - General matrix B
/// * `ldb` - Leading dimension of B
/// * `wr` - Output: real parts of eigenvalues (length n)
/// * `wi` - Output: imaginary parts of eigenvalues (length n)
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * 1 if B is singular
/// * 2 if EVD did not converge
///
/// # Safety
/// - `a`, `b` must point to valid n × n matrices
/// - `wr`, `wi` must point to arrays of length n
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sggev(
    layout: OblasLayout,
    n: c_int,
    a: *const f32,
    lda: c_int,
    b: *const f32,
    ldb: c_int,
    wr: *mut f32,
    wi: *mut f32,
) -> c_int {
    if n <= 0 || a.is_null() || b.is_null() || wr.is_null() || wi.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let lda_val = lda as usize;
    let ldb_val = ldb as usize;
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

    // Convert B to Mat
    let mut mat_b: Mat<f32> = Mat::zeros(n_val, n_val);
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * ldb_val + j
            } else {
                j * ldb_val + i
            };
            mat_b[(i, j)] = *b.add(idx);
        }
    }

    // Compute generalized EVD
    let gevd = match GeneralizedEvd::compute(mat_a.as_ref(), mat_b.as_ref()) {
        Ok(g) => g,
        Err(oxiblas_lapack::evd::GeneralizedEvdError::Singular) => return 1,
        Err(oxiblas_lapack::evd::GeneralizedEvdError::NotConverged) => return 2,
        Err(_) => return OblasReturn::InvalidArg as c_int,
    };

    // Copy eigenvalues
    let real_parts = gevd.eigenvalues_real();
    let imag_parts = gevd.eigenvalues_imag();

    for i in 0..n_val {
        *wr.add(i) = real_parts[i];
        *wi.add(i) = imag_parts[i];
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DGGEV - General generalized eigenvalue problem (double precision)
// =============================================================================

/// Computes generalized eigenvalues for general matrices (double precision).
///
/// # Safety
/// - `a`, `b` must point to valid n × n matrices
/// - `wr`, `wi` must point to arrays of length n
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dggev(
    layout: OblasLayout,
    n: c_int,
    a: *const f64,
    lda: c_int,
    b: *const f64,
    ldb: c_int,
    wr: *mut f64,
    wi: *mut f64,
) -> c_int {
    if n <= 0 || a.is_null() || b.is_null() || wr.is_null() || wi.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let lda_val = lda as usize;
    let ldb_val = ldb as usize;
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

    // Convert B to Mat
    let mut mat_b: Mat<f64> = Mat::zeros(n_val, n_val);
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * ldb_val + j
            } else {
                j * ldb_val + i
            };
            mat_b[(i, j)] = *b.add(idx);
        }
    }

    // Compute generalized EVD
    let gevd = match GeneralizedEvd::compute(mat_a.as_ref(), mat_b.as_ref()) {
        Ok(g) => g,
        Err(oxiblas_lapack::evd::GeneralizedEvdError::Singular) => return 1,
        Err(oxiblas_lapack::evd::GeneralizedEvdError::NotConverged) => return 2,
        Err(_) => return OblasReturn::InvalidArg as c_int,
    };

    // Copy eigenvalues
    let real_parts = gevd.eigenvalues_real();
    let imag_parts = gevd.eigenvalues_imag();

    for i in 0..n_val {
        *wr.add(i) = real_parts[i];
        *wi.add(i) = imag_parts[i];
    }

    OblasReturn::Success as c_int
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dsygv_identity() {
        // A = I, B = I => eigenvalues are 1
        let a = [1.0f64, 0.0, 0.0, 1.0]; // column-major
        let b = [1.0f64, 0.0, 0.0, 1.0]; // column-major
        let mut w = [0.0f64; 2];
        let mut v = [0.0f64; 4];

        unsafe {
            let ret = oblas_dsygv(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                w.as_mut_ptr(),
                v.as_mut_ptr(),
                2,
            );
            assert_eq!(ret, 0);

            // Eigenvalues should be 1
            assert!((w[0] - 1.0).abs() < 1e-10);
            assert!((w[1] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dsygv_scaled() {
        // A = 2*I, B = I => eigenvalues are 2
        let a = [2.0f64, 0.0, 0.0, 2.0]; // column-major
        let b = [1.0f64, 0.0, 0.0, 1.0]; // column-major
        let mut w = [0.0f64; 2];
        let mut v = [0.0f64; 4];

        unsafe {
            let ret = oblas_dsygv(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                w.as_mut_ptr(),
                v.as_mut_ptr(),
                2,
            );
            assert_eq!(ret, 0);

            // Eigenvalues should be 2
            assert!((w[0] - 2.0).abs() < 1e-10);
            assert!((w[1] - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dsygv_general() {
        // A = [[2, 1], [1, 2]], B = I => eigenvalues 1 and 3
        let a = [2.0f64, 1.0, 1.0, 2.0]; // column-major
        let b = [1.0f64, 0.0, 0.0, 1.0]; // column-major
        let mut w = [0.0f64; 2];
        let mut v = [0.0f64; 4];

        unsafe {
            let ret = oblas_dsygv(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                w.as_mut_ptr(),
                v.as_mut_ptr(),
                2,
            );
            assert_eq!(ret, 0);

            // Eigenvalues should be 1 and 3 (sorted ascending)
            assert!((w[0] - 1.0).abs() < 1e-10, "Expected 1, got {}", w[0]);
            assert!((w[1] - 3.0).abs() < 1e-10, "Expected 3, got {}", w[1]);
        }
    }

    #[test]
    fn test_dsygv_b_not_spd() {
        // B is not positive definite
        let a = [1.0f64, 0.0, 0.0, 1.0]; // column-major
        let b = [-1.0f64, 0.0, 0.0, 1.0]; // column-major (not SPD)
        let mut w = [0.0f64; 2];
        let mut v = [0.0f64; 4];

        unsafe {
            let ret = oblas_dsygv(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                w.as_mut_ptr(),
                v.as_mut_ptr(),
                2,
            );
            assert_eq!(ret, 1); // B not positive definite
        }
    }

    #[test]
    fn test_ssygv_basic() {
        let a = [1.0f32, 0.0, 0.0, 1.0]; // column-major
        let b = [1.0f32, 0.0, 0.0, 1.0]; // column-major
        let mut w = [0.0f32; 2];
        let mut v = [0.0f32; 4];

        unsafe {
            let ret = oblas_ssygv(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                w.as_mut_ptr(),
                v.as_mut_ptr(),
                2,
            );
            assert_eq!(ret, 0);
            assert!((w[0] - 1.0).abs() < 1e-5);
            assert!((w[1] - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dggev_diagonal() {
        // A = diag(2, 3), B = I => eigenvalues 2 and 3
        let a = [2.0f64, 0.0, 0.0, 3.0]; // column-major
        let b = [1.0f64, 0.0, 0.0, 1.0]; // column-major
        let mut wr = [0.0f64; 2];
        let mut wi = [0.0f64; 2];

        unsafe {
            let ret = oblas_dggev(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                wr.as_mut_ptr(),
                wi.as_mut_ptr(),
            );
            assert_eq!(ret, 0);

            // Sort eigenvalues
            let mut eigs = wr.to_vec();
            eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());

            assert!((eigs[0] - 2.0).abs() < 1e-10, "Expected 2, got {}", eigs[0]);
            assert!((eigs[1] - 3.0).abs() < 1e-10, "Expected 3, got {}", eigs[1]);

            // All eigenvalues should be real
            assert!(wi[0].abs() < 1e-10);
            assert!(wi[1].abs() < 1e-10);
        }
    }

    #[test]
    fn test_dggev_upper_triangular() {
        // A = [[1, 2], [0, 3]], B = I => eigenvalues 1 and 3
        let a = [1.0f64, 0.0, 2.0, 3.0]; // column-major
        let b = [1.0f64, 0.0, 0.0, 1.0]; // column-major
        let mut wr = [0.0f64; 2];
        let mut wi = [0.0f64; 2];

        unsafe {
            let ret = oblas_dggev(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                wr.as_mut_ptr(),
                wi.as_mut_ptr(),
            );
            assert_eq!(ret, 0);

            let mut eigs = wr.to_vec();
            eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());

            assert!((eigs[0] - 1.0).abs() < 1e-10);
            assert!((eigs[1] - 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sggev_basic() {
        let a = [2.0f32, 0.0, 0.0, 3.0]; // column-major
        let b = [1.0f32, 0.0, 0.0, 1.0]; // column-major
        let mut wr = [0.0f32; 2];
        let mut wi = [0.0f32; 2];

        unsafe {
            let ret = oblas_sggev(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                wr.as_mut_ptr(),
                wi.as_mut_ptr(),
            );
            assert_eq!(ret, 0);

            let mut eigs = wr.to_vec();
            eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());

            assert!((eigs[0] - 2.0).abs() < 1e-5);
            assert!((eigs[1] - 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dsygv_verify_eigenvectors() {
        // A = [[5, 2], [2, 3]], B = [[2, 0], [0, 2]]
        let a = [5.0f64, 2.0, 2.0, 3.0]; // column-major
        let b = [2.0f64, 0.0, 0.0, 2.0]; // column-major
        let mut w = [0.0f64; 2];
        let mut v = [0.0f64; 4];

        unsafe {
            let ret = oblas_dsygv(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                w.as_mut_ptr(),
                v.as_mut_ptr(),
                2,
            );
            assert_eq!(ret, 0);

            // Verify A*v = λ*B*v for each eigenpair
            for k in 0..2 {
                let lambda = w[k];
                let v0 = v[k * 2];
                let v1 = v[k * 2 + 1];

                // A*v (column-major: A = [[a[0], a[2]], [a[1], a[3]]])
                let av0 = a[0] * v0 + a[2] * v1;
                let av1 = a[1] * v0 + a[3] * v1;

                // λ*B*v
                let lbv0 = lambda * (b[0] * v0 + b[2] * v1);
                let lbv1 = lambda * (b[1] * v0 + b[3] * v1);

                assert!(
                    (av0 - lbv0).abs() < 1e-8,
                    "A*v[0] = {}, λ*B*v[0] = {}",
                    av0,
                    lbv0
                );
                assert!(
                    (av1 - lbv1).abs() < 1e-8,
                    "A*v[1] = {}, λ*B*v[1] = {}",
                    av1,
                    lbv1
                );
            }
        }
    }
}
