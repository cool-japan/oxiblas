//! LAPACK FFI - Schur decomposition routines.
//!
//! Schur decomposition computes A = Q T Q^T where Q is orthogonal
//! and T is quasi-upper triangular (real Schur form).

use crate::types::*;
use oxiblas_lapack::evd::Schur;
use oxiblas_matrix::Mat;
use std::ffi::c_int;

// =============================================================================
// SGEES - Schur decomposition (single precision)
// =============================================================================

/// Computes the Schur decomposition A = Q T Q^T.
///
/// For a real n×n matrix A, computes:
/// - T: quasi-upper triangular (real Schur form)
/// - Q: orthogonal matrix (Schur vectors)
///
/// The quasi-upper triangular matrix T has real eigenvalues on the diagonal
/// and 2×2 blocks for complex conjugate eigenvalue pairs.
///
/// # Arguments
/// * `layout` - Matrix layout
/// * `n` - Dimension of matrix A
/// * `a` - Input matrix A (on output, contains T)
/// * `lda` - Leading dimension of A
/// * `q` - Output orthogonal matrix Q
/// * `ldq` - Leading dimension of Q
/// * `wr` - Array to store real parts of eigenvalues (length n)
/// * `wi` - Array to store imaginary parts of eigenvalues (length n)
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * 1 if decomposition failed to converge
///
/// # Safety
/// - `a` must point to a valid n × n matrix
/// - `q` must point to pre-allocated n × n storage
/// - `wr` and `wi` must point to arrays of length n
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sgees(
    layout: OblasLayout,
    n: c_int,
    a: *mut f32,
    lda: c_int,
    q: *mut f32,
    ldq: c_int,
    wr: *mut f32,
    wi: *mut f32,
) -> c_int {
    if n <= 0 || a.is_null() || q.is_null() || wr.is_null() || wi.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let lda_val = lda as usize;
    let ldq_val = ldq as usize;
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

    // Compute Schur decomposition
    let schur = match Schur::compute(mat_a.as_ref()) {
        Ok(s) => s,
        Err(_) => return 1,
    };

    // Copy T back to A
    let t = schur.t();
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            *a.add(idx) = t[(i, j)];
        }
    }

    // Copy Q
    let q_mat = schur.q();
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * ldq_val + j
            } else {
                j * ldq_val + i
            };
            *q.add(idx) = q_mat[(i, j)];
        }
    }

    // Copy eigenvalues
    let eigenvalues = schur.eigenvalues();
    for (i, ev) in eigenvalues.iter().enumerate() {
        *wr.add(i) = ev.real;
        *wi.add(i) = ev.imag;
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DGEES - Schur decomposition (double precision)
// =============================================================================

/// Computes the Schur decomposition A = Q T Q^T (double precision).
///
/// # Safety
/// - `a` must point to a valid n × n matrix
/// - `q` must point to pre-allocated n × n storage
/// - `wr` and `wi` must point to arrays of length n
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dgees(
    layout: OblasLayout,
    n: c_int,
    a: *mut f64,
    lda: c_int,
    q: *mut f64,
    ldq: c_int,
    wr: *mut f64,
    wi: *mut f64,
) -> c_int {
    if n <= 0 || a.is_null() || q.is_null() || wr.is_null() || wi.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let lda_val = lda as usize;
    let ldq_val = ldq as usize;
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

    // Compute Schur decomposition
    let schur = match Schur::compute(mat_a.as_ref()) {
        Ok(s) => s,
        Err(_) => return 1,
    };

    // Copy T back to A
    let t = schur.t();
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            *a.add(idx) = t[(i, j)];
        }
    }

    // Copy Q
    let q_mat = schur.q();
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * ldq_val + j
            } else {
                j * ldq_val + i
            };
            *q.add(idx) = q_mat[(i, j)];
        }
    }

    // Copy eigenvalues
    let eigenvalues = schur.eigenvalues();
    for (i, ev) in eigenvalues.iter().enumerate() {
        *wr.add(i) = ev.real;
        *wi.add(i) = ev.imag;
    }

    OblasReturn::Success as c_int
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dgees_upper_triangular() {
        // Upper triangular matrix - Schur form should be itself
        // A = [[1, 2], [0, 3]]
        let mut a = [1.0f64, 0.0, 2.0, 3.0]; // column-major
        let mut q = [0.0f64; 4];
        let mut wr = [0.0f64; 2];
        let mut wi = [0.0f64; 2];

        unsafe {
            let ret = oblas_dgees(
                OblasLayout::ColMajor,
                2,
                a.as_mut_ptr(),
                2,
                q.as_mut_ptr(),
                2,
                wr.as_mut_ptr(),
                wi.as_mut_ptr(),
            );
            assert_eq!(ret, 0);

            // Eigenvalues should be 1 and 3
            let sorted_wr = {
                let mut v = wr.to_vec();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v
            };
            assert!((sorted_wr[0] - 1.0).abs() < 1e-10);
            assert!((sorted_wr[1] - 3.0).abs() < 1e-10);

            // All eigenvalues should be real
            assert!(wi[0].abs() < 1e-10);
            assert!(wi[1].abs() < 1e-10);
        }
    }

    #[test]
    fn test_dgees_symmetric() {
        // Symmetric matrix
        // A = [[4, 2], [2, 1]]
        let mut a = [4.0f64, 2.0, 2.0, 1.0]; // column-major
        let mut q = [0.0f64; 4];
        let mut wr = [0.0f64; 2];
        let mut wi = [0.0f64; 2];

        unsafe {
            let ret = oblas_dgees(
                OblasLayout::ColMajor,
                2,
                a.as_mut_ptr(),
                2,
                q.as_mut_ptr(),
                2,
                wr.as_mut_ptr(),
                wi.as_mut_ptr(),
            );
            assert_eq!(ret, 0);

            // All eigenvalues should be real for symmetric matrix
            assert!(wi[0].abs() < 1e-10);
            assert!(wi[1].abs() < 1e-10);

            // Verify Q is orthogonal: Q^T Q = I
            let q00 = q[0];
            let q10 = q[1];
            let q01 = q[2];
            let q11 = q[3];

            // First column dot first column should be 1
            let dot1 = q00 * q00 + q10 * q10;
            assert!((dot1 - 1.0).abs() < 1e-10);

            // Second column dot second column should be 1
            let dot2 = q01 * q01 + q11 * q11;
            assert!((dot2 - 1.0).abs() < 1e-10);

            // First column dot second column should be 0
            let dot12 = q00 * q01 + q10 * q11;
            assert!(dot12.abs() < 1e-10);
        }
    }

    #[test]
    fn test_dgees_complex_eigenvalues() {
        // Matrix with complex eigenvalues
        // A = [[0, -1], [1, 0]] -> eigenvalues ±i
        let mut a = [0.0f64, 1.0, -1.0, 0.0]; // column-major
        let mut q = [0.0f64; 4];
        let mut wr = [0.0f64; 2];
        let mut wi = [0.0f64; 2];

        unsafe {
            let ret = oblas_dgees(
                OblasLayout::ColMajor,
                2,
                a.as_mut_ptr(),
                2,
                q.as_mut_ptr(),
                2,
                wr.as_mut_ptr(),
                wi.as_mut_ptr(),
            );
            assert_eq!(ret, 0);

            // Real parts should be zero
            assert!(wr[0].abs() < 1e-10);
            assert!(wr[1].abs() < 1e-10);

            // Imaginary parts should be ±1
            let sorted_wi = {
                let mut v = wi.to_vec();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v
            };
            assert!((sorted_wi[0] + 1.0).abs() < 1e-10);
            assert!((sorted_wi[1] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sgees_basic() {
        let mut a = [1.0f32, 0.0, 2.0, 3.0]; // column-major [[1,2],[0,3]]
        let mut q = [0.0f32; 4];
        let mut wr = [0.0f32; 2];
        let mut wi = [0.0f32; 2];

        unsafe {
            let ret = oblas_sgees(
                OblasLayout::ColMajor,
                2,
                a.as_mut_ptr(),
                2,
                q.as_mut_ptr(),
                2,
                wr.as_mut_ptr(),
                wi.as_mut_ptr(),
            );
            assert_eq!(ret, 0);

            // Eigenvalues should be 1 and 3
            let sorted_wr = {
                let mut v = wr.to_vec();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v
            };
            assert!((sorted_wr[0] - 1.0).abs() < 1e-5);
            assert!((sorted_wr[1] - 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dgees_reconstruction() {
        // Test that A = Q T Q^T for a symmetric matrix (easier reconstruction)
        // A = [[4, 2], [2, 1]] (symmetric)
        let a_orig = [4.0f64, 2.0, 2.0, 1.0]; // column-major
        let mut a = a_orig;
        let mut q = [0.0f64; 4];
        let mut wr = [0.0f64; 2];
        let mut wi = [0.0f64; 2];

        unsafe {
            let ret = oblas_dgees(
                OblasLayout::ColMajor,
                2,
                a.as_mut_ptr(),
                2,
                q.as_mut_ptr(),
                2,
                wr.as_mut_ptr(),
                wi.as_mut_ptr(),
            );
            assert_eq!(ret, 0);

            // T is now in a (column-major)
            // Compute Q T Q^T and compare with original A
            // Using column-major indexing: matrix[i,j] = array[j*n + i]

            // Q T first (column-major)
            let mut qt = [0.0f64; 4];
            for i in 0..2 {
                for j in 0..2 {
                    let mut sum = 0.0;
                    for k in 0..2 {
                        // Q[i,k] * T[k,j]
                        sum += q[k * 2 + i] * a[j * 2 + k];
                    }
                    qt[j * 2 + i] = sum;
                }
            }

            // (Q T) Q^T (column-major)
            let mut qtqt = [0.0f64; 4];
            for i in 0..2 {
                for j in 0..2 {
                    let mut sum = 0.0;
                    for k in 0..2 {
                        // QT[i,k] * Q^T[k,j] = QT[i,k] * Q[j,k]
                        sum += qt[k * 2 + i] * q[k * 2 + j];
                    }
                    qtqt[j * 2 + i] = sum;
                }
            }

            // Compare with original - use reasonable tolerance
            for i in 0..4 {
                assert!(
                    (qtqt[i] - a_orig[i]).abs() < 1e-8,
                    "Reconstruction failed at index {}: got {}, expected {}",
                    i,
                    qtqt[i],
                    a_orig[i]
                );
            }
        }
    }
}
