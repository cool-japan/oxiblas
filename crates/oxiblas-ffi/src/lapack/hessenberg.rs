//! LAPACK FFI - Hessenberg reduction routines.
//!
//! Hessenberg reduction computes A = Q H Q^T where Q is orthogonal
//! and H is upper Hessenberg (zeros below subdiagonal).

use crate::types::*;
use oxiblas_lapack::evd::Hessenberg;
use oxiblas_matrix::Mat;
use std::ffi::c_int;

// =============================================================================
// SGEHRD - Hessenberg reduction (single precision)
// =============================================================================

/// Reduces a general matrix to upper Hessenberg form.
///
/// Computes A = Q H Q^T where Q is orthogonal and H is upper Hessenberg.
///
/// # Arguments
/// * `layout` - Matrix layout
/// * `n` - Dimension of matrix A
/// * `a` - On input: matrix A. On output: upper Hessenberg matrix H
/// * `lda` - Leading dimension of A
/// * `q` - Output orthogonal matrix Q (may be NULL if not needed)
/// * `ldq` - Leading dimension of Q (ignored if q is NULL)
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * 1 if reduction failed
///
/// # Safety
/// - `a` must point to a valid n × n matrix
/// - `q` must point to pre-allocated n × n storage, or be NULL
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sgehrd(
    layout: OblasLayout,
    n: c_int,
    a: *mut f32,
    lda: c_int,
    q: *mut f32,
    ldq: c_int,
) -> c_int {
    if n <= 0 || a.is_null() {
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

    // Compute Hessenberg reduction
    let hess = match Hessenberg::compute(mat_a.as_ref()) {
        Ok(h) => h,
        Err(_) => return 1,
    };

    // Copy H back to A
    let h = hess.h();
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            *a.add(idx) = h[(i, j)];
        }
    }

    // Copy Q if requested
    if !q.is_null() {
        let ldq_val = ldq as usize;
        let q_mat = hess.q();
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
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DGEHRD - Hessenberg reduction (double precision)
// =============================================================================

/// Reduces a general matrix to upper Hessenberg form (double precision).
///
/// Computes A = Q H Q^T where Q is orthogonal and H is upper Hessenberg.
///
/// # Safety
/// - `a` must point to a valid n × n matrix
/// - `q` must point to pre-allocated n × n storage, or be NULL
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dgehrd(
    layout: OblasLayout,
    n: c_int,
    a: *mut f64,
    lda: c_int,
    q: *mut f64,
    ldq: c_int,
) -> c_int {
    if n <= 0 || a.is_null() {
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

    // Compute Hessenberg reduction
    let hess = match Hessenberg::compute(mat_a.as_ref()) {
        Ok(h) => h,
        Err(_) => return 1,
    };

    // Copy H back to A
    let h = hess.h();
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            *a.add(idx) = h[(i, j)];
        }
    }

    // Copy Q if requested
    if !q.is_null() {
        let ldq_val = ldq as usize;
        let q_mat = hess.q();
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
    }

    OblasReturn::Success as c_int
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dgehrd_3x3() {
        // A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        let mut a = [1.0f64, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]; // column-major
        let mut q = [0.0f64; 9];

        unsafe {
            let ret = oblas_dgehrd(
                OblasLayout::ColMajor,
                3,
                a.as_mut_ptr(),
                3,
                q.as_mut_ptr(),
                3,
            );
            assert_eq!(ret, 0);

            // H should be upper Hessenberg (zero below subdiagonal)
            // In column-major: H[2,0] = a[2] (index 2)
            assert!(a[2].abs() < 1e-10, "H[2,0] should be zero: {}", a[2]);
        }
    }

    #[test]
    fn test_dgehrd_4x4() {
        // 4x4 matrix
        let mut a = [
            4.0f64, 1.0, -2.0, 2.0, // col 0
            1.0, 2.0, 0.0, 1.0, // col 1
            -2.0, 0.0, 3.0, -2.0, // col 2
            2.0, 1.0, -2.0, -1.0, // col 3
        ];
        let mut q = [0.0f64; 16];

        unsafe {
            let ret = oblas_dgehrd(
                OblasLayout::ColMajor,
                4,
                a.as_mut_ptr(),
                4,
                q.as_mut_ptr(),
                4,
            );
            assert_eq!(ret, 0);

            // H should be upper Hessenberg
            // H[2,0] = a[2], H[3,0] = a[3], H[3,1] = a[7]
            assert!(a[2].abs() < 1e-10, "H[2,0] should be zero: {}", a[2]);
            assert!(a[3].abs() < 1e-10, "H[3,0] should be zero: {}", a[3]);
            assert!(a[7].abs() < 1e-10, "H[3,1] should be zero: {}", a[7]);
        }
    }

    #[test]
    fn test_dgehrd_q_orthogonal() {
        let mut a = [1.0f64, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]; // column-major
        let mut q = [0.0f64; 9];

        unsafe {
            let ret = oblas_dgehrd(
                OblasLayout::ColMajor,
                3,
                a.as_mut_ptr(),
                3,
                q.as_mut_ptr(),
                3,
            );
            assert_eq!(ret, 0);

            // Verify Q is orthogonal: Q^T Q = I (column-major)
            for i in 0..3 {
                for j in 0..3 {
                    let mut dot = 0.0;
                    for k in 0..3 {
                        // Q[k,i] * Q[k,j] in column-major: q[i*3+k] * q[j*3+k]
                        dot += q[i * 3 + k] * q[j * 3 + k];
                    }
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (dot - expected).abs() < 1e-10,
                        "Q^T*Q[{},{}] = {}, expected {}",
                        i,
                        j,
                        dot,
                        expected
                    );
                }
            }
        }
    }

    #[test]
    fn test_dgehrd_reconstruction() {
        // Original matrix A
        let a_orig = [1.0f64, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]; // column-major
        let mut a = a_orig;
        let mut q = [0.0f64; 9];

        unsafe {
            let ret = oblas_dgehrd(
                OblasLayout::ColMajor,
                3,
                a.as_mut_ptr(),
                3,
                q.as_mut_ptr(),
                3,
            );
            assert_eq!(ret, 0);

            // Reconstruct: A = Q H Q^T
            // First compute Q H (column-major)
            let mut qh = [0.0f64; 9];
            for i in 0..3 {
                for j in 0..3 {
                    let mut sum = 0.0;
                    for k in 0..3 {
                        // Q[i,k] * H[k,j]
                        sum += q[k * 3 + i] * a[j * 3 + k];
                    }
                    qh[j * 3 + i] = sum;
                }
            }

            // Then compute (Q H) Q^T
            let mut result = [0.0f64; 9];
            for i in 0..3 {
                for j in 0..3 {
                    let mut sum = 0.0;
                    for k in 0..3 {
                        // QH[i,k] * Q^T[k,j] = QH[i,k] * Q[j,k]
                        sum += qh[k * 3 + i] * q[k * 3 + j];
                    }
                    result[j * 3 + i] = sum;
                }
            }

            // Compare with original
            for i in 0..9 {
                assert!(
                    (result[i] - a_orig[i]).abs() < 1e-10,
                    "Reconstruction failed at index {}: got {}, expected {}",
                    i,
                    result[i],
                    a_orig[i]
                );
            }
        }
    }

    #[test]
    fn test_sgehrd_basic() {
        let mut a = [1.0f32, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]; // column-major
        let mut q = [0.0f32; 9];

        unsafe {
            let ret = oblas_sgehrd(
                OblasLayout::ColMajor,
                3,
                a.as_mut_ptr(),
                3,
                q.as_mut_ptr(),
                3,
            );
            assert_eq!(ret, 0);

            // H should be upper Hessenberg
            assert!(a[2].abs() < 1e-5, "H[2,0] should be zero: {}", a[2]);
        }
    }

    #[test]
    fn test_dgehrd_no_q() {
        // Test without computing Q
        let mut a = [1.0f64, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];

        unsafe {
            let ret = oblas_dgehrd(
                OblasLayout::ColMajor,
                3,
                a.as_mut_ptr(),
                3,
                std::ptr::null_mut(),
                0,
            );
            assert_eq!(ret, 0);

            // H should still be upper Hessenberg
            assert!(a[2].abs() < 1e-10, "H[2,0] should be zero: {}", a[2]);
        }
    }

    #[test]
    fn test_dgehrd_identity() {
        // Identity matrix should stay identity (already Hessenberg)
        let mut a = [1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut q = [0.0f64; 9];

        unsafe {
            let ret = oblas_dgehrd(
                OblasLayout::ColMajor,
                3,
                a.as_mut_ptr(),
                3,
                q.as_mut_ptr(),
                3,
            );
            assert_eq!(ret, 0);

            // Result should be identity
            for i in 0..3 {
                for j in 0..3 {
                    let idx = j * 3 + i;
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (a[idx] - expected).abs() < 1e-10,
                        "H[{},{}] = {}, expected {}",
                        i,
                        j,
                        a[idx],
                        expected
                    );
                }
            }
        }
    }

    #[test]
    fn test_dgehrd_row_major() {
        // Row-major layout test
        let mut a = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]; // row-major
        let mut q = [0.0f64; 9];

        unsafe {
            let ret = oblas_dgehrd(
                OblasLayout::RowMajor,
                3,
                a.as_mut_ptr(),
                3,
                q.as_mut_ptr(),
                3,
            );
            assert_eq!(ret, 0);

            // H[2,0] in row-major is at index 2*3 + 0 = 6
            assert!(a[6].abs() < 1e-10, "H[2,0] should be zero: {}", a[6]);
        }
    }
}
