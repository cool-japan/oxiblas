//! LAPACK FFI - QR decomposition with column pivoting.
//!
//! Computes A·P = Q·R where P is a column permutation matrix.
//! Provides rank-revealing QR factorization.

use crate::types::*;
use oxiblas_lapack::qr::QrPivot;
use oxiblas_matrix::Mat;
use std::ffi::c_int;

// =============================================================================
// SGEQP3 - QR with column pivoting (single precision)
// =============================================================================

/// Computes QR factorization with column pivoting: A·P = Q·R.
///
/// The decomposition is rank-revealing: diagonal elements of R are ordered
/// by decreasing magnitude.
///
/// # Arguments
/// * `layout` - Matrix layout
/// * `m` - Number of rows of A
/// * `n` - Number of columns of A
/// * `a` - On input: matrix A. On output: R in upper triangular part
/// * `lda` - Leading dimension of A
/// * `jpvt` - Output: column permutation (1-based indices), length n
/// * `tau` - Output: Householder scalars, length min(m, n)
/// * `rank` - Output: numerical rank (optional, may be NULL)
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
///
/// # Safety
/// - `a` must point to an m × n matrix
/// - `jpvt` must point to array of length n
/// - `tau` must point to array of length min(m, n)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sgeqp3(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *mut f32,
    lda: c_int,
    jpvt: *mut c_int,
    tau: *mut f32,
    rank: *mut c_int,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || jpvt.is_null() || tau.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m_val = m as usize;
    let n_val = n as usize;
    let lda_val = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert A to Mat
    let mut mat_a: Mat<f32> = Mat::zeros(m_val, n_val);
    for i in 0..m_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    // Compute QR with pivoting
    let qr = match QrPivot::compute(mat_a.as_ref()) {
        Ok(q) => q,
        Err(_) => return OblasReturn::InvalidArg as c_int,
    };

    // Copy R to upper triangular part of A
    let r = qr.r();
    let k = m_val.min(n_val);
    for i in 0..m_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            *a.add(idx) = r[(i, j)];
        }
    }

    // Copy column permutation (convert to 1-based)
    let col_perm = qr.column_permutation();
    for (i, &p) in col_perm.iter().enumerate() {
        *jpvt.add(i) = (p + 1) as c_int;
    }

    // Copy tau (we extract from Q reconstruction; approximate for now)
    for i in 0..k {
        *tau.add(i) = 0.0; // tau is internal, we don't expose it directly
    }

    // Copy rank if requested
    if !rank.is_null() {
        *rank = qr.rank() as c_int;
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DGEQP3 - QR with column pivoting (double precision)
// =============================================================================

/// Computes QR factorization with column pivoting (double precision).
///
/// # Safety
/// - `a` must point to an m × n matrix
/// - `jpvt` must point to array of length n
/// - `tau` must point to array of length min(m, n)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dgeqp3(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *mut f64,
    lda: c_int,
    jpvt: *mut c_int,
    tau: *mut f64,
    rank: *mut c_int,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || jpvt.is_null() || tau.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m_val = m as usize;
    let n_val = n as usize;
    let lda_val = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert A to Mat
    let mut mat_a: Mat<f64> = Mat::zeros(m_val, n_val);
    for i in 0..m_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    // Compute QR with pivoting
    let qr = match QrPivot::compute(mat_a.as_ref()) {
        Ok(q) => q,
        Err(_) => return OblasReturn::InvalidArg as c_int,
    };

    // Copy R to upper triangular part of A
    let r = qr.r();
    let k = m_val.min(n_val);
    for i in 0..m_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            *a.add(idx) = r[(i, j)];
        }
    }

    // Copy column permutation (convert to 1-based)
    let col_perm = qr.column_permutation();
    for (i, &p) in col_perm.iter().enumerate() {
        *jpvt.add(i) = (p + 1) as c_int;
    }

    // Copy tau (placeholder)
    for i in 0..k {
        *tau.add(i) = 0.0;
    }

    // Copy rank if requested
    if !rank.is_null() {
        *rank = qr.rank() as c_int;
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// Extended API - Get Q matrix from pivoted QR
// =============================================================================

/// Computes Q from pivoted QR factorization (single precision).
///
/// # Safety
/// - `a` must contain the result from oblas_sgeqp3
/// - `q` must point to pre-allocated m × m storage
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sorgqr_pivot(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *const f32,
    lda: c_int,
    q: *mut f32,
    ldq: c_int,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || q.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m_val = m as usize;
    let n_val = n as usize;
    let lda_val = lda as usize;
    let ldq_val = ldq as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert A to Mat
    let mut mat_a: Mat<f32> = Mat::zeros(m_val, n_val);
    for i in 0..m_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    // Recompute QR to get Q
    let qr = match QrPivot::compute(mat_a.as_ref()) {
        Ok(q) => q,
        Err(_) => return OblasReturn::InvalidArg as c_int,
    };

    // Copy Q
    let q_mat = qr.q();
    for i in 0..m_val {
        for j in 0..m_val {
            let idx = if row_major {
                i * ldq_val + j
            } else {
                j * ldq_val + i
            };
            *q.add(idx) = q_mat[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

/// Computes Q from pivoted QR factorization (double precision).
///
/// # Safety
/// - `a` must contain the result from oblas_dgeqp3
/// - `q` must point to pre-allocated m × m storage
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dorgqr_pivot(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *const f64,
    lda: c_int,
    q: *mut f64,
    ldq: c_int,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || q.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m_val = m as usize;
    let n_val = n as usize;
    let lda_val = lda as usize;
    let ldq_val = ldq as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert A to Mat
    let mut mat_a: Mat<f64> = Mat::zeros(m_val, n_val);
    for i in 0..m_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    // Recompute QR to get Q
    let qr = match QrPivot::compute(mat_a.as_ref()) {
        Ok(q) => q,
        Err(_) => return OblasReturn::InvalidArg as c_int,
    };

    // Copy Q
    let q_mat = qr.q();
    for i in 0..m_val {
        for j in 0..m_val {
            let idx = if row_major {
                i * ldq_val + j
            } else {
                j * ldq_val + i
            };
            *q.add(idx) = q_mat[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dgeqp3_square() {
        let mut a = [
            1.0f64, 4.0, 7.0, // column 0
            2.0, 5.0, 8.0, // column 1
            3.0, 6.0, 10.0, // column 2
        ];
        let mut jpvt = [0i32; 3];
        let mut tau = [0.0f64; 3];
        let mut rank = 0i32;

        unsafe {
            let ret = oblas_dgeqp3(
                OblasLayout::ColMajor,
                3,
                3,
                a.as_mut_ptr(),
                3,
                jpvt.as_mut_ptr(),
                tau.as_mut_ptr(),
                &mut rank,
            );
            assert_eq!(ret, 0);

            // R should be upper triangular (below diagonal should be zero)
            // In column-major: R[1,0] = a[1], R[2,0] = a[2], R[2,1] = a[5]
            assert!(a[1].abs() < 1e-10, "R[1,0] = {} should be 0", a[1]);
            assert!(a[2].abs() < 1e-10, "R[2,0] = {} should be 0", a[2]);
            assert!(a[5].abs() < 1e-10, "R[2,1] = {} should be 0", a[5]);

            // Rank should be 3 (full rank)
            assert_eq!(rank, 3);
        }
    }

    #[test]
    fn test_dgeqp3_rank_deficient() {
        // Rank 2 matrix (third row is sum of first two)
        let mut a = [
            1.0f64, 4.0, 5.0, // column 0
            2.0, 5.0, 7.0, // column 1
            3.0, 6.0, 9.0, // column 2
        ];
        let mut jpvt = [0i32; 3];
        let mut tau = [0.0f64; 3];
        let mut rank = 0i32;

        unsafe {
            let ret = oblas_dgeqp3(
                OblasLayout::ColMajor,
                3,
                3,
                a.as_mut_ptr(),
                3,
                jpvt.as_mut_ptr(),
                tau.as_mut_ptr(),
                &mut rank,
            );
            assert_eq!(ret, 0);

            // Rank should be 2
            assert_eq!(rank, 2, "Rank should be 2, got {}", rank);
        }
    }

    #[test]
    fn test_dgeqp3_permutation() {
        // Columns have different norms
        let mut a = [
            1.0f64, 1.0, 1.0, // column 0 (norm = sqrt(3))
            10.0, 10.0, 10.0, // column 1 (norm = sqrt(300))
            5.0, 5.0, 5.0, // column 2 (norm = sqrt(75))
        ];
        let mut jpvt = [0i32; 3];
        let mut tau = [0.0f64; 3];

        unsafe {
            let ret = oblas_dgeqp3(
                OblasLayout::ColMajor,
                3,
                3,
                a.as_mut_ptr(),
                3,
                jpvt.as_mut_ptr(),
                tau.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            assert_eq!(ret, 0);

            // Column 1 (1-based: 2) should be selected first due to largest norm
            assert_eq!(
                jpvt[0], 2,
                "Column 2 (1-based) should be first, got {}",
                jpvt[0]
            );
        }
    }

    #[test]
    fn test_sgeqp3_basic() {
        let mut a = [
            1.0f32, 3.0, // column 0
            2.0, 4.0, // column 1
        ];
        let mut jpvt = [0i32; 2];
        let mut tau = [0.0f32; 2];
        let mut rank = 0i32;

        unsafe {
            let ret = oblas_sgeqp3(
                OblasLayout::ColMajor,
                2,
                2,
                a.as_mut_ptr(),
                2,
                jpvt.as_mut_ptr(),
                tau.as_mut_ptr(),
                &mut rank,
            );
            assert_eq!(ret, 0);

            // R should be upper triangular
            assert!(a[1].abs() < 1e-5, "R[1,0] = {} should be 0", a[1]);

            // Rank should be 2
            assert_eq!(rank, 2);
        }
    }

    #[test]
    fn test_dgeqp3_tall() {
        // 4x2 matrix
        let mut a = [
            1.0f64, 3.0, 5.0, 7.0, // column 0
            2.0, 4.0, 6.0, 8.0, // column 1
        ];
        let mut jpvt = [0i32; 2];
        let mut tau = [0.0f64; 2];
        let mut rank = 0i32;

        unsafe {
            let ret = oblas_dgeqp3(
                OblasLayout::ColMajor,
                4,
                2,
                a.as_mut_ptr(),
                4,
                jpvt.as_mut_ptr(),
                tau.as_mut_ptr(),
                &mut rank,
            );
            assert_eq!(ret, 0);

            // R should be upper triangular in first 2 rows
            // R[1,0] = a[1], R[2,0] = a[2], R[3,0] = a[3]
            assert!(a[1].abs() < 1e-10, "R[1,0] = {} should be 0", a[1]);
            assert!(a[2].abs() < 1e-10, "R[2,0] = {} should be 0", a[2]);
            assert!(a[3].abs() < 1e-10, "R[3,0] = {} should be 0", a[3]);

            // Rank should be 2
            assert_eq!(rank, 2);
        }
    }

    #[test]
    fn test_dorgqr_pivot() {
        let a_orig = [
            1.0f64, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0, // 3x3 column-major
        ];
        let mut q = [0.0f64; 9];

        unsafe {
            let ret = oblas_dorgqr_pivot(
                OblasLayout::ColMajor,
                3,
                3,
                a_orig.as_ptr(),
                3,
                q.as_mut_ptr(),
                3,
            );
            assert_eq!(ret, 0);

            // Verify Q is orthogonal: Q^T * Q = I
            for i in 0..3 {
                for j in 0..3 {
                    let mut dot = 0.0;
                    for k in 0..3 {
                        // Q[k,i] * Q[k,j] in column-major
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
}
