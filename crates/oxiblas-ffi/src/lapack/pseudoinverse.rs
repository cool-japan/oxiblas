//! LAPACK FFI - Pseudoinverse (Moore-Penrose) computation routines.

use crate::types::*;
use oxiblas_lapack::utils::{pinv, pinv_default};
use oxiblas_matrix::Mat;
use std::ffi::c_int;

// =============================================================================
// SPINV - Pseudoinverse (single precision)
// =============================================================================

/// Computes the Moore-Penrose pseudoinverse of a matrix A (single precision).
///
/// Uses SVD decomposition: A^+ = V * Σ^+ * U^T
///
/// # Arguments
/// * `m` - Number of rows of matrix A
/// * `n` - Number of columns of matrix A
/// * `a` - The m x n matrix (input, not modified)
/// * `lda` - Leading dimension of A
/// * `tol` - Tolerance for determining numerical rank (singular values below this are treated as zero)
/// * `result` - Output n x m matrix for the pseudoinverse
/// * `ldr` - Leading dimension of result
/// * `rank` - Optional pointer to store the numerical rank (can be null)
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * >0 SVD did not converge
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `result` must point to storage for an n x m matrix
/// - `rank` may be null
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_spinv(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *const f32,
    lda: c_int,
    tol: f32,
    result: *mut f32,
    ldr: c_int,
    rank: *mut c_int,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || result.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m_val = m as usize;
    let n_val = n as usize;
    let lda_val = lda as usize;
    let ldr_val = ldr as usize;
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

    // Compute pseudoinverse
    match pinv(mat.as_ref(), tol) {
        Ok(pinv_result) => {
            // Write pseudoinverse (n x m) to result
            for i in 0..n_val {
                for j in 0..m_val {
                    let idx = if row_major {
                        i * ldr_val + j
                    } else {
                        j * ldr_val + i
                    };
                    *result.add(idx) = pinv_result.pinv[(i, j)];
                }
            }

            // Store rank if pointer is provided
            if !rank.is_null() {
                *rank = pinv_result.rank as c_int;
            }

            OblasReturn::Success as c_int
        }
        Err(_) => 1, // SVD did not converge
    }
}

// =============================================================================
// DPINV - Pseudoinverse (double precision)
// =============================================================================

/// Computes the Moore-Penrose pseudoinverse of a matrix A (double precision).
///
/// Uses SVD decomposition: A^+ = V * Σ^+ * U^T
///
/// # Arguments
/// * `m` - Number of rows of matrix A
/// * `n` - Number of columns of matrix A
/// * `a` - The m x n matrix (input, not modified)
/// * `lda` - Leading dimension of A
/// * `tol` - Tolerance for determining numerical rank (singular values below this are treated as zero)
/// * `result` - Output n x m matrix for the pseudoinverse
/// * `ldr` - Leading dimension of result
/// * `rank` - Optional pointer to store the numerical rank (can be null)
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * >0 SVD did not converge
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `result` must point to storage for an n x m matrix
/// - `rank` may be null
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dpinv(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *const f64,
    lda: c_int,
    tol: f64,
    result: *mut f64,
    ldr: c_int,
    rank: *mut c_int,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || result.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m_val = m as usize;
    let n_val = n as usize;
    let lda_val = lda as usize;
    let ldr_val = ldr as usize;
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

    // Compute pseudoinverse
    match pinv(mat.as_ref(), tol) {
        Ok(pinv_result) => {
            // Write pseudoinverse (n x m) to result
            for i in 0..n_val {
                for j in 0..m_val {
                    let idx = if row_major {
                        i * ldr_val + j
                    } else {
                        j * ldr_val + i
                    };
                    *result.add(idx) = pinv_result.pinv[(i, j)];
                }
            }

            // Store rank if pointer is provided
            if !rank.is_null() {
                *rank = pinv_result.rank as c_int;
            }

            OblasReturn::Success as c_int
        }
        Err(_) => 1, // SVD did not converge
    }
}

// =============================================================================
// SPINV_DEFAULT - Pseudoinverse with default tolerance (single precision)
// =============================================================================

/// Computes the Moore-Penrose pseudoinverse with automatic tolerance (single precision).
///
/// The tolerance is automatically computed as `eps * max(m, n) * σ_max`
/// where `eps` is machine epsilon and `σ_max` is the largest singular value.
///
/// # Arguments
/// * `m` - Number of rows of matrix A
/// * `n` - Number of columns of matrix A
/// * `a` - The m x n matrix (input, not modified)
/// * `lda` - Leading dimension of A
/// * `result` - Output n x m matrix for the pseudoinverse
/// * `ldr` - Leading dimension of result
/// * `rank` - Optional pointer to store the numerical rank (can be null)
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `result` must point to storage for an n x m matrix
/// - `rank` may be null
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_spinv_default(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *const f32,
    lda: c_int,
    result: *mut f32,
    ldr: c_int,
    rank: *mut c_int,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || result.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m_val = m as usize;
    let n_val = n as usize;
    let lda_val = lda as usize;
    let ldr_val = ldr as usize;
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

    // Compute pseudoinverse with default tolerance
    match pinv_default(mat.as_ref()) {
        Ok(pinv_result) => {
            // Write pseudoinverse (n x m) to result
            for i in 0..n_val {
                for j in 0..m_val {
                    let idx = if row_major {
                        i * ldr_val + j
                    } else {
                        j * ldr_val + i
                    };
                    *result.add(idx) = pinv_result.pinv[(i, j)];
                }
            }

            // Store rank if pointer is provided
            if !rank.is_null() {
                *rank = pinv_result.rank as c_int;
            }

            OblasReturn::Success as c_int
        }
        Err(_) => 1, // SVD did not converge
    }
}

// =============================================================================
// DPINV_DEFAULT - Pseudoinverse with default tolerance (double precision)
// =============================================================================

/// Computes the Moore-Penrose pseudoinverse with automatic tolerance (double precision).
///
/// The tolerance is automatically computed as `eps * max(m, n) * σ_max`
/// where `eps` is machine epsilon and `σ_max` is the largest singular value.
///
/// # Arguments
/// * `m` - Number of rows of matrix A
/// * `n` - Number of columns of matrix A
/// * `a` - The m x n matrix (input, not modified)
/// * `lda` - Leading dimension of A
/// * `result` - Output n x m matrix for the pseudoinverse
/// * `ldr` - Leading dimension of result
/// * `rank` - Optional pointer to store the numerical rank (can be null)
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `result` must point to storage for an n x m matrix
/// - `rank` may be null
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dpinv_default(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *const f64,
    lda: c_int,
    result: *mut f64,
    ldr: c_int,
    rank: *mut c_int,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || result.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m_val = m as usize;
    let n_val = n as usize;
    let lda_val = lda as usize;
    let ldr_val = ldr as usize;
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

    // Compute pseudoinverse with default tolerance
    match pinv_default(mat.as_ref()) {
        Ok(pinv_result) => {
            // Write pseudoinverse (n x m) to result
            for i in 0..n_val {
                for j in 0..m_val {
                    let idx = if row_major {
                        i * ldr_val + j
                    } else {
                        j * ldr_val + i
                    };
                    *result.add(idx) = pinv_result.pinv[(i, j)];
                }
            }

            // Store rank if pointer is provided
            if !rank.is_null() {
                *rank = pinv_result.rank as c_int;
            }

            OblasReturn::Success as c_int
        }
        Err(_) => 1, // SVD did not converge
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dpinv_square() {
        // Diagonal matrix: A = [[1, 0], [0, 2]] (column-major)
        // Pinv = [[1, 0], [0, 0.5]]
        let a = [1.0f64, 0.0, 0.0, 2.0];
        let mut result = [0.0f64; 4];
        let mut rank = 0i32;

        unsafe {
            let ret = oblas_dpinv(
                OblasLayout::ColMajor,
                2,
                2,
                a.as_ptr(),
                2,
                1e-10,
                result.as_mut_ptr(),
                2,
                &mut rank,
            );
            assert_eq!(ret, 0);
            assert_eq!(rank, 2);

            // Check diagonal values
            assert!((result[0] - 1.0).abs() < 1e-10);
            assert!(result[1].abs() < 1e-10);
            assert!(result[2].abs() < 1e-10);
            assert!((result[3] - 0.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dpinv_tall() {
        // Tall matrix (3×2): A = [[1, 0], [0, 1], [0, 0]] (column-major)
        // A is stored as [1, 0, 0, 0, 1, 0]
        let a = [1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut result = [0.0f64; 6]; // Pinv is 2×3
        let mut rank = 0i32;

        unsafe {
            let ret = oblas_dpinv(
                OblasLayout::ColMajor,
                3,
                2,
                a.as_ptr(),
                3,
                1e-10,
                result.as_mut_ptr(),
                2,
                &mut rank,
            );
            assert_eq!(ret, 0);
            assert_eq!(rank, 2);

            // Result should be 2×3 (column-major)
            // Expected: [[1, 0, 0], [0, 1, 0]] stored as [1, 0, 0, 1, 0, 0]
            assert!((result[0] - 1.0).abs() < 1e-10);
            assert!(result[1].abs() < 1e-10);
            assert!(result[2].abs() < 1e-10);
            assert!((result[3] - 1.0).abs() < 1e-10);
            assert!(result[4].abs() < 1e-10);
            assert!(result[5].abs() < 1e-10);
        }
    }

    #[test]
    fn test_dpinv_wide() {
        // Wide matrix (2×3): A = [[1, 0, 0], [0, 1, 0]] (column-major)
        // A is stored as [1, 0, 0, 1, 0, 0]
        let a = [1.0f64, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mut result = [0.0f64; 6]; // Pinv is 3×2
        let mut rank = 0i32;

        unsafe {
            let ret = oblas_dpinv(
                OblasLayout::ColMajor,
                2,
                3,
                a.as_ptr(),
                2,
                1e-10,
                result.as_mut_ptr(),
                3,
                &mut rank,
            );
            assert_eq!(ret, 0);
            assert_eq!(rank, 2);

            // Result should be 3×2 (column-major)
            // Expected: [[1, 0], [0, 1], [0, 0]] stored as [1, 0, 0, 0, 1, 0]
            assert!((result[0] - 1.0).abs() < 1e-10);
            assert!(result[1].abs() < 1e-10);
            assert!(result[2].abs() < 1e-10);
            assert!(result[3].abs() < 1e-10);
            assert!((result[4] - 1.0).abs() < 1e-10);
            assert!(result[5].abs() < 1e-10);
        }
    }

    #[test]
    fn test_dpinv_rank_deficient() {
        // Rank 1 matrix (2×2): A = [[1, 2], [2, 4]] (column-major)
        // A is stored as [1, 2, 2, 4]
        let a = [1.0f64, 2.0, 2.0, 4.0];
        let mut result = [0.0f64; 4];
        let mut rank = 0i32;

        unsafe {
            let ret = oblas_dpinv(
                OblasLayout::ColMajor,
                2,
                2,
                a.as_ptr(),
                2,
                1e-10,
                result.as_mut_ptr(),
                2,
                &mut rank,
            );
            assert_eq!(ret, 0);
            assert_eq!(rank, 1);

            // Verify A * A^+ * A = A
            // A^+ is stored in result (column-major)
            // Compute A * A^+
            let mut aa_pinv = [0.0f64; 4];
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        aa_pinv[j * 2 + i] += a[k * 2 + i] * result[j * 2 + k];
                    }
                }
            }

            // Compute (A * A^+) * A
            let mut product = [0.0f64; 4];
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        product[j * 2 + i] += aa_pinv[k * 2 + i] * a[j * 2 + k];
                    }
                }
            }

            // Check A * A^+ * A ≈ A
            for idx in 0..4 {
                assert!(
                    (product[idx] - a[idx]).abs() < 1e-9,
                    "product[{}] = {}, a[{}] = {}",
                    idx,
                    product[idx],
                    idx,
                    a[idx]
                );
            }
        }
    }

    #[test]
    fn test_dpinv_default() {
        // Test with automatic tolerance
        let a = [1.0f64, 0.0, 0.0, 2.0];
        let mut result = [0.0f64; 4];
        let mut rank = 0i32;

        unsafe {
            let ret = oblas_dpinv_default(
                OblasLayout::ColMajor,
                2,
                2,
                a.as_ptr(),
                2,
                result.as_mut_ptr(),
                2,
                &mut rank,
            );
            assert_eq!(ret, 0);
            assert_eq!(rank, 2);
        }
    }

    #[test]
    fn test_spinv_square() {
        let a = [1.0f32, 0.0, 0.0, 2.0];
        let mut result = [0.0f32; 4];
        let mut rank = 0i32;

        unsafe {
            let ret = oblas_spinv(
                OblasLayout::ColMajor,
                2,
                2,
                a.as_ptr(),
                2,
                1e-5,
                result.as_mut_ptr(),
                2,
                &mut rank,
            );
            assert_eq!(ret, 0);
            assert_eq!(rank, 2);
            assert!((result[0] - 1.0).abs() < 1e-5);
            assert!((result[3] - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dpinv_null_rank() {
        // Test that null rank pointer works
        let a = [1.0f64, 0.0, 0.0, 2.0];
        let mut result = [0.0f64; 4];

        unsafe {
            let ret = oblas_dpinv(
                OblasLayout::ColMajor,
                2,
                2,
                a.as_ptr(),
                2,
                1e-10,
                result.as_mut_ptr(),
                2,
                std::ptr::null_mut(),
            );
            assert_eq!(ret, 0);
        }
    }
}
