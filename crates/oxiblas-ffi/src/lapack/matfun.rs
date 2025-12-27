//! LAPACK FFI - Matrix function routines.
//!
//! Provides C-compatible FFI for matrix functions:
//! - Matrix exponential (expm)
//! - Matrix logarithm (logm)
//! - Matrix square root (sqrtm)

use crate::types::*;
use oxiblas_lapack::utils::{expm, logm, sqrtm};
use oxiblas_matrix::Mat;
use std::ffi::c_int;

// =============================================================================
// SEXPM - Matrix exponential (single precision)
// =============================================================================

/// Computes the matrix exponential e^A (single precision).
///
/// Uses the Padé approximation with scaling and squaring method.
///
/// # Arguments
/// * `n` - Order of the square matrix A (n x n)
/// * `a` - The n x n matrix (input, not modified)
/// * `lda` - Leading dimension of A
/// * `result` - Output n x n matrix for e^A
/// * `ldr` - Leading dimension of result
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * 1 if computation failed
///
/// # Safety
/// - `a` must point to a valid n x n matrix
/// - `result` must point to storage for an n x n matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sexpm(
    layout: OblasLayout,
    n: c_int,
    a: *const f32,
    lda: c_int,
    result: *mut f32,
    ldr: c_int,
) -> c_int {
    if n <= 0 || a.is_null() || result.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let lda_val = lda as usize;
    let ldr_val = ldr as usize;
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

    // Compute matrix exponential
    match expm(mat.as_ref()) {
        Ok(exp_mat) => {
            // Write result
            for i in 0..n_val {
                for j in 0..n_val {
                    let idx = if row_major {
                        i * ldr_val + j
                    } else {
                        j * ldr_val + i
                    };
                    *result.add(idx) = exp_mat[(i, j)];
                }
            }
            OblasReturn::Success as c_int
        }
        Err(_) => 1,
    }
}

// =============================================================================
// DEXPM - Matrix exponential (double precision)
// =============================================================================

/// Computes the matrix exponential e^A (double precision).
///
/// Uses the Padé approximation with scaling and squaring method.
///
/// # Arguments
/// * `n` - Order of the square matrix A (n x n)
/// * `a` - The n x n matrix (input, not modified)
/// * `lda` - Leading dimension of A
/// * `result` - Output n x n matrix for e^A
/// * `ldr` - Leading dimension of result
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * 1 if computation failed
///
/// # Safety
/// - `a` must point to a valid n x n matrix
/// - `result` must point to storage for an n x n matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dexpm(
    layout: OblasLayout,
    n: c_int,
    a: *const f64,
    lda: c_int,
    result: *mut f64,
    ldr: c_int,
) -> c_int {
    if n <= 0 || a.is_null() || result.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let lda_val = lda as usize;
    let ldr_val = ldr as usize;
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

    // Compute matrix exponential
    match expm(mat.as_ref()) {
        Ok(exp_mat) => {
            // Write result
            for i in 0..n_val {
                for j in 0..n_val {
                    let idx = if row_major {
                        i * ldr_val + j
                    } else {
                        j * ldr_val + i
                    };
                    *result.add(idx) = exp_mat[(i, j)];
                }
            }
            OblasReturn::Success as c_int
        }
        Err(_) => 1,
    }
}

// =============================================================================
// SLOGM - Matrix logarithm (single precision)
// =============================================================================

/// Computes the principal matrix logarithm log(A) (single precision).
///
/// Uses the inverse scaling and squaring method.
/// The matrix A must have positive eigenvalues.
///
/// # Arguments
/// * `n` - Order of the square matrix A (n x n)
/// * `a` - The n x n matrix (input, not modified)
/// * `lda` - Leading dimension of A
/// * `result` - Output n x n matrix for log(A)
/// * `ldr` - Leading dimension of result
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * 1 if computation failed (e.g., matrix has non-positive eigenvalues)
///
/// # Safety
/// - `a` must point to a valid n x n matrix
/// - `result` must point to storage for an n x n matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_slogm(
    layout: OblasLayout,
    n: c_int,
    a: *const f32,
    lda: c_int,
    result: *mut f32,
    ldr: c_int,
) -> c_int {
    if n <= 0 || a.is_null() || result.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let lda_val = lda as usize;
    let ldr_val = ldr as usize;
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

    // Compute matrix logarithm
    match logm(mat.as_ref()) {
        Ok(log_mat) => {
            // Write result
            for i in 0..n_val {
                for j in 0..n_val {
                    let idx = if row_major {
                        i * ldr_val + j
                    } else {
                        j * ldr_val + i
                    };
                    *result.add(idx) = log_mat[(i, j)];
                }
            }
            OblasReturn::Success as c_int
        }
        Err(_) => 1,
    }
}

// =============================================================================
// DLOGM - Matrix logarithm (double precision)
// =============================================================================

/// Computes the principal matrix logarithm log(A) (double precision).
///
/// Uses the inverse scaling and squaring method.
/// The matrix A must have positive eigenvalues.
///
/// # Arguments
/// * `n` - Order of the square matrix A (n x n)
/// * `a` - The n x n matrix (input, not modified)
/// * `lda` - Leading dimension of A
/// * `result` - Output n x n matrix for log(A)
/// * `ldr` - Leading dimension of result
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * 1 if computation failed (e.g., matrix has non-positive eigenvalues)
///
/// # Safety
/// - `a` must point to a valid n x n matrix
/// - `result` must point to storage for an n x n matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dlogm(
    layout: OblasLayout,
    n: c_int,
    a: *const f64,
    lda: c_int,
    result: *mut f64,
    ldr: c_int,
) -> c_int {
    if n <= 0 || a.is_null() || result.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let lda_val = lda as usize;
    let ldr_val = ldr as usize;
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

    // Compute matrix logarithm
    match logm(mat.as_ref()) {
        Ok(log_mat) => {
            // Write result
            for i in 0..n_val {
                for j in 0..n_val {
                    let idx = if row_major {
                        i * ldr_val + j
                    } else {
                        j * ldr_val + i
                    };
                    *result.add(idx) = log_mat[(i, j)];
                }
            }
            OblasReturn::Success as c_int
        }
        Err(_) => 1,
    }
}

// =============================================================================
// SSQRTM - Matrix square root (single precision)
// =============================================================================

/// Computes the principal matrix square root A^(1/2) (single precision).
///
/// Uses the Denman-Beavers iteration.
///
/// # Arguments
/// * `n` - Order of the square matrix A (n x n)
/// * `a` - The n x n matrix (input, not modified)
/// * `lda` - Leading dimension of A
/// * `result` - Output n x n matrix for A^(1/2)
/// * `ldr` - Leading dimension of result
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * 1 if computation failed
///
/// # Safety
/// - `a` must point to a valid n x n matrix
/// - `result` must point to storage for an n x n matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ssqrtm(
    layout: OblasLayout,
    n: c_int,
    a: *const f32,
    lda: c_int,
    result: *mut f32,
    ldr: c_int,
) -> c_int {
    if n <= 0 || a.is_null() || result.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let lda_val = lda as usize;
    let ldr_val = ldr as usize;
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

    // Compute matrix square root
    match sqrtm(mat.as_ref()) {
        Ok(sqrt_mat) => {
            // Write result
            for i in 0..n_val {
                for j in 0..n_val {
                    let idx = if row_major {
                        i * ldr_val + j
                    } else {
                        j * ldr_val + i
                    };
                    *result.add(idx) = sqrt_mat[(i, j)];
                }
            }
            OblasReturn::Success as c_int
        }
        Err(_) => 1,
    }
}

// =============================================================================
// DSQRTM - Matrix square root (double precision)
// =============================================================================

/// Computes the principal matrix square root A^(1/2) (double precision).
///
/// Uses the Denman-Beavers iteration.
///
/// # Arguments
/// * `n` - Order of the square matrix A (n x n)
/// * `a` - The n x n matrix (input, not modified)
/// * `lda` - Leading dimension of A
/// * `result` - Output n x n matrix for A^(1/2)
/// * `ldr` - Leading dimension of result
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * 1 if computation failed
///
/// # Safety
/// - `a` must point to a valid n x n matrix
/// - `result` must point to storage for an n x n matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dsqrtm(
    layout: OblasLayout,
    n: c_int,
    a: *const f64,
    lda: c_int,
    result: *mut f64,
    ldr: c_int,
) -> c_int {
    if n <= 0 || a.is_null() || result.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let lda_val = lda as usize;
    let ldr_val = ldr as usize;
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

    // Compute matrix square root
    match sqrtm(mat.as_ref()) {
        Ok(sqrt_mat) => {
            // Write result
            for i in 0..n_val {
                for j in 0..n_val {
                    let idx = if row_major {
                        i * ldr_val + j
                    } else {
                        j * ldr_val + i
                    };
                    *result.add(idx) = sqrt_mat[(i, j)];
                }
            }
            OblasReturn::Success as c_int
        }
        Err(_) => 1,
    }
}

// =============================================================================
// SPOWM - Matrix power (single precision)
// =============================================================================

/// Computes the matrix power A^p for integer p (single precision).
///
/// Uses binary exponentiation (repeated squaring) for efficiency.
/// For p=0, returns identity matrix.
/// For p<0, computes (A^(-1))^|p|.
///
/// # Arguments
/// * `n` - Order of the square matrix A (n x n)
/// * `a` - The n x n matrix (input, not modified)
/// * `lda` - Leading dimension of A
/// * `p` - The integer exponent
/// * `result` - Output n x n matrix for A^p
/// * `ldr` - Leading dimension of result
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * 1 if computation failed (e.g., singular matrix for negative power)
///
/// # Safety
/// - `a` must point to a valid n x n matrix
/// - `result` must point to storage for an n x n matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_spowm(
    layout: OblasLayout,
    n: c_int,
    a: *const f32,
    lda: c_int,
    p: c_int,
    result: *mut f32,
    ldr: c_int,
) -> c_int {
    if n <= 0 || a.is_null() || result.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let lda_val = lda as usize;
    let ldr_val = ldr as usize;
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

    // Compute matrix power
    let pow_mat = match matrix_power(&mat, p) {
        Ok(m) => m,
        Err(_) => return 1,
    };

    // Write result
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * ldr_val + j
            } else {
                j * ldr_val + i
            };
            *result.add(idx) = pow_mat[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DPOWM - Matrix power (double precision)
// =============================================================================

/// Computes the matrix power A^p for integer p (double precision).
///
/// Uses binary exponentiation (repeated squaring) for efficiency.
/// For p=0, returns identity matrix.
/// For p<0, computes (A^(-1))^|p|.
///
/// # Safety
/// - `a` must point to a valid n x n matrix
/// - `result` must point to storage for an n x n matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dpowm(
    layout: OblasLayout,
    n: c_int,
    a: *const f64,
    lda: c_int,
    p: c_int,
    result: *mut f64,
    ldr: c_int,
) -> c_int {
    if n <= 0 || a.is_null() || result.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let lda_val = lda as usize;
    let ldr_val = ldr as usize;
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

    // Compute matrix power
    let pow_mat = match matrix_power(&mat, p) {
        Ok(m) => m,
        Err(_) => return 1,
    };

    // Write result
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * ldr_val + j
            } else {
                j * ldr_val + i
            };
            *result.add(idx) = pow_mat[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

/// Internal helper: Matrix power using binary exponentiation.
fn matrix_power<
    T: oxiblas_core::scalar::Field + oxiblas_core::scalar::Real + bytemuck::Zeroable,
>(
    a: &Mat<T>,
    p: c_int,
) -> Result<Mat<T>, ()> {
    let n = a.nrows();

    // Handle p = 0: return identity
    if p == 0 {
        return Ok(Mat::<T>::eye(n));
    }

    // Handle negative powers: compute inverse first
    let (base, power) = if p < 0 {
        let inv = oxiblas_lapack::utils::inv(a.as_ref()).map_err(|_| ())?;
        (inv, (-p) as u32)
    } else {
        let mut copy = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                copy[(i, j)] = a[(i, j)];
            }
        }
        (copy, p as u32)
    };

    // Binary exponentiation
    let mut result = Mat::<T>::eye(n);
    let mut current = base;
    let mut exp = power;

    while exp > 0 {
        if exp & 1 == 1 {
            result = mat_mult(&result, &current);
        }
        current = mat_mult(&current, &current);
        exp >>= 1;
    }

    Ok(result)
}

/// Helper: Matrix multiplication.
fn mat_mult<T: oxiblas_core::scalar::Field + bytemuck::Zeroable>(a: &Mat<T>, b: &Mat<T>) -> Mat<T> {
    let m = a.nrows();
    let n = b.ncols();
    let k = a.ncols();

    let mut c = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let mut sum = T::zero();
            for l in 0..k {
                sum = sum + a[(i, l)] * b[(l, j)];
            }
            c[(i, j)] = sum;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dexpm_identity() {
        // exp(I) = e * I
        let a = [1.0f64, 0.0, 0.0, 1.0]; // Identity (column-major)
        let mut result = [0.0f64; 4];

        unsafe {
            let ret = oblas_dexpm(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                result.as_mut_ptr(),
                2,
            );
            assert_eq!(ret, 0);

            let e = std::f64::consts::E;
            assert!((result[0] - e).abs() < 1e-10);
            assert!(result[1].abs() < 1e-10);
            assert!(result[2].abs() < 1e-10);
            assert!((result[3] - e).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dexpm_zero() {
        // exp(0) = I
        let a = [0.0f64; 4];
        let mut result = [0.0f64; 4];

        unsafe {
            let ret = oblas_dexpm(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                result.as_mut_ptr(),
                2,
            );
            assert_eq!(ret, 0);

            assert!((result[0] - 1.0).abs() < 1e-10);
            assert!(result[1].abs() < 1e-10);
            assert!(result[2].abs() < 1e-10);
            assert!((result[3] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dexpm_diagonal() {
        // A = [[1, 0], [0, 2]] (column-major)
        // exp(A) = [[e, 0], [0, e^2]]
        let a = [1.0f64, 0.0, 0.0, 2.0];
        let mut result = [0.0f64; 4];

        unsafe {
            let ret = oblas_dexpm(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                result.as_mut_ptr(),
                2,
            );
            assert_eq!(ret, 0);

            let e1 = std::f64::consts::E;
            let e2 = e1 * e1;
            assert!((result[0] - e1).abs() < 1e-10);
            assert!(result[1].abs() < 1e-10);
            assert!(result[2].abs() < 1e-10);
            assert!((result[3] - e2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dlogm_e_identity() {
        // log(e * I) = I
        let e = std::f64::consts::E;
        let a = [e, 0.0, 0.0, e]; // e * Identity (column-major)
        let mut result = [0.0f64; 4];

        unsafe {
            let ret = oblas_dlogm(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                result.as_mut_ptr(),
                2,
            );
            assert_eq!(ret, 0);

            // log(e*I) = I (with relaxed tolerance for iterative method)
            assert!((result[0] - 1.0).abs() < 1e-4);
            assert!(result[1].abs() < 1e-8);
            assert!(result[2].abs() < 1e-8);
            assert!((result[3] - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_dlogm_identity() {
        // log(I) = 0
        let a = [1.0f64, 0.0, 0.0, 1.0]; // Identity (column-major)
        let mut result = [0.0f64; 4];

        unsafe {
            let ret = oblas_dlogm(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                result.as_mut_ptr(),
                2,
            );
            assert_eq!(ret, 0);

            for i in 0..4 {
                assert!(result[i].abs() < 1e-8, "result[{}] = {}", i, result[i]);
            }
        }
    }

    #[test]
    fn test_dsqrtm_diagonal() {
        // sqrt([[4, 0], [0, 9]]) = [[2, 0], [0, 3]]
        let a = [4.0f64, 0.0, 0.0, 9.0]; // column-major
        let mut result = [0.0f64; 4];

        unsafe {
            let ret = oblas_dsqrtm(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                result.as_mut_ptr(),
                2,
            );
            assert_eq!(ret, 0);

            assert!((result[0] - 2.0).abs() < 1e-10);
            assert!(result[1].abs() < 1e-10);
            assert!(result[2].abs() < 1e-10);
            assert!((result[3] - 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dsqrtm_identity() {
        // sqrt(I) = I
        let a = [1.0f64, 0.0, 0.0, 1.0];
        let mut result = [0.0f64; 4];

        unsafe {
            let ret = oblas_dsqrtm(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                result.as_mut_ptr(),
                2,
            );
            assert_eq!(ret, 0);

            assert!((result[0] - 1.0).abs() < 1e-10);
            assert!(result[1].abs() < 1e-10);
            assert!(result[2].abs() < 1e-10);
            assert!((result[3] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dsqrtm_squared() {
        // (sqrt(A))^2 should equal A
        let a = [2.0f64, 0.5, 0.5, 2.0]; // symmetric positive definite
        let mut sqrt_a = [0.0f64; 4];

        unsafe {
            let ret = oblas_dsqrtm(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                sqrt_a.as_mut_ptr(),
                2,
            );
            assert_eq!(ret, 0);

            // Compute sqrt_a^2 (column-major matrix multiply)
            let mut a_back = [0.0f64; 4];
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        a_back[j * 2 + i] += sqrt_a[k * 2 + i] * sqrt_a[j * 2 + k];
                    }
                }
            }

            for idx in 0..4 {
                assert!(
                    (a[idx] - a_back[idx]).abs() < 1e-8,
                    "a[{}] = {}, a_back[{}] = {}",
                    idx,
                    a[idx],
                    idx,
                    a_back[idx]
                );
            }
        }
    }

    #[test]
    fn test_sexpm_identity() {
        let a = [1.0f32, 0.0, 0.0, 1.0];
        let mut result = [0.0f32; 4];

        unsafe {
            let ret = oblas_sexpm(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                result.as_mut_ptr(),
                2,
            );
            assert_eq!(ret, 0);

            let e = std::f32::consts::E;
            assert!((result[0] - e).abs() < 1e-5);
            assert!((result[3] - e).abs() < 1e-5);
        }
    }

    #[test]
    fn test_expm_logm_inverse() {
        // log(exp(A)) should be A for small A
        let a = [0.5f64, 0.1, 0.1, 0.3]; // small values
        let mut exp_a = [0.0f64; 4];
        let mut log_exp_a = [0.0f64; 4];

        unsafe {
            let ret = oblas_dexpm(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                exp_a.as_mut_ptr(),
                2,
            );
            assert_eq!(ret, 0);

            let ret = oblas_dlogm(
                OblasLayout::ColMajor,
                2,
                exp_a.as_ptr(),
                2,
                log_exp_a.as_mut_ptr(),
                2,
            );
            assert_eq!(ret, 0);

            // log(exp(A)) ≈ A (relaxed tolerance for iterative methods)
            for idx in 0..4 {
                assert!(
                    (a[idx] - log_exp_a[idx]).abs() < 1e-4,
                    "a[{}] = {}, log(exp(a))[{}] = {}",
                    idx,
                    a[idx],
                    idx,
                    log_exp_a[idx]
                );
            }
        }
    }

    #[test]
    fn test_dpowm_zero() {
        // A^0 = I for any A
        let a = [1.0f64, 2.0, 3.0, 4.0]; // column-major [[1,3],[2,4]]
        let mut result = [0.0f64; 4];

        unsafe {
            let ret = oblas_dpowm(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                0,
                result.as_mut_ptr(),
                2,
            );
            assert_eq!(ret, 0);

            // Result should be identity
            assert!((result[0] - 1.0).abs() < 1e-10);
            assert!(result[1].abs() < 1e-10);
            assert!(result[2].abs() < 1e-10);
            assert!((result[3] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dpowm_one() {
        // A^1 = A
        let a = [1.0f64, 2.0, 3.0, 4.0]; // column-major
        let mut result = [0.0f64; 4];

        unsafe {
            let ret = oblas_dpowm(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                1,
                result.as_mut_ptr(),
                2,
            );
            assert_eq!(ret, 0);

            for idx in 0..4 {
                assert!((result[idx] - a[idx]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_dpowm_two() {
        // A^2 = A * A
        // A = [[1, 3], [2, 4]] (column-major: [1, 2, 3, 4])
        // A^2 = [[7, 15], [10, 22]]
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let mut result = [0.0f64; 4];

        unsafe {
            let ret = oblas_dpowm(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                2,
                result.as_mut_ptr(),
                2,
            );
            assert_eq!(ret, 0);

            // Expected: column-major [[7, 15], [10, 22]] = [7, 10, 15, 22]
            assert!((result[0] - 7.0).abs() < 1e-10, "result[0] = {}", result[0]);
            assert!(
                (result[1] - 10.0).abs() < 1e-10,
                "result[1] = {}",
                result[1]
            );
            assert!(
                (result[2] - 15.0).abs() < 1e-10,
                "result[2] = {}",
                result[2]
            );
            assert!(
                (result[3] - 22.0).abs() < 1e-10,
                "result[3] = {}",
                result[3]
            );
        }
    }

    #[test]
    fn test_dpowm_three() {
        // A^3 = A^2 * A
        let a = [1.0f64, 2.0, 3.0, 4.0];
        let mut result = [0.0f64; 4];

        unsafe {
            let ret = oblas_dpowm(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                3,
                result.as_mut_ptr(),
                2,
            );
            assert_eq!(ret, 0);

            // A^2 = [[7, 15], [10, 22]]
            // A^3 = A^2 * A = [[37, 81], [54, 118]] (column-major: [37, 54, 81, 118])
            assert!((result[0] - 37.0).abs() < 1e-10);
            assert!((result[1] - 54.0).abs() < 1e-10);
            assert!((result[2] - 81.0).abs() < 1e-10);
            assert!((result[3] - 118.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dpowm_negative_one() {
        // A^(-1) = inverse of A
        // A = [[4, 7], [2, 6]] (column-major: [4, 2, 7, 6])
        // det(A) = 10, A^(-1) = [[0.6, -0.7], [-0.2, 0.4]]
        let a = [4.0f64, 2.0, 7.0, 6.0];
        let mut result = [0.0f64; 4];

        unsafe {
            let ret = oblas_dpowm(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                -1,
                result.as_mut_ptr(),
                2,
            );
            assert_eq!(ret, 0);

            // Column-major: [0.6, -0.2, -0.7, 0.4]
            assert!((result[0] - 0.6).abs() < 1e-10);
            assert!((result[1] + 0.2).abs() < 1e-10);
            assert!((result[2] + 0.7).abs() < 1e-10);
            assert!((result[3] - 0.4).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dpowm_negative_two() {
        // A^(-2) = (A^(-1))^2
        let a = [4.0f64, 2.0, 7.0, 6.0];
        let mut result = [0.0f64; 4];

        unsafe {
            let ret = oblas_dpowm(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                -2,
                result.as_mut_ptr(),
                2,
            );
            assert_eq!(ret, 0);

            // Compute manually: A^(-1) = [[0.6, -0.7], [-0.2, 0.4]]
            // (A^(-1))^2 = [[0.5, -0.7], [-0.2, 0.3]]
            // Actually: (A^(-1))^2 = [[0.6*0.6 + (-0.7)*(-0.2), 0.6*(-0.7) + (-0.7)*0.4],
            //                         [(-0.2)*0.6 + 0.4*(-0.2), (-0.2)*(-0.7) + 0.4*0.4]]
            //                      = [[0.36 + 0.14, -0.42 - 0.28], [-0.12 - 0.08, 0.14 + 0.16]]
            //                      = [[0.5, -0.7], [-0.2, 0.3]]
            // Column-major: [0.5, -0.2, -0.7, 0.3]
            assert!((result[0] - 0.5).abs() < 1e-10);
            assert!((result[1] + 0.2).abs() < 1e-10);
            assert!((result[2] + 0.7).abs() < 1e-10);
            assert!((result[3] - 0.3).abs() < 1e-10);
        }
    }

    #[test]
    fn test_spowm_two() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let mut result = [0.0f32; 4];

        unsafe {
            let ret = oblas_spowm(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                2,
                result.as_mut_ptr(),
                2,
            );
            assert_eq!(ret, 0);

            assert!((result[0] - 7.0).abs() < 1e-5);
            assert!((result[1] - 10.0).abs() < 1e-5);
            assert!((result[2] - 15.0).abs() < 1e-5);
            assert!((result[3] - 22.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dpowm_diagonal() {
        // For diagonal matrix, A^p has diagonals raised to p
        // A = [[2, 0], [0, 3]], A^3 = [[8, 0], [0, 27]]
        let a = [2.0f64, 0.0, 0.0, 3.0];
        let mut result = [0.0f64; 4];

        unsafe {
            let ret = oblas_dpowm(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                3,
                result.as_mut_ptr(),
                2,
            );
            assert_eq!(ret, 0);

            assert!((result[0] - 8.0).abs() < 1e-10);
            assert!(result[1].abs() < 1e-10);
            assert!(result[2].abs() < 1e-10);
            assert!((result[3] - 27.0).abs() < 1e-10);
        }
    }
}
