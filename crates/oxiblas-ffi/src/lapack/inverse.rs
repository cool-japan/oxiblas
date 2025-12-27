//! LAPACK FFI - Matrix inverse routines.

use crate::types::*;
use oxiblas_lapack::utils::inv;
use oxiblas_matrix::Mat;
use std::ffi::c_int;
use std::slice;

// =============================================================================
// SGETRI - Matrix inverse from LU factorization (single precision)
// =============================================================================

/// Computes the inverse of a matrix using the LU factorization computed by
/// oblas_sgetrf.
///
/// # Safety
/// - `a` must point to a valid n x n matrix containing LU factors from sgetrf
/// - `ipiv` must point to the pivot array from sgetrf
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sgetri(
    layout: OblasLayout,
    n: c_int,
    a: *mut f32,
    lda: c_int,
    ipiv: *const c_int,
) -> c_int {
    if n <= 0 || a.is_null() || ipiv.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Read LU factors
    let mut lu = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            lu[i * n + j] = *a.add(idx);
        }
    }

    // Read pivot indices (convert from 1-based to 0-based)
    let ipiv_slice = slice::from_raw_parts(ipiv, n);
    let perm: Vec<usize> = ipiv_slice.iter().map(|&p| (p - 1) as usize).collect();

    // Compute inverse by solving AX = I for each column of identity
    // Using PA = LU, for each column: solve LUx = P*e_j
    let mut inv = vec![0.0f32; n * n];

    for col in 0..n {
        // Start with e_col (column of identity)
        let mut x = vec![0.0f32; n];
        x[col] = 1.0;

        // Apply permutations in order: swap rows k and perm[k]
        for k in 0..n {
            let pk = perm[k];
            if k != pk {
                x.swap(k, pk);
            }
        }

        // Forward substitution: Ly = Pb (L has unit diagonal)
        for k in 0..n {
            for i in (k + 1)..n {
                let mult = lu[i * n + k];
                x[i] -= mult * x[k];
            }
        }

        // Back substitution: Ux = y
        for k in (0..n).rev() {
            let diag = lu[k * n + k];
            if diag.abs() < f32::EPSILON {
                return OblasReturn::Singular as c_int;
            }
            x[k] /= diag;
            for i in 0..k {
                let mult = lu[i * n + k];
                x[i] -= mult * x[k];
            }
        }

        // Store column in inverse
        for i in 0..n {
            inv[i * n + col] = x[i];
        }
    }

    // Write result back
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            *a.add(idx) = inv[i * n + j];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DGETRI - Matrix inverse from LU factorization (double precision)
// =============================================================================

/// Computes the inverse of a matrix using the LU factorization computed by
/// oblas_dgetrf.
///
/// # Safety
/// - `a` must point to a valid n x n matrix containing LU factors from dgetrf
/// - `ipiv` must point to the pivot array from dgetrf
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dgetri(
    layout: OblasLayout,
    n: c_int,
    a: *mut f64,
    lda: c_int,
    ipiv: *const c_int,
) -> c_int {
    if n <= 0 || a.is_null() || ipiv.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Read LU factors
    let mut lu = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            lu[i * n + j] = *a.add(idx);
        }
    }

    // Read pivot indices (convert from 1-based to 0-based)
    let ipiv_slice = slice::from_raw_parts(ipiv, n);
    let perm: Vec<usize> = ipiv_slice.iter().map(|&p| (p - 1) as usize).collect();

    // Compute inverse by solving AX = I for each column of identity
    // Using PA = LU, for each column: solve LUx = P*e_j
    let mut inv = vec![0.0f64; n * n];

    for col in 0..n {
        // Start with e_col (column of identity)
        let mut x = vec![0.0f64; n];
        x[col] = 1.0;

        // Apply permutations in order: swap rows k and perm[k]
        for k in 0..n {
            let pk = perm[k];
            if k != pk {
                x.swap(k, pk);
            }
        }

        // Forward substitution: Ly = Pb (L has unit diagonal)
        for k in 0..n {
            for i in (k + 1)..n {
                let mult = lu[i * n + k];
                x[i] -= mult * x[k];
            }
        }

        // Back substitution: Ux = y
        for k in (0..n).rev() {
            let diag = lu[k * n + k];
            if diag.abs() < f64::EPSILON {
                return OblasReturn::Singular as c_int;
            }
            x[k] /= diag;
            for i in 0..k {
                let mult = lu[i * n + k];
                x[i] -= mult * x[k];
            }
        }

        // Store column in inverse
        for i in 0..n {
            inv[i * n + col] = x[i];
        }
    }

    // Write result back
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            *a.add(idx) = inv[i * n + j];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// SPOTRI - Inverse from Cholesky factorization (single precision)
// =============================================================================

/// Computes the inverse of a symmetric positive definite matrix using the
/// Cholesky factorization computed by oblas_spotrf.
///
/// # Safety
/// - `a` must point to a valid n x n matrix containing Cholesky factor from spotrf
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_spotri(
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

    // Read Cholesky factor
    let mut l = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            if upper {
                // Input is upper triangular (U), transpose to get L
                if j >= i {
                    l[j * n + i] = *a.add(idx);
                }
            } else {
                // Input is lower triangular (L)
                if i >= j {
                    l[i * n + j] = *a.add(idx);
                }
            }
        }
    }

    // Compute inverse by solving L L^T X = I for each column
    let mut inv = vec![0.0f32; n * n];

    for col in 0..n {
        // Start with e_col
        let mut x = vec![0.0f32; n];
        x[col] = 1.0;

        // Forward substitution: Ly = b
        for k in 0..n {
            let diag = l[k * n + k];
            if diag.abs() < f32::EPSILON {
                return OblasReturn::Singular as c_int;
            }
            x[k] /= diag;
            for i in (k + 1)..n {
                x[i] -= l[i * n + k] * x[k];
            }
        }

        // Back substitution: L^T x = y
        for k in (0..n).rev() {
            let diag = l[k * n + k];
            x[k] /= diag;
            for i in 0..k {
                x[i] -= l[k * n + i] * x[k];
            }
        }

        // Store column
        for i in 0..n {
            inv[i * n + col] = x[i];
        }
    }

    // Write result back (symmetric, so fill both triangles)
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            if upper {
                if j >= i {
                    *a.add(idx) = inv[i * n + j];
                }
            } else {
                if i >= j {
                    *a.add(idx) = inv[i * n + j];
                }
            }
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DPOTRI - Inverse from Cholesky factorization (double precision)
// =============================================================================

/// Computes the inverse of a symmetric positive definite matrix using the
/// Cholesky factorization computed by oblas_dpotrf.
///
/// # Safety
/// - `a` must point to a valid n x n matrix containing Cholesky factor from dpotrf
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dpotri(
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

    // Read Cholesky factor
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            if upper {
                // Input is upper triangular (U), transpose to get L
                if j >= i {
                    l[j * n + i] = *a.add(idx);
                }
            } else {
                // Input is lower triangular (L)
                if i >= j {
                    l[i * n + j] = *a.add(idx);
                }
            }
        }
    }

    // Compute inverse by solving L L^T X = I for each column
    let mut inv = vec![0.0f64; n * n];

    for col in 0..n {
        // Start with e_col
        let mut x = vec![0.0f64; n];
        x[col] = 1.0;

        // Forward substitution: Ly = b
        for k in 0..n {
            let diag = l[k * n + k];
            if diag.abs() < f64::EPSILON {
                return OblasReturn::Singular as c_int;
            }
            x[k] /= diag;
            for i in (k + 1)..n {
                x[i] -= l[i * n + k] * x[k];
            }
        }

        // Back substitution: L^T x = y
        for k in (0..n).rev() {
            let diag = l[k * n + k];
            x[k] /= diag;
            for i in 0..k {
                x[i] -= l[k * n + i] * x[k];
            }
        }

        // Store column
        for i in 0..n {
            inv[i * n + col] = x[i];
        }
    }

    // Write result back (symmetric, so fill both triangles)
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            if upper {
                if j >= i {
                    *a.add(idx) = inv[i * n + j];
                }
            } else {
                if i >= j {
                    *a.add(idx) = inv[i * n + j];
                }
            }
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// SINV - Direct matrix inverse (single precision)
// =============================================================================

/// Computes the inverse of a general square matrix directly.
///
/// This is a convenience function that performs LU factorization and
/// inverse computation in a single call.
///
/// # Arguments
/// * `layout` - Matrix layout
/// * `n` - Order of the matrix A
/// * `a` - On entry, the n x n matrix A. On exit, the inverse of A.
/// * `lda` - Leading dimension of A
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * 1 if matrix is singular
///
/// # Safety
/// - `a` must point to a valid n x n matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sinv(
    layout: OblasLayout,
    n: c_int,
    a: *mut f32,
    lda: c_int,
) -> c_int {
    if n <= 0 || a.is_null() {
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

    // Compute inverse
    match inv(mat.as_ref()) {
        Ok(inv_mat) => {
            // Write result back
            for i in 0..n_val {
                for j in 0..n_val {
                    let idx = if row_major {
                        i * lda_val + j
                    } else {
                        j * lda_val + i
                    };
                    *a.add(idx) = inv_mat[(i, j)];
                }
            }
            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::Singular as c_int,
    }
}

// =============================================================================
// DINV - Direct matrix inverse (double precision)
// =============================================================================

/// Computes the inverse of a general square matrix directly.
///
/// This is a convenience function that performs LU factorization and
/// inverse computation in a single call.
///
/// # Arguments
/// * `layout` - Matrix layout
/// * `n` - Order of the matrix A
/// * `a` - On entry, the n x n matrix A. On exit, the inverse of A.
/// * `lda` - Leading dimension of A
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * 1 if matrix is singular
///
/// # Safety
/// - `a` must point to a valid n x n matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dinv(
    layout: OblasLayout,
    n: c_int,
    a: *mut f64,
    lda: c_int,
) -> c_int {
    if n <= 0 || a.is_null() {
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

    // Compute inverse
    match inv(mat.as_ref()) {
        Ok(inv_mat) => {
            // Write result back
            for i in 0..n_val {
                for j in 0..n_val {
                    let idx = if row_major {
                        i * lda_val + j
                    } else {
                        j * lda_val + i
                    };
                    *a.add(idx) = inv_mat[(i, j)];
                }
            }
            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::Singular as c_int,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lapack::factorization::{oblas_dgetrf, oblas_dpotrf};

    #[test]
    fn test_dgetri() {
        // Matrix A = [[3, 1], [2, 4]] (column-major)
        // First compute LU, then inverse
        let original = [3.0f64, 2.0, 1.0, 4.0];
        let mut a = original.clone();
        let mut ipiv = [0i32; 2];

        unsafe {
            // First, compute LU factorization
            let result = oblas_dgetrf(
                OblasLayout::ColMajor,
                2,
                2,
                a.as_mut_ptr(),
                2,
                ipiv.as_mut_ptr(),
            );
            assert_eq!(result, 0);

            // Then compute inverse
            let result = oblas_dgetri(OblasLayout::ColMajor, 2, a.as_mut_ptr(), 2, ipiv.as_ptr());
            assert_eq!(result, 0);

            // Verify A * A^(-1) = I
            // A^(-1) is now in 'a' (column-major)
            // original = [[3, 1], [2, 4]] (column-major storage: [3, 2, 1, 4])
            let a_inv = a;
            let mut result_mat = [0.0f64; 4];

            // Matrix multiply
            for i in 0..2 {
                for j in 0..2 {
                    let mut sum = 0.0;
                    for k in 0..2 {
                        sum += original[k * 2 + i] * a_inv[j * 2 + k];
                    }
                    result_mat[j * 2 + i] = sum;
                }
            }

            // Should be identity
            assert!((result_mat[0] - 1.0).abs() < 1e-8);
            assert!(result_mat[1].abs() < 1e-8);
            assert!(result_mat[2].abs() < 1e-8);
            assert!((result_mat[3] - 1.0).abs() < 1e-8);
        }
    }

    #[test]
    fn test_dpotri() {
        // Matrix A = [[4, 2], [2, 5]] (column-major, symmetric positive definite)
        let original = [4.0f64, 2.0, 2.0, 5.0];
        let mut a = original.clone();

        unsafe {
            // First, compute Cholesky factorization
            let result = oblas_dpotrf(
                OblasLayout::ColMajor,
                OblasUplo::Lower,
                2,
                a.as_mut_ptr(),
                2,
            );
            assert_eq!(result, 0);

            // Then compute inverse
            let result = oblas_dpotri(
                OblasLayout::ColMajor,
                OblasUplo::Lower,
                2,
                a.as_mut_ptr(),
                2,
            );
            assert_eq!(result, 0);

            // Verify A * A^(-1) = I
            // For symmetric result, need to expand
            let a_inv_full = [a[0], a[1], a[1], a[3]]; // Lower triangular filled

            let mut result_mat = [0.0f64; 4];
            for i in 0..2 {
                for j in 0..2 {
                    let mut sum = 0.0;
                    for k in 0..2 {
                        sum += original[k * 2 + i] * a_inv_full[j * 2 + k];
                    }
                    result_mat[j * 2 + i] = sum;
                }
            }

            // Should be identity
            assert!((result_mat[0] - 1.0).abs() < 1e-8);
            assert!(result_mat[1].abs() < 1e-8);
            assert!(result_mat[2].abs() < 1e-8);
            assert!((result_mat[3] - 1.0).abs() < 1e-8);
        }
    }

    #[test]
    fn test_dinv_direct() {
        // Matrix A = [[3, 1], [2, 4]] (column-major)
        let original = [3.0f64, 2.0, 1.0, 4.0];
        let mut a = original.clone();

        unsafe {
            let result = oblas_dinv(OblasLayout::ColMajor, 2, a.as_mut_ptr(), 2);
            assert_eq!(result, 0);

            // Verify A * A^(-1) = I
            let mut result_mat = [0.0f64; 4];
            for i in 0..2 {
                for j in 0..2 {
                    let mut sum = 0.0;
                    for k in 0..2 {
                        sum += original[k * 2 + i] * a[j * 2 + k];
                    }
                    result_mat[j * 2 + i] = sum;
                }
            }

            // Should be identity
            assert!((result_mat[0] - 1.0).abs() < 1e-8);
            assert!(result_mat[1].abs() < 1e-8);
            assert!(result_mat[2].abs() < 1e-8);
            assert!((result_mat[3] - 1.0).abs() < 1e-8);
        }
    }

    #[test]
    fn test_sinv_direct() {
        // Matrix A = [[3, 1], [2, 4]] (column-major)
        let original = [3.0f32, 2.0, 1.0, 4.0];
        let mut a = original.clone();

        unsafe {
            let result = oblas_sinv(OblasLayout::ColMajor, 2, a.as_mut_ptr(), 2);
            assert_eq!(result, 0);

            // Verify A * A^(-1) = I
            let mut result_mat = [0.0f32; 4];
            for i in 0..2 {
                for j in 0..2 {
                    let mut sum = 0.0;
                    for k in 0..2 {
                        sum += original[k * 2 + i] * a[j * 2 + k];
                    }
                    result_mat[j * 2 + i] = sum;
                }
            }

            // Should be identity
            assert!((result_mat[0] - 1.0).abs() < 1e-5);
            assert!(result_mat[1].abs() < 1e-5);
            assert!(result_mat[2].abs() < 1e-5);
            assert!((result_mat[3] - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dinv_3x3() {
        // 3x3 matrix
        let original = [1.0f64, 0.0, 0.0, 2.0, 1.0, 0.0, 3.0, 2.0, 1.0]; // lower triangular
        let mut a = original.clone();

        unsafe {
            let result = oblas_dinv(OblasLayout::ColMajor, 3, a.as_mut_ptr(), 3);
            assert_eq!(result, 0);

            // Verify A * A^(-1) = I (spot check diagonal)
            let mut result_mat = [0.0f64; 9];
            for i in 0..3 {
                for j in 0..3 {
                    let mut sum = 0.0;
                    for k in 0..3 {
                        sum += original[k * 3 + i] * a[j * 3 + k];
                    }
                    result_mat[j * 3 + i] = sum;
                }
            }

            // Diagonal should be 1
            assert!((result_mat[0] - 1.0).abs() < 1e-8);
            assert!((result_mat[4] - 1.0).abs() < 1e-8);
            assert!((result_mat[8] - 1.0).abs() < 1e-8);
        }
    }

    #[test]
    fn test_dinv_singular() {
        // Singular matrix (rank deficient)
        let mut a = [1.0f64, 2.0, 2.0, 4.0]; // rows are multiples

        unsafe {
            let result = oblas_dinv(OblasLayout::ColMajor, 2, a.as_mut_ptr(), 2);
            assert_eq!(result, OblasReturn::Singular as c_int);
        }
    }
}
