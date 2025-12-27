//! LAPACK FFI - LU decomposition with full pivoting.
//!
//! Full pivoting provides maximum numerical stability by selecting the
//! largest element in the remaining submatrix as the pivot.
//! PAQ = LU where P is row permutation and Q is column permutation.

use crate::types::*;
use oxiblas_lapack::lu::LuFullPiv;
use oxiblas_matrix::Mat;
use std::ffi::c_int;

// =============================================================================
// SGETC2 - LU with complete pivoting (single precision)
// =============================================================================

/// Computes LU factorization with complete (full) pivoting.
///
/// PAQ = LU where P is row permutation, Q is column permutation.
///
/// # Arguments
/// * `layout` - Matrix layout
/// * `n` - Dimension of matrix A
/// * `a` - On input: matrix A. On output: LU factors (L below diagonal, U on/above)
/// * `lda` - Leading dimension of A
/// * `ipiv` - Output: row pivot indices (1-based), length n
/// * `jpiv` - Output: column pivot indices (1-based), length n
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * positive value k if matrix is singular at index k-1
///
/// # Safety
/// - `a` must point to a valid n × n matrix
/// - `ipiv`, `jpiv` must point to arrays of length n
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sgetc2(
    layout: OblasLayout,
    n: c_int,
    a: *mut f32,
    lda: c_int,
    ipiv: *mut c_int,
    jpiv: *mut c_int,
) -> c_int {
    if n <= 0 || a.is_null() || ipiv.is_null() || jpiv.is_null() {
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

    // Compute LU with full pivoting
    let lu = match LuFullPiv::compute(mat_a.as_ref()) {
        Ok(l) => l,
        Err(oxiblas_lapack::lu::LuFullPivError::Singular { index }) => {
            return (index + 1) as c_int;
        }
        Err(_) => return OblasReturn::InvalidArg as c_int,
    };

    // Copy LU factors back to A
    let lu_mat = lu.lu_matrix();
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            *a.add(idx) = lu_mat[(i, j)];
        }
    }

    // Copy row pivots (convert to 1-based)
    let row_piv = lu.row_pivot();
    for (i, &p) in row_piv.iter().enumerate() {
        *ipiv.add(i) = (p + 1) as c_int;
    }

    // Copy column pivots (convert to 1-based)
    let col_piv = lu.col_pivot();
    for (i, &p) in col_piv.iter().enumerate() {
        *jpiv.add(i) = (p + 1) as c_int;
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DGETC2 - LU with complete pivoting (double precision)
// =============================================================================

/// Computes LU factorization with complete (full) pivoting (double precision).
///
/// # Safety
/// - `a` must point to a valid n × n matrix
/// - `ipiv`, `jpiv` must point to arrays of length n
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dgetc2(
    layout: OblasLayout,
    n: c_int,
    a: *mut f64,
    lda: c_int,
    ipiv: *mut c_int,
    jpiv: *mut c_int,
) -> c_int {
    if n <= 0 || a.is_null() || ipiv.is_null() || jpiv.is_null() {
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

    // Compute LU with full pivoting
    let lu = match LuFullPiv::compute(mat_a.as_ref()) {
        Ok(l) => l,
        Err(oxiblas_lapack::lu::LuFullPivError::Singular { index }) => {
            return (index + 1) as c_int;
        }
        Err(_) => return OblasReturn::InvalidArg as c_int,
    };

    // Copy LU factors back to A
    let lu_mat = lu.lu_matrix();
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            *a.add(idx) = lu_mat[(i, j)];
        }
    }

    // Copy row pivots (convert to 1-based)
    let row_piv = lu.row_pivot();
    for (i, &p) in row_piv.iter().enumerate() {
        *ipiv.add(i) = (p + 1) as c_int;
    }

    // Copy column pivots (convert to 1-based)
    let col_piv = lu.col_pivot();
    for (i, &p) in col_piv.iter().enumerate() {
        *jpiv.add(i) = (p + 1) as c_int;
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// Extended API - Solve from LU with full pivoting
// =============================================================================

/// Solves A*X = B using LU factors from full pivoting (single precision).
///
/// # Arguments
/// * `layout` - Matrix layout
/// * `n` - Dimension of matrix A
/// * `nrhs` - Number of right-hand side columns
/// * `a` - LU factors from oblas_sgetc2
/// * `lda` - Leading dimension of A
/// * `ipiv` - Row pivot indices from oblas_sgetc2
/// * `jpiv` - Column pivot indices from oblas_sgetc2
/// * `b` - On input: right-hand side B. On output: solution X
/// * `ldb` - Leading dimension of B
///
/// # Safety
/// - All pointers must be valid for their respective sizes
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sgesc2(
    layout: OblasLayout,
    n: c_int,
    nrhs: c_int,
    a: *const f32,
    lda: c_int,
    ipiv: *const c_int,
    jpiv: *const c_int,
    b: *mut f32,
    ldb: c_int,
) -> c_int {
    if n <= 0 || nrhs <= 0 || a.is_null() || ipiv.is_null() || jpiv.is_null() || b.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let nrhs_val = nrhs as usize;
    let lda_val = lda as usize;
    let ldb_val = ldb as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Read LU matrix
    let mut lu_mat: Mat<f32> = Mat::zeros(n_val, n_val);
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            lu_mat[(i, j)] = *a.add(idx);
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

    // Apply row permutations
    for k in 0..n_val {
        let pk = (*ipiv.add(k) - 1) as usize; // Convert from 1-based
        if k != pk {
            for j in 0..nrhs_val {
                let tmp = b_mat[(k, j)];
                b_mat[(k, j)] = b_mat[(pk, j)];
                b_mat[(pk, j)] = tmp;
            }
        }
    }

    // Forward substitution: L * z = Pb (L has unit diagonal)
    for k in 0..n_val {
        for i in (k + 1)..n_val {
            let mult = lu_mat[(i, k)];
            for j in 0..nrhs_val {
                b_mat[(i, j)] -= mult * b_mat[(k, j)];
            }
        }
    }

    // Back substitution: U * y = z
    for k in (0..n_val).rev() {
        let diag = lu_mat[(k, k)];
        for j in 0..nrhs_val {
            b_mat[(k, j)] /= diag;
        }
        for i in 0..k {
            let mult = lu_mat[(i, k)];
            for j in 0..nrhs_val {
                b_mat[(i, j)] -= mult * b_mat[(k, j)];
            }
        }
    }

    // Apply inverse column permutations
    let mut x_mat = b_mat.clone();
    for k in (0..n_val).rev() {
        let pk = (*jpiv.add(k) - 1) as usize; // Convert from 1-based
        if k != pk {
            for j in 0..nrhs_val {
                let tmp = x_mat[(k, j)];
                x_mat[(k, j)] = x_mat[(pk, j)];
                x_mat[(pk, j)] = tmp;
            }
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
            *b.add(idx) = x_mat[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

/// Solves A*X = B using LU factors from full pivoting (double precision).
///
/// # Safety
/// - All pointers must be valid for their respective sizes
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dgesc2(
    layout: OblasLayout,
    n: c_int,
    nrhs: c_int,
    a: *const f64,
    lda: c_int,
    ipiv: *const c_int,
    jpiv: *const c_int,
    b: *mut f64,
    ldb: c_int,
) -> c_int {
    if n <= 0 || nrhs <= 0 || a.is_null() || ipiv.is_null() || jpiv.is_null() || b.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let nrhs_val = nrhs as usize;
    let lda_val = lda as usize;
    let ldb_val = ldb as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Read LU matrix
    let mut lu_mat: Mat<f64> = Mat::zeros(n_val, n_val);
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            lu_mat[(i, j)] = *a.add(idx);
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

    // Apply row permutations
    for k in 0..n_val {
        let pk = (*ipiv.add(k) - 1) as usize;
        if k != pk {
            for j in 0..nrhs_val {
                let tmp = b_mat[(k, j)];
                b_mat[(k, j)] = b_mat[(pk, j)];
                b_mat[(pk, j)] = tmp;
            }
        }
    }

    // Forward substitution
    for k in 0..n_val {
        for i in (k + 1)..n_val {
            let mult = lu_mat[(i, k)];
            for j in 0..nrhs_val {
                b_mat[(i, j)] -= mult * b_mat[(k, j)];
            }
        }
    }

    // Back substitution
    for k in (0..n_val).rev() {
        let diag = lu_mat[(k, k)];
        for j in 0..nrhs_val {
            b_mat[(k, j)] /= diag;
        }
        for i in 0..k {
            let mult = lu_mat[(i, k)];
            for j in 0..nrhs_val {
                b_mat[(i, j)] -= mult * b_mat[(k, j)];
            }
        }
    }

    // Apply inverse column permutations
    let mut x_mat = b_mat.clone();
    for k in (0..n_val).rev() {
        let pk = (*jpiv.add(k) - 1) as usize;
        if k != pk {
            for j in 0..nrhs_val {
                let tmp = x_mat[(k, j)];
                x_mat[(k, j)] = x_mat[(pk, j)];
                x_mat[(pk, j)] = tmp;
            }
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
            *b.add(idx) = x_mat[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dgetc2_simple() {
        let mut a = [4.0f64, 6.0, 3.0, 3.0]; // column-major [[4,3],[6,3]]
        let mut ipiv = [0i32; 2];
        let mut jpiv = [0i32; 2];

        unsafe {
            let ret = oblas_dgetc2(
                OblasLayout::ColMajor,
                2,
                a.as_mut_ptr(),
                2,
                ipiv.as_mut_ptr(),
                jpiv.as_mut_ptr(),
            );
            assert_eq!(ret, 0);
        }
    }

    #[test]
    fn test_dgetc2_singular() {
        // Singular matrix [[1,2],[2,4]]
        let mut a = [1.0f64, 2.0, 2.0, 4.0]; // column-major
        let mut ipiv = [0i32; 2];
        let mut jpiv = [0i32; 2];

        unsafe {
            let ret = oblas_dgetc2(
                OblasLayout::ColMajor,
                2,
                a.as_mut_ptr(),
                2,
                ipiv.as_mut_ptr(),
                jpiv.as_mut_ptr(),
            );
            // Should return positive value indicating singularity
            assert!(
                ret > 0,
                "Expected positive return for singular matrix, got {}",
                ret
            );
        }
    }

    #[test]
    fn test_dgesc2_solve() {
        // A = [[2, 1], [4, 3]], b = [3, 7] => x = [1, 1]
        let mut a = [2.0f64, 4.0, 1.0, 3.0]; // column-major
        let mut ipiv = [0i32; 2];
        let mut jpiv = [0i32; 2];

        unsafe {
            let ret = oblas_dgetc2(
                OblasLayout::ColMajor,
                2,
                a.as_mut_ptr(),
                2,
                ipiv.as_mut_ptr(),
                jpiv.as_mut_ptr(),
            );
            assert_eq!(ret, 0);

            // Now solve
            let mut b = [3.0f64, 7.0]; // column-major, 1 rhs
            let ret2 = oblas_dgesc2(
                OblasLayout::ColMajor,
                2,
                1,
                a.as_ptr(),
                2,
                ipiv.as_ptr(),
                jpiv.as_ptr(),
                b.as_mut_ptr(),
                2,
            );
            assert_eq!(ret2, 0);

            // Solution should be approximately [1, 1]
            assert!((b[0] - 1.0).abs() < 1e-10, "x[0] = {}, expected 1.0", b[0]);
            assert!((b[1] - 1.0).abs() < 1e-10, "x[1] = {}, expected 1.0", b[1]);
        }
    }

    #[test]
    fn test_sgetc2_basic() {
        let mut a = [2.0f32, 4.0, 1.0, 3.0]; // column-major
        let mut ipiv = [0i32; 2];
        let mut jpiv = [0i32; 2];

        unsafe {
            let ret = oblas_sgetc2(
                OblasLayout::ColMajor,
                2,
                a.as_mut_ptr(),
                2,
                ipiv.as_mut_ptr(),
                jpiv.as_mut_ptr(),
            );
            assert_eq!(ret, 0);
        }
    }

    #[test]
    fn test_dgetc2_3x3() {
        // A = [[2, 1, 1], [4, 3, 3], [8, 7, 9]]
        let mut a = [
            2.0f64, 4.0, 8.0, // column 0
            1.0, 3.0, 7.0, // column 1
            1.0, 3.0, 9.0, // column 2
        ];
        let mut ipiv = [0i32; 3];
        let mut jpiv = [0i32; 3];

        unsafe {
            let ret = oblas_dgetc2(
                OblasLayout::ColMajor,
                3,
                a.as_mut_ptr(),
                3,
                ipiv.as_mut_ptr(),
                jpiv.as_mut_ptr(),
            );
            assert_eq!(ret, 0);
        }
    }

    #[test]
    fn test_dgesc2_solve_3x3() {
        // A = [[2, 1, 1], [4, 3, 3], [8, 7, 9]]
        // b = [4, 10, 24] => x = [1, 1, 1]
        let mut a = [
            2.0f64, 4.0, 8.0, // column 0
            1.0, 3.0, 7.0, // column 1
            1.0, 3.0, 9.0, // column 2
        ];
        let mut ipiv = [0i32; 3];
        let mut jpiv = [0i32; 3];

        unsafe {
            let ret = oblas_dgetc2(
                OblasLayout::ColMajor,
                3,
                a.as_mut_ptr(),
                3,
                ipiv.as_mut_ptr(),
                jpiv.as_mut_ptr(),
            );
            assert_eq!(ret, 0);

            let mut b = [4.0f64, 10.0, 24.0];
            let ret2 = oblas_dgesc2(
                OblasLayout::ColMajor,
                3,
                1,
                a.as_ptr(),
                3,
                ipiv.as_ptr(),
                jpiv.as_ptr(),
                b.as_mut_ptr(),
                3,
            );
            assert_eq!(ret2, 0);

            // Solution should be [1, 1, 1]
            for (i, &val) in b.iter().enumerate() {
                assert!((val - 1.0).abs() < 1e-8, "x[{}] = {}, expected 1.0", i, val);
            }
        }
    }

    #[test]
    fn test_dgetc2_identity() {
        // Identity matrix
        let mut a = [1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; // column-major
        let mut ipiv = [0i32; 3];
        let mut jpiv = [0i32; 3];

        unsafe {
            let ret = oblas_dgetc2(
                OblasLayout::ColMajor,
                3,
                a.as_mut_ptr(),
                3,
                ipiv.as_mut_ptr(),
                jpiv.as_mut_ptr(),
            );
            assert_eq!(ret, 0);

            // LU of identity is identity
            // Diagonal should be 1
            assert!((a[0] - 1.0).abs() < 1e-10);
            assert!((a[4] - 1.0).abs() < 1e-10);
            assert!((a[8] - 1.0).abs() < 1e-10);
        }
    }
}
