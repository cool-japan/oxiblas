//! LAPACK FFI - Subspace computation routines.
//!
//! This module provides C-compatible FFI for subspace computations:
//!
//! - `oblas_snull`, `oblas_dnull` - Null space basis
//! - `oblas_scolspace`, `oblas_dcolspace` - Column space basis
//! - `oblas_srowspace`, `oblas_drowspace` - Row space basis
//! - `oblas_slnull`, `oblas_dlnull` - Left null space basis

use crate::types::*;
use oxiblas_lapack::utils::{col_space, left_null_space, null_space, row_space};
use oxiblas_matrix::Mat;
use std::ffi::c_int;

// =============================================================================
// SNULL - Null space (single precision)
// =============================================================================

/// Computes an orthonormal basis for the null space of a matrix.
///
/// The null space N(A) consists of all vectors x such that Ax = 0.
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `ns` must point to a valid n x null_dim matrix (output)
/// - `null_dim` must point to a valid integer (output: dimension of null space)
/// - `tol` is optional tolerance; use 0 for automatic
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_snull(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *const f32,
    lda: c_int,
    ns: *mut f32,
    ldns: c_int,
    null_dim: *mut c_int,
    tol: f32,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || ns.is_null() || null_dim.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldns = ldns as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat
    let mut mat_a = Mat::<f32>::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    let tolerance = if tol > 0.0 { Some(tol) } else { None };

    match null_space(mat_a.as_ref(), tolerance) {
        Ok(result) => {
            let dim = result.ncols();
            *null_dim = dim as c_int;

            // Copy null space basis to output
            for i in 0..n {
                for j in 0..dim {
                    let idx = if row_major {
                        i * ldns + j
                    } else {
                        j * ldns + i
                    };
                    *ns.add(idx) = result[(i, j)];
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::InvalidArg as c_int,
    }
}

// =============================================================================
// DNULL - Null space (double precision)
// =============================================================================

/// Computes an orthonormal basis for the null space of a matrix.
///
/// The null space N(A) consists of all vectors x such that Ax = 0.
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `ns` must point to a valid n x null_dim matrix (output)
/// - `null_dim` must point to a valid integer (output: dimension of null space)
/// - `tol` is optional tolerance; use 0 for automatic
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dnull(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *const f64,
    lda: c_int,
    ns: *mut f64,
    ldns: c_int,
    null_dim: *mut c_int,
    tol: f64,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || ns.is_null() || null_dim.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldns = ldns as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat
    let mut mat_a = Mat::<f64>::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    let tolerance = if tol > 0.0 { Some(tol) } else { None };

    match null_space(mat_a.as_ref(), tolerance) {
        Ok(result) => {
            let dim = result.ncols();
            *null_dim = dim as c_int;

            // Copy null space basis to output
            for i in 0..n {
                for j in 0..dim {
                    let idx = if row_major {
                        i * ldns + j
                    } else {
                        j * ldns + i
                    };
                    *ns.add(idx) = result[(i, j)];
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::InvalidArg as c_int,
    }
}

// =============================================================================
// SCOLSPACE - Column space (single precision)
// =============================================================================

/// Computes an orthonormal basis for the column space (range) of a matrix.
///
/// The column space R(A) is the span of A's columns.
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `cs` must point to a valid m x rank matrix (output)
/// - `rank` must point to a valid integer (output: dimension of column space)
/// - `tol` is optional tolerance; use 0 for automatic
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_scolspace(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *const f32,
    lda: c_int,
    cs: *mut f32,
    ldcs: c_int,
    rank: *mut c_int,
    tol: f32,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || cs.is_null() || rank.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldcs = ldcs as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat
    let mut mat_a = Mat::<f32>::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    let tolerance = if tol > 0.0 { Some(tol) } else { None };

    match col_space(mat_a.as_ref(), tolerance) {
        Ok(result) => {
            let r = result.ncols();
            *rank = r as c_int;

            // Copy column space basis to output
            for i in 0..m {
                for j in 0..r {
                    let idx = if row_major {
                        i * ldcs + j
                    } else {
                        j * ldcs + i
                    };
                    *cs.add(idx) = result[(i, j)];
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::InvalidArg as c_int,
    }
}

// =============================================================================
// DCOLSPACE - Column space (double precision)
// =============================================================================

/// Computes an orthonormal basis for the column space (range) of a matrix.
///
/// The column space R(A) is the span of A's columns.
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `cs` must point to a valid m x rank matrix (output)
/// - `rank` must point to a valid integer (output: dimension of column space)
/// - `tol` is optional tolerance; use 0 for automatic
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dcolspace(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *const f64,
    lda: c_int,
    cs: *mut f64,
    ldcs: c_int,
    rank: *mut c_int,
    tol: f64,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || cs.is_null() || rank.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldcs = ldcs as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat
    let mut mat_a = Mat::<f64>::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    let tolerance = if tol > 0.0 { Some(tol) } else { None };

    match col_space(mat_a.as_ref(), tolerance) {
        Ok(result) => {
            let r = result.ncols();
            *rank = r as c_int;

            // Copy column space basis to output
            for i in 0..m {
                for j in 0..r {
                    let idx = if row_major {
                        i * ldcs + j
                    } else {
                        j * ldcs + i
                    };
                    *cs.add(idx) = result[(i, j)];
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::InvalidArg as c_int,
    }
}

// =============================================================================
// SROWSPACE - Row space (single precision)
// =============================================================================

/// Computes an orthonormal basis for the row space of a matrix.
///
/// The row space is the column space of A^T.
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `rs` must point to a valid n x rank matrix (output)
/// - `rank` must point to a valid integer (output: dimension of row space)
/// - `tol` is optional tolerance; use 0 for automatic
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_srowspace(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *const f32,
    lda: c_int,
    rs: *mut f32,
    ldrs: c_int,
    rank: *mut c_int,
    tol: f32,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || rs.is_null() || rank.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldrs = ldrs as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat
    let mut mat_a = Mat::<f32>::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    let tolerance = if tol > 0.0 { Some(tol) } else { None };

    match row_space(mat_a.as_ref(), tolerance) {
        Ok(result) => {
            let r = result.ncols();
            *rank = r as c_int;

            // Copy row space basis to output
            for i in 0..n {
                for j in 0..r {
                    let idx = if row_major {
                        i * ldrs + j
                    } else {
                        j * ldrs + i
                    };
                    *rs.add(idx) = result[(i, j)];
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::InvalidArg as c_int,
    }
}

// =============================================================================
// DROWSPACE - Row space (double precision)
// =============================================================================

/// Computes an orthonormal basis for the row space of a matrix.
///
/// The row space is the column space of A^T.
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `rs` must point to a valid n x rank matrix (output)
/// - `rank` must point to a valid integer (output: dimension of row space)
/// - `tol` is optional tolerance; use 0 for automatic
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_drowspace(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *const f64,
    lda: c_int,
    rs: *mut f64,
    ldrs: c_int,
    rank: *mut c_int,
    tol: f64,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || rs.is_null() || rank.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldrs = ldrs as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat
    let mut mat_a = Mat::<f64>::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    let tolerance = if tol > 0.0 { Some(tol) } else { None };

    match row_space(mat_a.as_ref(), tolerance) {
        Ok(result) => {
            let r = result.ncols();
            *rank = r as c_int;

            // Copy row space basis to output
            for i in 0..n {
                for j in 0..r {
                    let idx = if row_major {
                        i * ldrs + j
                    } else {
                        j * ldrs + i
                    };
                    *rs.add(idx) = result[(i, j)];
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::InvalidArg as c_int,
    }
}

// =============================================================================
// SLNULL - Left null space (single precision)
// =============================================================================

/// Computes an orthonormal basis for the left null space of a matrix.
///
/// The left null space N(A^T) consists of all vectors y such that y^T A = 0.
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `lns` must point to a valid m x left_null_dim matrix (output)
/// - `left_null_dim` must point to a valid integer (output)
/// - `tol` is optional tolerance; use 0 for automatic
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_slnull(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *const f32,
    lda: c_int,
    lns: *mut f32,
    ldlns: c_int,
    left_null_dim: *mut c_int,
    tol: f32,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || lns.is_null() || left_null_dim.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldlns = ldlns as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat
    let mut mat_a = Mat::<f32>::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    let tolerance = if tol > 0.0 { Some(tol) } else { None };

    match left_null_space(mat_a.as_ref(), tolerance) {
        Ok(result) => {
            let dim = result.ncols();
            *left_null_dim = dim as c_int;

            // Copy left null space basis to output
            for i in 0..m {
                for j in 0..dim {
                    let idx = if row_major {
                        i * ldlns + j
                    } else {
                        j * ldlns + i
                    };
                    *lns.add(idx) = result[(i, j)];
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::InvalidArg as c_int,
    }
}

// =============================================================================
// DLNULL - Left null space (double precision)
// =============================================================================

/// Computes an orthonormal basis for the left null space of a matrix.
///
/// The left null space N(A^T) consists of all vectors y such that y^T A = 0.
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `lns` must point to a valid m x left_null_dim matrix (output)
/// - `left_null_dim` must point to a valid integer (output)
/// - `tol` is optional tolerance; use 0 for automatic
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dlnull(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *const f64,
    lda: c_int,
    lns: *mut f64,
    ldlns: c_int,
    left_null_dim: *mut c_int,
    tol: f64,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || lns.is_null() || left_null_dim.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let lda = lda as usize;
    let ldlns = ldlns as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert to Mat
    let mut mat_a = Mat::<f64>::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    let tolerance = if tol > 0.0 { Some(tol) } else { None };

    match left_null_space(mat_a.as_ref(), tolerance) {
        Ok(result) => {
            let dim = result.ncols();
            *left_null_dim = dim as c_int;

            // Copy left null space basis to output
            for i in 0..m {
                for j in 0..dim {
                    let idx = if row_major {
                        i * ldlns + j
                    } else {
                        j * ldlns + i
                    };
                    *lns.add(idx) = result[(i, j)];
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::InvalidArg as c_int,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_dnull_rank_deficient() {
        // Rank 1 matrix (rank deficient)
        // [[1, 2], [2, 4]] has null space dimension 1
        let a = [1.0f64, 2.0, 2.0, 4.0]; // Row major
        let mut ns = [0.0f64; 4]; // Max size n x n
        let mut null_dim: c_int = 0;

        unsafe {
            let result = oblas_dnull(
                OblasLayout::RowMajor,
                2,
                2,
                a.as_ptr(),
                2,
                ns.as_mut_ptr(),
                2,
                &mut null_dim,
                0.0,
            );

            assert_eq!(result, 0);
            assert_eq!(null_dim, 1);

            // Verify A * ns = 0
            // ns is n x null_dim = 2 x 1 in row-major with ldns=2
            // ns[(0, 0)] = ns[0 * 2 + 0] = ns[0]
            // ns[(1, 0)] = ns[1 * 2 + 0] = ns[2]
            let v0 = ns[0]; // ns[(0, 0)]
            let v1 = ns[2]; // ns[(1, 0)] in row-major with ldns=2

            // A * v should be zero
            let prod0 = 1.0 * v0 + 2.0 * v1;
            let prod1 = 2.0 * v0 + 4.0 * v1;
            assert!(approx_eq(prod0, 0.0, 1e-10));
            assert!(approx_eq(prod1, 0.0, 1e-10));
        }
    }

    #[test]
    fn test_dnull_full_rank() {
        // Full rank matrix has trivial null space
        let a = [1.0f64, 2.0, 3.0, 4.0]; // Row major
        let mut ns = [0.0f64; 4];
        let mut null_dim: c_int = 0;

        unsafe {
            let result = oblas_dnull(
                OblasLayout::RowMajor,
                2,
                2,
                a.as_ptr(),
                2,
                ns.as_mut_ptr(),
                2,
                &mut null_dim,
                0.0,
            );

            assert_eq!(result, 0);
            assert_eq!(null_dim, 0);
        }
    }

    #[test]
    fn test_dcolspace_full_rank() {
        let a = [1.0f64, 2.0, 3.0, 4.0]; // Row major
        let mut cs = [0.0f64; 4];
        let mut rank: c_int = 0;

        unsafe {
            let result = oblas_dcolspace(
                OblasLayout::RowMajor,
                2,
                2,
                a.as_ptr(),
                2,
                cs.as_mut_ptr(),
                2,
                &mut rank,
                0.0,
            );

            assert_eq!(result, 0);
            assert_eq!(rank, 2);

            // Verify orthonormality of column space basis
            // Column 0: cs[0], cs[2]
            // Column 1: cs[1], cs[3]
            let col0_norm = cs[0] * cs[0] + cs[2] * cs[2];
            let col1_norm = cs[1] * cs[1] + cs[3] * cs[3];
            let dot = cs[0] * cs[1] + cs[2] * cs[3];

            assert!(approx_eq(col0_norm, 1.0, 1e-10));
            assert!(approx_eq(col1_norm, 1.0, 1e-10));
            assert!(approx_eq(dot, 0.0, 1e-10));
        }
    }

    #[test]
    fn test_dcolspace_rank_deficient() {
        // [[1, 2], [2, 4]] has rank 1
        let a = [1.0f64, 2.0, 2.0, 4.0]; // Row major
        let mut cs = [0.0f64; 4];
        let mut rank: c_int = 0;

        unsafe {
            let result = oblas_dcolspace(
                OblasLayout::RowMajor,
                2,
                2,
                a.as_ptr(),
                2,
                cs.as_mut_ptr(),
                2,
                &mut rank,
                0.0,
            );

            assert_eq!(result, 0);
            assert_eq!(rank, 1);
        }
    }

    #[test]
    fn test_drowspace() {
        let a = [1.0f64, 2.0, 3.0, 4.0]; // Row major
        let mut rs = [0.0f64; 4];
        let mut rank: c_int = 0;

        unsafe {
            let result = oblas_drowspace(
                OblasLayout::RowMajor,
                2,
                2,
                a.as_ptr(),
                2,
                rs.as_mut_ptr(),
                2,
                &mut rank,
                0.0,
            );

            assert_eq!(result, 0);
            assert_eq!(rank, 2);
        }
    }

    #[test]
    fn test_dlnull() {
        // [[1, 2], [2, 4]] has left null space dimension 1
        let a = [1.0f64, 2.0, 2.0, 4.0]; // Row major
        let mut lns = [0.0f64; 4];
        let mut left_null_dim: c_int = 0;

        unsafe {
            let result = oblas_dlnull(
                OblasLayout::RowMajor,
                2,
                2,
                a.as_ptr(),
                2,
                lns.as_mut_ptr(),
                2,
                &mut left_null_dim,
                0.0,
            );

            assert_eq!(result, 0);
            assert_eq!(left_null_dim, 1);

            // Verify y^T * A = 0
            let y0 = lns[0];
            let y1 = lns[2]; // Row major: second row

            // y^T * A should be zero (2 elements)
            let prod0 = y0 * 1.0 + y1 * 2.0;
            let prod1 = y0 * 2.0 + y1 * 4.0;
            assert!(approx_eq(prod0, 0.0, 1e-10));
            assert!(approx_eq(prod1, 0.0, 1e-10));
        }
    }

    #[test]
    fn test_snull_basic() {
        let a = [1.0f32, 2.0, 2.0, 4.0]; // Row major, rank 1
        let mut ns = [0.0f32; 4];
        let mut null_dim: c_int = 0;

        unsafe {
            let result = oblas_snull(
                OblasLayout::RowMajor,
                2,
                2,
                a.as_ptr(),
                2,
                ns.as_mut_ptr(),
                2,
                &mut null_dim,
                0.0,
            );

            assert_eq!(result, 0);
            assert_eq!(null_dim, 1);
        }
    }

    #[test]
    fn test_four_fundamental_subspaces() {
        // Test the Fundamental Theorem of Linear Algebra
        // For 3x3 matrix with rank 2:
        // - dim(col space) = 2
        // - dim(null space) = 1
        // - dim(row space) = 2
        // - dim(left null space) = 1

        // This matrix has rank 2:
        // [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        let a = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]; // Row major

        let mut cs = [0.0f64; 9];
        let mut ns = [0.0f64; 9];
        let mut rs = [0.0f64; 9];
        let mut lns = [0.0f64; 9];

        let mut col_rank: c_int = 0;
        let mut null_dim: c_int = 0;
        let mut row_rank: c_int = 0;
        let mut left_null_dim: c_int = 0;

        unsafe {
            oblas_dcolspace(
                OblasLayout::RowMajor,
                3,
                3,
                a.as_ptr(),
                3,
                cs.as_mut_ptr(),
                3,
                &mut col_rank,
                0.0,
            );

            oblas_dnull(
                OblasLayout::RowMajor,
                3,
                3,
                a.as_ptr(),
                3,
                ns.as_mut_ptr(),
                3,
                &mut null_dim,
                0.0,
            );

            oblas_drowspace(
                OblasLayout::RowMajor,
                3,
                3,
                a.as_ptr(),
                3,
                rs.as_mut_ptr(),
                3,
                &mut row_rank,
                0.0,
            );

            oblas_dlnull(
                OblasLayout::RowMajor,
                3,
                3,
                a.as_ptr(),
                3,
                lns.as_mut_ptr(),
                3,
                &mut left_null_dim,
                0.0,
            );

            // Verify fundamental theorem
            assert_eq!(col_rank, 2);
            assert_eq!(null_dim, 1); // n - rank = 3 - 2 = 1
            assert_eq!(row_rank, 2);
            assert_eq!(left_null_dim, 1); // m - rank = 3 - 2 = 1
        }
    }
}
