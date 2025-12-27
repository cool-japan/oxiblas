//! LAPACK FFI - Tridiagonal system solvers.
//!
//! Provides C-compatible FFI for tridiagonal matrix operations:
//!
//! - `oblas_sgtsv`, `oblas_dgtsv` - General tridiagonal solve
//! - `oblas_sptsv`, `oblas_dptsv` - SPD tridiagonal solve
//! - `oblas_sgttrf`, `oblas_dgttrf` - Tridiagonal factorization
//! - `oblas_sgttrs`, `oblas_dgttrs` - Solve from tridiagonal factors
//! - `oblas_spttrf`, `oblas_dpttrf` - SPD tridiagonal factorization
//! - `oblas_spttrs`, `oblas_dpttrs` - Solve from SPD tridiagonal factors

use crate::types::*;
use oxiblas_lapack::solve::{
    tridiag_factor, tridiag_factor_spd, tridiag_solve, tridiag_solve_factored,
    tridiag_solve_factored_spd, tridiag_solve_spd,
};
use std::ffi::c_int;

// =============================================================================
// SGTSV - General tridiagonal solve (single precision)
// =============================================================================

/// Solves a tridiagonal system Ax = b.
///
/// The matrix A is n-by-n tridiagonal with:
/// - `dl`: sub-diagonal (length n-1)
/// - `d`: main diagonal (length n)
/// - `du`: super-diagonal (length n-1)
///
/// Uses the Thomas algorithm (O(n) complexity).
///
/// # Arguments
/// * `n` - Order of the matrix A
/// * `nrhs` - Number of right-hand sides (columns of B)
/// * `dl` - Sub-diagonal elements (length n-1)
/// * `d` - Main diagonal elements (length n)
/// * `du` - Super-diagonal elements (length n-1)
/// * `b` - On input: right-hand side matrix B (n × nrhs). On output: solution X
/// * `ldb` - Leading dimension of B
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * positive value k if matrix is singular at index k-1
///
/// # Safety
/// - `dl` must point to an array of length n-1
/// - `d` must point to an array of length n
/// - `du` must point to an array of length n-1
/// - `b` must point to an n × nrhs matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sgtsv(
    n: c_int,
    nrhs: c_int,
    dl: *const f32,
    d: *const f32,
    du: *const f32,
    b: *mut f32,
    ldb: c_int,
) -> c_int {
    if n <= 0 || nrhs <= 0 || dl.is_null() || d.is_null() || du.is_null() || b.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let nrhs_val = nrhs as usize;
    let ldb_val = ldb as usize;

    // Read arrays
    let dl_slice = std::slice::from_raw_parts(dl, n_val.saturating_sub(1));
    let d_slice = std::slice::from_raw_parts(d, n_val);
    let du_slice = std::slice::from_raw_parts(du, n_val.saturating_sub(1));

    // Solve each column
    for j in 0..nrhs_val {
        // Extract column j of B
        let b_col: Vec<f32> = (0..n_val).map(|i| *b.add(j * ldb_val + i)).collect();

        match tridiag_solve(dl_slice, d_slice, du_slice, &b_col) {
            Ok(x) => {
                // Copy solution back
                for i in 0..n_val {
                    *b.add(j * ldb_val + i) = x[i];
                }
            }
            Err(oxiblas_lapack::solve::TridiagError::Singular { index }) => {
                return (index + 1) as c_int;
            }
            Err(_) => return OblasReturn::InvalidArg as c_int,
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DGTSV - General tridiagonal solve (double precision)
// =============================================================================

/// Solves a tridiagonal system Ax = b (double precision).
///
/// # Safety
/// - `dl` must point to an array of length n-1
/// - `d` must point to an array of length n
/// - `du` must point to an array of length n-1
/// - `b` must point to an n × nrhs matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dgtsv(
    n: c_int,
    nrhs: c_int,
    dl: *const f64,
    d: *const f64,
    du: *const f64,
    b: *mut f64,
    ldb: c_int,
) -> c_int {
    if n <= 0 || nrhs <= 0 || dl.is_null() || d.is_null() || du.is_null() || b.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let nrhs_val = nrhs as usize;
    let ldb_val = ldb as usize;

    // Read arrays
    let dl_slice = std::slice::from_raw_parts(dl, n_val.saturating_sub(1));
    let d_slice = std::slice::from_raw_parts(d, n_val);
    let du_slice = std::slice::from_raw_parts(du, n_val.saturating_sub(1));

    // Solve each column
    for j in 0..nrhs_val {
        // Extract column j of B
        let b_col: Vec<f64> = (0..n_val).map(|i| *b.add(j * ldb_val + i)).collect();

        match tridiag_solve(dl_slice, d_slice, du_slice, &b_col) {
            Ok(x) => {
                // Copy solution back
                for i in 0..n_val {
                    *b.add(j * ldb_val + i) = x[i];
                }
            }
            Err(oxiblas_lapack::solve::TridiagError::Singular { index }) => {
                return (index + 1) as c_int;
            }
            Err(_) => return OblasReturn::InvalidArg as c_int,
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// SPTSV - SPD tridiagonal solve (single precision)
// =============================================================================

/// Solves a symmetric positive definite tridiagonal system Ax = b.
///
/// The matrix A is n-by-n SPD tridiagonal with:
/// - `d`: main diagonal (length n, must be positive)
/// - `e`: off-diagonal (length n-1)
///
/// Uses LDL^T factorization for numerical stability.
///
/// # Arguments
/// * `n` - Order of the matrix A
/// * `nrhs` - Number of right-hand sides
/// * `d` - Main diagonal (length n)
/// * `e` - Off-diagonal (length n-1)
/// * `b` - On input: RHS. On output: solution
/// * `ldb` - Leading dimension of B
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * positive value k if not positive definite at index k-1
///
/// # Safety
/// - `d` must point to an array of length n
/// - `e` must point to an array of length n-1
/// - `b` must point to an n × nrhs matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sptsv(
    n: c_int,
    nrhs: c_int,
    d: *const f32,
    e: *const f32,
    b: *mut f32,
    ldb: c_int,
) -> c_int {
    if n <= 0 || nrhs <= 0 || d.is_null() || e.is_null() || b.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let nrhs_val = nrhs as usize;
    let ldb_val = ldb as usize;

    let d_slice = std::slice::from_raw_parts(d, n_val);
    let e_slice = std::slice::from_raw_parts(e, n_val.saturating_sub(1));

    for j in 0..nrhs_val {
        let b_col: Vec<f32> = (0..n_val).map(|i| *b.add(j * ldb_val + i)).collect();

        match tridiag_solve_spd(d_slice, e_slice, &b_col) {
            Ok(x) => {
                for i in 0..n_val {
                    *b.add(j * ldb_val + i) = x[i];
                }
            }
            Err(oxiblas_lapack::solve::TridiagError::Singular { index }) => {
                return (index + 1) as c_int;
            }
            Err(_) => return OblasReturn::InvalidArg as c_int,
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DPTSV - SPD tridiagonal solve (double precision)
// =============================================================================

/// Solves a symmetric positive definite tridiagonal system (double precision).
///
/// # Safety
/// - `d` must point to an array of length n
/// - `e` must point to an array of length n-1
/// - `b` must point to an n × nrhs matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dptsv(
    n: c_int,
    nrhs: c_int,
    d: *const f64,
    e: *const f64,
    b: *mut f64,
    ldb: c_int,
) -> c_int {
    if n <= 0 || nrhs <= 0 || d.is_null() || e.is_null() || b.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let nrhs_val = nrhs as usize;
    let ldb_val = ldb as usize;

    let d_slice = std::slice::from_raw_parts(d, n_val);
    let e_slice = std::slice::from_raw_parts(e, n_val.saturating_sub(1));

    for j in 0..nrhs_val {
        let b_col: Vec<f64> = (0..n_val).map(|i| *b.add(j * ldb_val + i)).collect();

        match tridiag_solve_spd(d_slice, e_slice, &b_col) {
            Ok(x) => {
                for i in 0..n_val {
                    *b.add(j * ldb_val + i) = x[i];
                }
            }
            Err(oxiblas_lapack::solve::TridiagError::Singular { index }) => {
                return (index + 1) as c_int;
            }
            Err(_) => return OblasReturn::InvalidArg as c_int,
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// SGTTRF - Tridiagonal factorization (single precision)
// =============================================================================

/// Computes the LU factorization of a tridiagonal matrix.
///
/// A = LU where L is unit lower bidiagonal and U is upper bidiagonal.
///
/// # Arguments
/// * `n` - Order of the matrix A
/// * `dl` - On input: sub-diagonal. On output: multipliers for L
/// * `d` - On input: main diagonal. On output: diagonal of U
/// * `du` - Super-diagonal (unchanged)
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * positive value k if singular at index k-1
///
/// # Safety
/// - `dl` must point to an array of length n-1
/// - `d` must point to an array of length n
/// - `du` must point to an array of length n-1
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sgttrf(
    n: c_int,
    dl: *mut f32,
    d: *mut f32,
    du: *const f32,
) -> c_int {
    if n <= 0 || dl.is_null() || d.is_null() || du.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;

    let dl_slice = std::slice::from_raw_parts(dl, n_val.saturating_sub(1));
    let d_slice = std::slice::from_raw_parts(d, n_val);
    let du_slice = std::slice::from_raw_parts(du, n_val.saturating_sub(1));

    match tridiag_factor(dl_slice, d_slice, du_slice) {
        Ok(factors) => {
            // Copy modified values back
            for i in 0..(n_val - 1) {
                *dl.add(i) = factors.dl_modified[i];
            }
            for i in 0..n_val {
                *d.add(i) = factors.d_modified[i];
            }
            OblasReturn::Success as c_int
        }
        Err(oxiblas_lapack::solve::TridiagError::Singular { index }) => (index + 1) as c_int,
        Err(_) => OblasReturn::InvalidArg as c_int,
    }
}

// =============================================================================
// DGTTRF - Tridiagonal factorization (double precision)
// =============================================================================

/// Computes the LU factorization of a tridiagonal matrix (double precision).
///
/// # Safety
/// - `dl` must point to an array of length n-1
/// - `d` must point to an array of length n
/// - `du` must point to an array of length n-1
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dgttrf(
    n: c_int,
    dl: *mut f64,
    d: *mut f64,
    du: *const f64,
) -> c_int {
    if n <= 0 || dl.is_null() || d.is_null() || du.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;

    let dl_slice = std::slice::from_raw_parts(dl, n_val.saturating_sub(1));
    let d_slice = std::slice::from_raw_parts(d, n_val);
    let du_slice = std::slice::from_raw_parts(du, n_val.saturating_sub(1));

    match tridiag_factor(dl_slice, d_slice, du_slice) {
        Ok(factors) => {
            // Copy modified values back
            for i in 0..(n_val - 1) {
                *dl.add(i) = factors.dl_modified[i];
            }
            for i in 0..n_val {
                *d.add(i) = factors.d_modified[i];
            }
            OblasReturn::Success as c_int
        }
        Err(oxiblas_lapack::solve::TridiagError::Singular { index }) => (index + 1) as c_int,
        Err(_) => OblasReturn::InvalidArg as c_int,
    }
}

// =============================================================================
// SGTTRS - Solve from tridiagonal factors (single precision)
// =============================================================================

/// Solves Ax = b using precomputed LU factors from sgttrf.
///
/// # Arguments
/// * `n` - Order of the matrix A
/// * `nrhs` - Number of right-hand sides
/// * `dl` - Multipliers from factorization (length n-1)
/// * `d` - Diagonal of U from factorization (length n)
/// * `du` - Super-diagonal (length n-1)
/// * `b` - On input: RHS. On output: solution
/// * `ldb` - Leading dimension of B
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
///
/// # Safety
/// - Arrays must be properly sized
/// - Factors must be from a successful sgttrf call
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sgttrs(
    n: c_int,
    nrhs: c_int,
    dl: *const f32,
    d: *const f32,
    du: *const f32,
    b: *mut f32,
    ldb: c_int,
) -> c_int {
    if n <= 0 || nrhs <= 0 || dl.is_null() || d.is_null() || du.is_null() || b.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let nrhs_val = nrhs as usize;
    let ldb_val = ldb as usize;

    // Reconstruct factors
    let dl_slice = std::slice::from_raw_parts(dl, n_val.saturating_sub(1));
    let d_slice = std::slice::from_raw_parts(d, n_val);
    let du_slice = std::slice::from_raw_parts(du, n_val.saturating_sub(1));

    let factors = oxiblas_lapack::solve::TridiagFactors {
        dl_modified: dl_slice.to_vec(),
        d_modified: d_slice.to_vec(),
        du: du_slice.to_vec(),
        n: n_val,
    };

    for j in 0..nrhs_val {
        let b_col: Vec<f32> = (0..n_val).map(|i| *b.add(j * ldb_val + i)).collect();

        match tridiag_solve_factored(&factors, &b_col) {
            Ok(x) => {
                for i in 0..n_val {
                    *b.add(j * ldb_val + i) = x[i];
                }
            }
            Err(_) => return OblasReturn::InvalidArg as c_int,
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DGTTRS - Solve from tridiagonal factors (double precision)
// =============================================================================

/// Solves Ax = b using precomputed LU factors (double precision).
///
/// # Safety
/// - Arrays must be properly sized
/// - Factors must be from a successful dgttrf call
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dgttrs(
    n: c_int,
    nrhs: c_int,
    dl: *const f64,
    d: *const f64,
    du: *const f64,
    b: *mut f64,
    ldb: c_int,
) -> c_int {
    if n <= 0 || nrhs <= 0 || dl.is_null() || d.is_null() || du.is_null() || b.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let nrhs_val = nrhs as usize;
    let ldb_val = ldb as usize;

    // Reconstruct factors
    let dl_slice = std::slice::from_raw_parts(dl, n_val.saturating_sub(1));
    let d_slice = std::slice::from_raw_parts(d, n_val);
    let du_slice = std::slice::from_raw_parts(du, n_val.saturating_sub(1));

    let factors = oxiblas_lapack::solve::TridiagFactors {
        dl_modified: dl_slice.to_vec(),
        d_modified: d_slice.to_vec(),
        du: du_slice.to_vec(),
        n: n_val,
    };

    for j in 0..nrhs_val {
        let b_col: Vec<f64> = (0..n_val).map(|i| *b.add(j * ldb_val + i)).collect();

        match tridiag_solve_factored(&factors, &b_col) {
            Ok(x) => {
                for i in 0..n_val {
                    *b.add(j * ldb_val + i) = x[i];
                }
            }
            Err(_) => return OblasReturn::InvalidArg as c_int,
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// SPTTRF - SPD tridiagonal factorization (single precision)
// =============================================================================

/// Computes the LDL^T factorization of a symmetric positive definite
/// tridiagonal matrix.
///
/// The matrix A is n-by-n SPD tridiagonal with:
/// - `d`: main diagonal (length n, must be positive)
/// - `e`: off-diagonal (length n-1)
///
/// On successful completion:
/// - `d` contains the diagonal of D
/// - `e` contains the subdiagonal of L
///
/// # Arguments
/// * `n` - Order of the matrix A
/// * `d` - On input: main diagonal. On output: diagonal of D
/// * `e` - On input: off-diagonal. On output: subdiagonal of L
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * positive value k if not positive definite at index k-1
///
/// # Safety
/// - `d` must point to an array of length n
/// - `e` must point to an array of length n-1
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_spttrf(n: c_int, d: *mut f32, e: *mut f32) -> c_int {
    if n <= 0 || d.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;

    let d_slice = std::slice::from_raw_parts(d, n_val);
    let e_slice = if n_val > 1 {
        std::slice::from_raw_parts(e, n_val.saturating_sub(1))
    } else {
        &[]
    };

    match tridiag_factor_spd(d_slice, e_slice) {
        Ok(factors) => {
            // Copy factored values back
            for i in 0..n_val {
                *d.add(i) = factors.d_factor[i];
            }
            for i in 0..(n_val.saturating_sub(1)) {
                *e.add(i) = factors.l_factor[i];
            }
            OblasReturn::Success as c_int
        }
        Err(oxiblas_lapack::solve::TridiagError::Singular { index }) => (index + 1) as c_int,
        Err(_) => OblasReturn::InvalidArg as c_int,
    }
}

// =============================================================================
// DPTTRF - SPD tridiagonal factorization (double precision)
// =============================================================================

/// Computes the LDL^T factorization of a symmetric positive definite
/// tridiagonal matrix (double precision).
///
/// # Safety
/// - `d` must point to an array of length n
/// - `e` must point to an array of length n-1
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dpttrf(n: c_int, d: *mut f64, e: *mut f64) -> c_int {
    if n <= 0 || d.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;

    let d_slice = std::slice::from_raw_parts(d, n_val);
    let e_slice = if n_val > 1 {
        std::slice::from_raw_parts(e, n_val.saturating_sub(1))
    } else {
        &[]
    };

    match tridiag_factor_spd(d_slice, e_slice) {
        Ok(factors) => {
            // Copy factored values back
            for i in 0..n_val {
                *d.add(i) = factors.d_factor[i];
            }
            for i in 0..(n_val.saturating_sub(1)) {
                *e.add(i) = factors.l_factor[i];
            }
            OblasReturn::Success as c_int
        }
        Err(oxiblas_lapack::solve::TridiagError::Singular { index }) => (index + 1) as c_int,
        Err(_) => OblasReturn::InvalidArg as c_int,
    }
}

// =============================================================================
// SPTTRS - Solve from SPD tridiagonal factors (single precision)
// =============================================================================

/// Solves Ax = b using precomputed LDL^T factors from spttrf.
///
/// # Arguments
/// * `n` - Order of the matrix A
/// * `nrhs` - Number of right-hand sides
/// * `d` - Diagonal of D from factorization (length n)
/// * `e` - Subdiagonal of L from factorization (length n-1)
/// * `b` - On input: RHS. On output: solution
/// * `ldb` - Leading dimension of B
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
///
/// # Safety
/// - Arrays must be properly sized
/// - Factors must be from a successful spttrf call
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_spttrs(
    n: c_int,
    nrhs: c_int,
    d: *const f32,
    e: *const f32,
    b: *mut f32,
    ldb: c_int,
) -> c_int {
    if n <= 0 || nrhs <= 0 || d.is_null() || b.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let nrhs_val = nrhs as usize;
    let ldb_val = ldb as usize;

    // Reconstruct factors
    let d_slice = std::slice::from_raw_parts(d, n_val);
    let e_slice = if n_val > 1 {
        std::slice::from_raw_parts(e, n_val.saturating_sub(1))
    } else {
        &[]
    };

    let factors = oxiblas_lapack::solve::TridiagSPDFactors {
        d_factor: d_slice.to_vec(),
        l_factor: e_slice.to_vec(),
        n: n_val,
    };

    for j in 0..nrhs_val {
        let b_col: Vec<f32> = (0..n_val).map(|i| *b.add(j * ldb_val + i)).collect();

        match tridiag_solve_factored_spd(&factors, &b_col) {
            Ok(x) => {
                for i in 0..n_val {
                    *b.add(j * ldb_val + i) = x[i];
                }
            }
            Err(_) => return OblasReturn::InvalidArg as c_int,
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DPTTRS - Solve from SPD tridiagonal factors (double precision)
// =============================================================================

/// Solves Ax = b using precomputed LDL^T factors (double precision).
///
/// # Safety
/// - Arrays must be properly sized
/// - Factors must be from a successful dpttrf call
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dpttrs(
    n: c_int,
    nrhs: c_int,
    d: *const f64,
    e: *const f64,
    b: *mut f64,
    ldb: c_int,
) -> c_int {
    if n <= 0 || nrhs <= 0 || d.is_null() || b.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let nrhs_val = nrhs as usize;
    let ldb_val = ldb as usize;

    // Reconstruct factors
    let d_slice = std::slice::from_raw_parts(d, n_val);
    let e_slice = if n_val > 1 {
        std::slice::from_raw_parts(e, n_val.saturating_sub(1))
    } else {
        &[]
    };

    let factors = oxiblas_lapack::solve::TridiagSPDFactors {
        d_factor: d_slice.to_vec(),
        l_factor: e_slice.to_vec(),
        n: n_val,
    };

    for j in 0..nrhs_val {
        let b_col: Vec<f64> = (0..n_val).map(|i| *b.add(j * ldb_val + i)).collect();

        match tridiag_solve_factored_spd(&factors, &b_col) {
            Ok(x) => {
                for i in 0..n_val {
                    *b.add(j * ldb_val + i) = x[i];
                }
            }
            Err(_) => return OblasReturn::InvalidArg as c_int,
        }
    }

    OblasReturn::Success as c_int
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dgtsv_simple() {
        // Solve: [2 -1  0] [x0]   [1]
        //        [-1 2 -1] [x1] = [0]
        //        [0 -1  2] [x2]   [1]
        // Solution: x = [1, 1, 1]

        let dl = [-1.0f64, -1.0];
        let d = [2.0f64, 2.0, 2.0];
        let du = [-1.0f64, -1.0];
        let mut b = [1.0f64, 0.0, 1.0];

        unsafe {
            let ret = oblas_dgtsv(
                3,
                1,
                dl.as_ptr(),
                d.as_ptr(),
                du.as_ptr(),
                b.as_mut_ptr(),
                3,
            );
            assert_eq!(ret, 0);
            assert!((b[0] - 1.0).abs() < 1e-10);
            assert!((b[1] - 1.0).abs() < 1e-10);
            assert!((b[2] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sgtsv_simple() {
        let dl = [-1.0f32, -1.0];
        let d = [2.0f32, 2.0, 2.0];
        let du = [-1.0f32, -1.0];
        let mut b = [1.0f32, 0.0, 1.0];

        unsafe {
            let ret = oblas_sgtsv(
                3,
                1,
                dl.as_ptr(),
                d.as_ptr(),
                du.as_ptr(),
                b.as_mut_ptr(),
                3,
            );
            assert_eq!(ret, 0);
            assert!((b[0] - 1.0).abs() < 1e-5);
            assert!((b[1] - 1.0).abs() < 1e-5);
            assert!((b[2] - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dgtsv_multiple_rhs() {
        let dl = [-1.0f64, -1.0];
        let d = [2.0f64, 2.0, 2.0];
        let du = [-1.0f64, -1.0];
        // Two RHS columns (column-major)
        let mut b = [1.0f64, 0.0, 1.0, 2.0, 0.0, 2.0];

        unsafe {
            let ret = oblas_dgtsv(
                3,
                2,
                dl.as_ptr(),
                d.as_ptr(),
                du.as_ptr(),
                b.as_mut_ptr(),
                3,
            );
            assert_eq!(ret, 0);

            // First column: [1, 1, 1]
            assert!((b[0] - 1.0).abs() < 1e-10);
            assert!((b[1] - 1.0).abs() < 1e-10);
            assert!((b[2] - 1.0).abs() < 1e-10);

            // Second column: [2, 2, 2]
            assert!((b[3] - 2.0).abs() < 1e-10);
            assert!((b[4] - 2.0).abs() < 1e-10);
            assert!((b[5] - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dgtsv_2x2() {
        // [2 1] [x0]   [5]
        // [1 3] [x1] = [7]
        // Solution: x0 = 1.6, x1 = 1.8

        let dl = [1.0f64];
        let d = [2.0f64, 3.0];
        let du = [1.0f64];
        let mut b = [5.0f64, 7.0];

        unsafe {
            let ret = oblas_dgtsv(
                2,
                1,
                dl.as_ptr(),
                d.as_ptr(),
                du.as_ptr(),
                b.as_mut_ptr(),
                2,
            );
            assert_eq!(ret, 0);
            assert!((b[0] - 1.6).abs() < 1e-10);
            assert!((b[1] - 1.8).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dptsv_spd() {
        // SPD tridiagonal: [4 -1  0]
        //                  [-1 4 -1]
        //                  [0 -1  4]

        let d = [4.0f64, 4.0, 4.0];
        let e = [-1.0f64, -1.0];
        let b_orig = [3.0f64, 2.0, 3.0];
        let mut b = b_orig;

        unsafe {
            let ret = oblas_dptsv(3, 1, d.as_ptr(), e.as_ptr(), b.as_mut_ptr(), 3);
            assert_eq!(ret, 0);

            // Verify Ax = b_original where x = b (solution)
            // For positive definite tridiagonal, off-diagonals are stored as e
            // A[i,i+1] = A[i+1,i] = e[i] (symmetric)
            let ax0 = d[0] * b[0] + e[0] * b[1];
            let ax1 = e[0] * b[0] + d[1] * b[1] + e[1] * b[2];
            let ax2 = e[1] * b[1] + d[2] * b[2];

            assert!(
                (ax0 - b_orig[0]).abs() < 1e-10,
                "ax0={} != b0={}",
                ax0,
                b_orig[0]
            );
            assert!(
                (ax1 - b_orig[1]).abs() < 1e-10,
                "ax1={} != b1={}",
                ax1,
                b_orig[1]
            );
            assert!(
                (ax2 - b_orig[2]).abs() < 1e-10,
                "ax2={} != b2={}",
                ax2,
                b_orig[2]
            );
        }
    }

    #[test]
    fn test_sptsv_spd() {
        let d = [4.0f32, 4.0, 4.0];
        let e = [-1.0f32, -1.0];
        let b_orig = [3.0f32, 2.0, 3.0];
        let mut b = b_orig;

        unsafe {
            let ret = oblas_sptsv(3, 1, d.as_ptr(), e.as_ptr(), b.as_mut_ptr(), 3);
            assert_eq!(ret, 0);

            // Verify Ax = b_original where x = b (solution)
            let ax0 = d[0] * b[0] + e[0] * b[1];
            let ax1 = e[0] * b[0] + d[1] * b[1] + e[1] * b[2];
            let ax2 = e[1] * b[1] + d[2] * b[2];

            assert!(
                (ax0 - b_orig[0]).abs() < 1e-5,
                "ax0={} != b0={}",
                ax0,
                b_orig[0]
            );
            assert!(
                (ax1 - b_orig[1]).abs() < 1e-5,
                "ax1={} != b1={}",
                ax1,
                b_orig[1]
            );
            assert!(
                (ax2 - b_orig[2]).abs() < 1e-5,
                "ax2={} != b2={}",
                ax2,
                b_orig[2]
            );
        }
    }

    #[test]
    fn test_dgttrf_dgttrs() {
        // Factor then solve
        let mut dl = [-1.0f64, -1.0];
        let mut d = [2.0f64, 2.0, 2.0];
        let du = [-1.0f64, -1.0];

        unsafe {
            // Factor
            let ret = oblas_dgttrf(3, dl.as_mut_ptr(), d.as_mut_ptr(), du.as_ptr());
            assert_eq!(ret, 0);

            // Solve
            let mut b = [1.0f64, 0.0, 1.0];
            let ret = oblas_dgttrs(
                3,
                1,
                dl.as_ptr(),
                d.as_ptr(),
                du.as_ptr(),
                b.as_mut_ptr(),
                3,
            );
            assert_eq!(ret, 0);

            assert!((b[0] - 1.0).abs() < 1e-10);
            assert!((b[1] - 1.0).abs() < 1e-10);
            assert!((b[2] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sgttrf_sgttrs() {
        let mut dl = [-1.0f32, -1.0];
        let mut d = [2.0f32, 2.0, 2.0];
        let du = [-1.0f32, -1.0];

        unsafe {
            let ret = oblas_sgttrf(3, dl.as_mut_ptr(), d.as_mut_ptr(), du.as_ptr());
            assert_eq!(ret, 0);

            let mut b = [1.0f32, 0.0, 1.0];
            let ret = oblas_sgttrs(
                3,
                1,
                dl.as_ptr(),
                d.as_ptr(),
                du.as_ptr(),
                b.as_mut_ptr(),
                3,
            );
            assert_eq!(ret, 0);

            assert!((b[0] - 1.0).abs() < 1e-5);
            assert!((b[1] - 1.0).abs() < 1e-5);
            assert!((b[2] - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dgtsv_singular() {
        let dl = [1.0f64];
        let d = [0.0f64, 1.0]; // Zero pivot
        let du = [1.0f64];
        let mut b = [1.0f64, 1.0];

        unsafe {
            let ret = oblas_dgtsv(
                2,
                1,
                dl.as_ptr(),
                d.as_ptr(),
                du.as_ptr(),
                b.as_mut_ptr(),
                2,
            );
            assert_eq!(ret, 1); // Singular at index 0
        }
    }

    #[test]
    fn test_dgtsv_verify_solution() {
        // Random system, verify Ax = b
        let dl = [1.0f64, 2.0, 1.5];
        let d = [4.0f64, 5.0, 6.0, 7.0];
        let du = [1.0f64, 1.0, 2.0];
        let b_orig = [10.0f64, 20.0, 30.0, 40.0];
        let mut b = b_orig.clone();

        unsafe {
            let ret = oblas_dgtsv(
                4,
                1,
                dl.as_ptr(),
                d.as_ptr(),
                du.as_ptr(),
                b.as_mut_ptr(),
                4,
            );
            assert_eq!(ret, 0);

            // Verify Ax = b
            let ax0 = d[0] * b[0] + du[0] * b[1];
            let ax1 = dl[0] * b[0] + d[1] * b[1] + du[1] * b[2];
            let ax2 = dl[1] * b[1] + d[2] * b[2] + du[2] * b[3];
            let ax3 = dl[2] * b[2] + d[3] * b[3];

            assert!((ax0 - b_orig[0]).abs() < 1e-10);
            assert!((ax1 - b_orig[1]).abs() < 1e-10);
            assert!((ax2 - b_orig[2]).abs() < 1e-10);
            assert!((ax3 - b_orig[3]).abs() < 1e-10);
        }
    }

    // ===== PTTRF/PTTRS Tests =====

    #[test]
    fn test_dpttrf_dpttrs() {
        // SPD tridiagonal: [4 -1  0]
        //                  [-1 4 -1]
        //                  [0 -1  4]
        let d_orig = [4.0f64, 4.0, 4.0];
        let e_orig = [-1.0f64, -1.0];
        let b_orig = [3.0f64, 2.0, 3.0];

        let mut d = d_orig.clone();
        let mut e = e_orig.clone();
        let mut b = b_orig.clone();

        unsafe {
            // Factor
            let ret = oblas_dpttrf(3, d.as_mut_ptr(), e.as_mut_ptr());
            assert_eq!(ret, 0);

            // Solve
            let ret = oblas_dpttrs(3, 1, d.as_ptr(), e.as_ptr(), b.as_mut_ptr(), 3);
            assert_eq!(ret, 0);

            // Verify Ax = b
            let ax0 = d_orig[0] * b[0] + e_orig[0] * b[1];
            let ax1 = e_orig[0] * b[0] + d_orig[1] * b[1] + e_orig[1] * b[2];
            let ax2 = e_orig[1] * b[1] + d_orig[2] * b[2];

            assert!((ax0 - b_orig[0]).abs() < 1e-10);
            assert!((ax1 - b_orig[1]).abs() < 1e-10);
            assert!((ax2 - b_orig[2]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_spttrf_spttrs() {
        let d_orig = [4.0f32, 4.0, 4.0];
        let e_orig = [-1.0f32, -1.0];
        let b_orig = [3.0f32, 2.0, 3.0];

        let mut d = d_orig.clone();
        let mut e = e_orig.clone();
        let mut b = b_orig.clone();

        unsafe {
            // Factor
            let ret = oblas_spttrf(3, d.as_mut_ptr(), e.as_mut_ptr());
            assert_eq!(ret, 0);

            // Solve
            let ret = oblas_spttrs(3, 1, d.as_ptr(), e.as_ptr(), b.as_mut_ptr(), 3);
            assert_eq!(ret, 0);

            // Verify Ax = b (with lower precision)
            let ax0 = d_orig[0] * b[0] + e_orig[0] * b[1];
            let ax1 = e_orig[0] * b[0] + d_orig[1] * b[1] + e_orig[1] * b[2];
            let ax2 = e_orig[1] * b[1] + d_orig[2] * b[2];

            assert!((ax0 - b_orig[0]).abs() < 1e-5);
            assert!((ax1 - b_orig[1]).abs() < 1e-5);
            assert!((ax2 - b_orig[2]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dpttrf_multiple_rhs() {
        let d_orig = [4.0f64, 4.0, 4.0];
        let e_orig = [-1.0f64, -1.0];

        let mut d = d_orig.clone();
        let mut e = e_orig.clone();
        // Two RHS columns (column-major)
        let mut b = [3.0f64, 2.0, 3.0, 6.0, 4.0, 6.0];

        unsafe {
            // Factor
            let ret = oblas_dpttrf(3, d.as_mut_ptr(), e.as_mut_ptr());
            assert_eq!(ret, 0);

            // Solve both columns at once
            let ret = oblas_dpttrs(3, 2, d.as_ptr(), e.as_ptr(), b.as_mut_ptr(), 3);
            assert_eq!(ret, 0);

            // Second column should be 2x first column
            assert!((b[3] - 2.0 * b[0]).abs() < 1e-10);
            assert!((b[4] - 2.0 * b[1]).abs() < 1e-10);
            assert!((b[5] - 2.0 * b[2]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dpttrf_not_positive_definite() {
        // Not positive definite: first pivot is negative
        let mut d = [-4.0f64, 4.0, 4.0];
        let mut e = [-1.0f64, -1.0];

        unsafe {
            let ret = oblas_dpttrf(3, d.as_mut_ptr(), e.as_mut_ptr());
            assert_eq!(ret, 1); // Singular at index 0
        }
    }

    #[test]
    fn test_dpttrf_becomes_indefinite() {
        // Becomes indefinite during factorization
        let mut d = [1.0f64, 1.0, 1.0];
        let mut e = [2.0f64, 2.0]; // Off-diagonal too large

        unsafe {
            let ret = oblas_dpttrf(3, d.as_mut_ptr(), e.as_mut_ptr());
            assert!(ret > 0); // Should fail
        }
    }

    #[test]
    fn test_dpttrf_1x1() {
        let mut d = [4.0f64];

        unsafe {
            let ret = oblas_dpttrf(1, d.as_mut_ptr(), std::ptr::null_mut());
            assert_eq!(ret, 0);
            assert!((d[0] - 4.0).abs() < 1e-10);

            let mut b = [8.0f64];
            let ret = oblas_dpttrs(1, 1, d.as_ptr(), std::ptr::null(), b.as_mut_ptr(), 1);
            assert_eq!(ret, 0);
            assert!((b[0] - 2.0).abs() < 1e-10);
        }
    }
}
