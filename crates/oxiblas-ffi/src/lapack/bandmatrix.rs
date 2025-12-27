//! LAPACK FFI - Band Matrix Operations.
//!
//! This module provides C-compatible FFI for band matrix routines:
//!
//! - `oblas_sgbtrf`, `oblas_dgbtrf` - Band LU factorization
//! - `oblas_sgbtrs`, `oblas_dgbtrs` - Solve from band LU factors
//! - `oblas_sgbsv`, `oblas_dgbsv` - Direct band system solve

use crate::types::*;
use oxiblas_lapack::lu::BandLu;
use std::ffi::c_int;
use std::slice;

// =============================================================================
// SGBTRF - Band LU factorization (single precision)
// =============================================================================

/// Computes LU factorization of a general band matrix.
///
/// The band matrix A is stored in band storage format with 2*kl+ku+1 rows
/// and n columns (column-major order).
///
/// On output, AB contains the L and U factors, and ipiv contains the pivot indices.
///
/// # Arguments
///
/// * `n` - Order of the matrix A (n >= 0)
/// * `kl` - Number of sub-diagonals (kl >= 0)
/// * `ku` - Number of super-diagonals (ku >= 0)
/// * `ab` - Band matrix in column-major band storage (ldab × n)
/// * `ldab` - Leading dimension of ab (ldab >= 2*kl+ku+1)
/// * `ipiv` - Pivot indices (length n)
///
/// # Returns
///
/// * 0 on success
/// * -i if argument i had an illegal value
/// * i if U(i,i) = 0 (singular)
///
/// # Safety
/// - `ab` must point to a valid band storage array of size ldab * n
/// - `ipiv` must point to an array of n integers
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sgbtrf(
    n: c_int,
    kl: c_int,
    ku: c_int,
    ab: *mut f32,
    ldab: c_int,
    ipiv: *mut c_int,
) -> c_int {
    // Validate arguments
    if n < 0 {
        return -1;
    }
    if kl < 0 {
        return -2;
    }
    if ku < 0 {
        return -3;
    }
    if ab.is_null() {
        return -4;
    }
    if ldab < (2 * kl + ku + 1) {
        return -5;
    }
    if ipiv.is_null() {
        return -6;
    }

    if n == 0 {
        return OblasReturn::Success as c_int;
    }

    let n = n as usize;
    let kl = kl as usize;
    let ku = ku as usize;
    let ldab = ldab as usize;

    // Copy band storage to working array
    let expected_ldab = 2 * kl + ku + 1;
    let ab_slice = slice::from_raw_parts(ab, ldab * n);

    // Convert to expected format if ldab differs
    let ab_work: Vec<f32> = if ldab == expected_ldab {
        ab_slice.to_vec()
    } else {
        let mut work = vec![0.0f32; expected_ldab * n];
        for j in 0..n {
            for i in 0..expected_ldab {
                work[i + j * expected_ldab] = ab_slice[i + j * ldab];
            }
        }
        work
    };

    // Compute band LU
    match BandLu::compute(n, kl, ku, &ab_work) {
        Ok(lu_result) => {
            // Copy factored matrix back
            let factored = lu_result.ab();
            let ab_out = slice::from_raw_parts_mut(ab, ldab * n);
            for j in 0..n {
                for i in 0..expected_ldab {
                    ab_out[i + j * ldab] = factored[i + j * expected_ldab];
                }
            }

            // Copy pivot indices (1-based for LAPACK compatibility)
            let pivot = lu_result.pivot();
            let ipiv_slice = slice::from_raw_parts_mut(ipiv, n);
            for i in 0..n {
                ipiv_slice[i] = (pivot[i] + 1) as c_int;
            }

            OblasReturn::Success as c_int
        }
        Err(e) => match e {
            oxiblas_lapack::lu::BandLuError::Singular { index } => (index + 1) as c_int,
            _ => OblasReturn::InvalidArg as c_int,
        },
    }
}

// =============================================================================
// DGBTRF - Band LU factorization (double precision)
// =============================================================================

/// Computes LU factorization of a general band matrix (double precision).
///
/// # Safety
/// - `ab` must point to a valid band storage array of size ldab * n
/// - `ipiv` must point to an array of n integers
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dgbtrf(
    n: c_int,
    kl: c_int,
    ku: c_int,
    ab: *mut f64,
    ldab: c_int,
    ipiv: *mut c_int,
) -> c_int {
    if n < 0 {
        return -1;
    }
    if kl < 0 {
        return -2;
    }
    if ku < 0 {
        return -3;
    }
    if ab.is_null() {
        return -4;
    }
    if ldab < (2 * kl + ku + 1) {
        return -5;
    }
    if ipiv.is_null() {
        return -6;
    }

    if n == 0 {
        return OblasReturn::Success as c_int;
    }

    let n = n as usize;
    let kl = kl as usize;
    let ku = ku as usize;
    let ldab = ldab as usize;

    let expected_ldab = 2 * kl + ku + 1;
    let ab_slice = slice::from_raw_parts(ab, ldab * n);

    let ab_work: Vec<f64> = if ldab == expected_ldab {
        ab_slice.to_vec()
    } else {
        let mut work = vec![0.0f64; expected_ldab * n];
        for j in 0..n {
            for i in 0..expected_ldab {
                work[i + j * expected_ldab] = ab_slice[i + j * ldab];
            }
        }
        work
    };

    match BandLu::compute(n, kl, ku, &ab_work) {
        Ok(lu_result) => {
            let factored = lu_result.ab();
            let ab_out = slice::from_raw_parts_mut(ab, ldab * n);
            for j in 0..n {
                for i in 0..expected_ldab {
                    ab_out[i + j * ldab] = factored[i + j * expected_ldab];
                }
            }

            let pivot = lu_result.pivot();
            let ipiv_slice = slice::from_raw_parts_mut(ipiv, n);
            for i in 0..n {
                ipiv_slice[i] = (pivot[i] + 1) as c_int;
            }

            OblasReturn::Success as c_int
        }
        Err(e) => match e {
            oxiblas_lapack::lu::BandLuError::Singular { index } => (index + 1) as c_int,
            _ => OblasReturn::InvalidArg as c_int,
        },
    }
}

// =============================================================================
// SGBTRS - Solve band system from LU factors (single precision)
// =============================================================================

/// Solves a system of linear equations A*X = B using band LU factorization.
///
/// # Arguments
///
/// * `trans` - 'N' for A*X=B, 'T' for A^T*X=B, 'C' for A^H*X=B
/// * `n` - Order of the matrix A
/// * `kl` - Number of sub-diagonals
/// * `ku` - Number of super-diagonals
/// * `nrhs` - Number of right-hand sides (columns of B)
/// * `ab` - Factored band matrix from GBTRF
/// * `ldab` - Leading dimension of ab
/// * `ipiv` - Pivot indices from GBTRF
/// * `b` - On entry, RHS matrix B; on exit, solution X
/// * `ldb` - Leading dimension of b
///
/// # Safety
/// - `ab` must point to a valid factored band matrix
/// - `ipiv` must point to valid pivot indices from GBTRF
/// - `b` must point to a valid n × nrhs matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sgbtrs(
    trans: CChar,
    n: c_int,
    kl: c_int,
    ku: c_int,
    nrhs: c_int,
    ab: *const f32,
    ldab: c_int,
    ipiv: *const c_int,
    b: *mut f32,
    ldb: c_int,
) -> c_int {
    // Validate arguments
    let trans_upper = (trans as u8 as char).to_ascii_uppercase();
    if trans_upper != 'N' && trans_upper != 'T' && trans_upper != 'C' {
        return -1;
    }
    if n < 0 {
        return -2;
    }
    if kl < 0 {
        return -3;
    }
    if ku < 0 {
        return -4;
    }
    if nrhs < 0 {
        return -5;
    }
    if ab.is_null() {
        return -6;
    }
    if ldab < (2 * kl + ku + 1) {
        return -7;
    }
    if ipiv.is_null() {
        return -8;
    }
    if b.is_null() {
        return -9;
    }
    if ldb < n.max(1) {
        return -10;
    }

    if n == 0 || nrhs == 0 {
        return OblasReturn::Success as c_int;
    }

    // Only non-transposed solve is currently supported
    if trans_upper != 'N' {
        return OblasReturn::NotConverged as c_int;
    }

    let n = n as usize;
    let kl = kl as usize;
    let ku = ku as usize;
    let nrhs = nrhs as usize;
    let ldab = ldab as usize;
    let ldb = ldb as usize;

    let expected_ldab = 2 * kl + ku + 1;
    let ab_slice = slice::from_raw_parts(ab, ldab * n);

    // Reconstruct BandLu from factored matrix
    let ab_work: Vec<f32> = if ldab == expected_ldab {
        ab_slice.to_vec()
    } else {
        let mut work = vec![0.0f32; expected_ldab * n];
        for j in 0..n {
            for i in 0..expected_ldab {
                work[i + j * expected_ldab] = ab_slice[i + j * ldab];
            }
        }
        work
    };

    // Read pivot indices (convert from 1-based to 0-based)
    let ipiv_slice = slice::from_raw_parts(ipiv, n);
    let pivot: Vec<usize> = ipiv_slice.iter().map(|&p| (p - 1) as usize).collect();

    // Read B and solve
    let b_slice = slice::from_raw_parts_mut(b, ldb * nrhs);

    // For each RHS
    for rhs in 0..nrhs {
        // Extract column from B (column-major)
        let mut x: Vec<f32> = (0..n).map(|i| b_slice[i + rhs * ldb]).collect();

        // Apply row permutations (forward)
        for k in 0..n {
            let pk = pivot[k];
            if k != pk {
                x.swap(k, pk);
            }
        }

        // Forward substitution: Ly = Pb
        for j in 0..n {
            let km = kl.min(n - 1 - j);
            for i in 1..=km {
                let row_in_band = kl + ku + i;
                let l_elem = ab_work[row_in_band + j * expected_ldab];
                x[j + i] = x[j + i] - l_elem * x[j];
            }
        }

        // Back substitution: Ux = y
        for j in (0..n).rev() {
            let row_diag = kl + ku;
            let diag = ab_work[row_diag + j * expected_ldab];
            x[j] = x[j] / diag;

            let kmax = ku + kl;
            for i in j.saturating_sub(kmax)..j {
                let diag_dist = j as isize - i as isize;
                let row_in_band = (kl + ku) as isize - diag_dist;
                if row_in_band >= 0 {
                    let u_elem = ab_work[row_in_band as usize + j * expected_ldab];
                    x[i] = x[i] - u_elem * x[j];
                }
            }
        }

        // Write back solution
        for i in 0..n {
            b_slice[i + rhs * ldb] = x[i];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DGBTRS - Solve band system from LU factors (double precision)
// =============================================================================

/// Solves a system of linear equations A*X = B using band LU factorization (double precision).
///
/// # Safety
/// - `ab` must point to a valid factored band matrix
/// - `ipiv` must point to valid pivot indices from GBTRF
/// - `b` must point to a valid n × nrhs matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dgbtrs(
    trans: CChar,
    n: c_int,
    kl: c_int,
    ku: c_int,
    nrhs: c_int,
    ab: *const f64,
    ldab: c_int,
    ipiv: *const c_int,
    b: *mut f64,
    ldb: c_int,
) -> c_int {
    let trans_upper = (trans as u8 as char).to_ascii_uppercase();
    if trans_upper != 'N' && trans_upper != 'T' && trans_upper != 'C' {
        return -1;
    }
    if n < 0 {
        return -2;
    }
    if kl < 0 {
        return -3;
    }
    if ku < 0 {
        return -4;
    }
    if nrhs < 0 {
        return -5;
    }
    if ab.is_null() {
        return -6;
    }
    if ldab < (2 * kl + ku + 1) {
        return -7;
    }
    if ipiv.is_null() {
        return -8;
    }
    if b.is_null() {
        return -9;
    }
    if ldb < n.max(1) {
        return -10;
    }

    if n == 0 || nrhs == 0 {
        return OblasReturn::Success as c_int;
    }

    if trans_upper != 'N' {
        return OblasReturn::NotConverged as c_int;
    }

    let n = n as usize;
    let kl = kl as usize;
    let ku = ku as usize;
    let nrhs = nrhs as usize;
    let ldab = ldab as usize;
    let ldb = ldb as usize;

    let expected_ldab = 2 * kl + ku + 1;
    let ab_slice = slice::from_raw_parts(ab, ldab * n);

    let ab_work: Vec<f64> = if ldab == expected_ldab {
        ab_slice.to_vec()
    } else {
        let mut work = vec![0.0f64; expected_ldab * n];
        for j in 0..n {
            for i in 0..expected_ldab {
                work[i + j * expected_ldab] = ab_slice[i + j * ldab];
            }
        }
        work
    };

    let ipiv_slice = slice::from_raw_parts(ipiv, n);
    let pivot: Vec<usize> = ipiv_slice.iter().map(|&p| (p - 1) as usize).collect();

    let b_slice = slice::from_raw_parts_mut(b, ldb * nrhs);

    for rhs in 0..nrhs {
        let mut x: Vec<f64> = (0..n).map(|i| b_slice[i + rhs * ldb]).collect();

        for k in 0..n {
            let pk = pivot[k];
            if k != pk {
                x.swap(k, pk);
            }
        }

        for j in 0..n {
            let km = kl.min(n - 1 - j);
            for i in 1..=km {
                let row_in_band = kl + ku + i;
                let l_elem = ab_work[row_in_band + j * expected_ldab];
                x[j + i] = x[j + i] - l_elem * x[j];
            }
        }

        for j in (0..n).rev() {
            let row_diag = kl + ku;
            let diag = ab_work[row_diag + j * expected_ldab];
            x[j] = x[j] / diag;

            let kmax = ku + kl;
            for i in j.saturating_sub(kmax)..j {
                let diag_dist = j as isize - i as isize;
                let row_in_band = (kl + ku) as isize - diag_dist;
                if row_in_band >= 0 {
                    let u_elem = ab_work[row_in_band as usize + j * expected_ldab];
                    x[i] = x[i] - u_elem * x[j];
                }
            }
        }

        for i in 0..n {
            b_slice[i + rhs * ldb] = x[i];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// SGBSV - Direct band system solve (single precision)
// =============================================================================

/// Solves a general band system A*X = B by computing LU factorization and solving.
///
/// This is a convenience function that combines GBTRF and GBTRS.
///
/// # Arguments
///
/// * `n` - Order of the matrix A
/// * `kl` - Number of sub-diagonals
/// * `ku` - Number of super-diagonals
/// * `nrhs` - Number of right-hand sides
/// * `ab` - Band matrix (modified on exit to contain LU factors)
/// * `ldab` - Leading dimension of ab
/// * `ipiv` - Pivot indices (output)
/// * `b` - On entry, RHS; on exit, solution
/// * `ldb` - Leading dimension of b
///
/// # Safety
/// - `ab` must point to a valid band storage array
/// - `ipiv` must point to an array of n integers
/// - `b` must point to a valid n × nrhs matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sgbsv(
    n: c_int,
    kl: c_int,
    ku: c_int,
    nrhs: c_int,
    ab: *mut f32,
    ldab: c_int,
    ipiv: *mut c_int,
    b: *mut f32,
    ldb: c_int,
) -> c_int {
    // First, compute LU factorization
    let info = oblas_sgbtrf(n, kl, ku, ab, ldab, ipiv);
    if info != 0 {
        return info;
    }

    // Then solve
    oblas_sgbtrs(
        b'N' as CChar,
        n,
        kl,
        ku,
        nrhs,
        ab as *const f32,
        ldab,
        ipiv as *const c_int,
        b,
        ldb,
    )
}

// =============================================================================
// DGBSV - Direct band system solve (double precision)
// =============================================================================

/// Solves a general band system A*X = B (double precision).
///
/// # Safety
/// - `ab` must point to a valid band storage array
/// - `ipiv` must point to an array of n integers
/// - `b` must point to a valid n × nrhs matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dgbsv(
    n: c_int,
    kl: c_int,
    ku: c_int,
    nrhs: c_int,
    ab: *mut f64,
    ldab: c_int,
    ipiv: *mut c_int,
    b: *mut f64,
    ldb: c_int,
) -> c_int {
    let info = oblas_dgbtrf(n, kl, ku, ab, ldab, ipiv);
    if info != 0 {
        return info;
    }

    oblas_dgbtrs(
        b'N' as CChar,
        n,
        kl,
        ku,
        nrhs,
        ab as *const f64,
        ldab,
        ipiv as *const c_int,
        b,
        ldb,
    )
}

// =============================================================================
// SPBTRF - Band Cholesky factorization (single precision)
// =============================================================================

/// Computes Cholesky factorization of a symmetric positive definite band matrix.
///
/// The band matrix A is stored in lower triangular band storage format with
/// kd+1 rows and n columns (column-major order).
///
/// On output, AB contains the L factor in lower triangular band storage.
///
/// # Arguments
///
/// * `uplo` - 'L' for lower triangular storage (required)
/// * `n` - Order of the matrix A (n >= 0)
/// * `kd` - Number of sub-diagonals (kd >= 0)
/// * `ab` - Band matrix in column-major band storage (ldab × n)
/// * `ldab` - Leading dimension of ab (ldab >= kd+1)
///
/// # Returns
///
/// * 0 on success
/// * -i if argument i had an illegal value
/// * i if the leading minor of order i is not positive definite
///
/// # Safety
/// - `ab` must point to a valid band storage array of size ldab * n
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_spbtrf(
    uplo: CChar,
    n: c_int,
    kd: c_int,
    ab: *mut f32,
    ldab: c_int,
) -> c_int {
    use oxiblas_lapack::cholesky::BandCholesky;

    let uplo_upper = (uplo as u8 as char).to_ascii_uppercase();
    if uplo_upper != 'L' && uplo_upper != 'U' {
        return -1;
    }
    if n < 0 {
        return -2;
    }
    if kd < 0 {
        return -3;
    }
    if ab.is_null() {
        return -4;
    }
    if ldab < (kd + 1) {
        return -5;
    }

    // Only lower triangular storage supported
    if uplo_upper != 'L' {
        return OblasReturn::InvalidArg as c_int;
    }

    if n == 0 {
        return OblasReturn::Success as c_int;
    }

    let n = n as usize;
    let kd = kd as usize;
    let ldab = ldab as usize;

    let expected_ldab = kd + 1;
    let ab_slice = slice::from_raw_parts(ab, ldab * n);

    let ab_work: Vec<f32> = if ldab == expected_ldab {
        ab_slice.to_vec()
    } else {
        let mut work = vec![0.0f32; expected_ldab * n];
        for j in 0..n {
            for i in 0..expected_ldab {
                work[i + j * expected_ldab] = ab_slice[i + j * ldab];
            }
        }
        work
    };

    match BandCholesky::compute(n, kd, &ab_work) {
        Ok(chol) => {
            let factored = chol.ab();
            let ab_out = slice::from_raw_parts_mut(ab, ldab * n);
            for j in 0..n {
                for i in 0..expected_ldab {
                    ab_out[i + j * ldab] = factored[i + j * expected_ldab];
                }
            }
            OblasReturn::Success as c_int
        }
        Err(e) => match e {
            oxiblas_lapack::cholesky::BandCholeskyError::NotPositiveDefinite { index } => {
                (index + 1) as c_int
            }
            _ => OblasReturn::InvalidArg as c_int,
        },
    }
}

// =============================================================================
// DPBTRF - Band Cholesky factorization (double precision)
// =============================================================================

/// Computes Cholesky factorization of a symmetric positive definite band matrix (double precision).
///
/// # Safety
/// - `ab` must point to a valid band storage array of size ldab * n
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dpbtrf(
    uplo: CChar,
    n: c_int,
    kd: c_int,
    ab: *mut f64,
    ldab: c_int,
) -> c_int {
    use oxiblas_lapack::cholesky::BandCholesky;

    let uplo_upper = (uplo as u8 as char).to_ascii_uppercase();
    if uplo_upper != 'L' && uplo_upper != 'U' {
        return -1;
    }
    if n < 0 {
        return -2;
    }
    if kd < 0 {
        return -3;
    }
    if ab.is_null() {
        return -4;
    }
    if ldab < (kd + 1) {
        return -5;
    }

    if uplo_upper != 'L' {
        return OblasReturn::InvalidArg as c_int;
    }

    if n == 0 {
        return OblasReturn::Success as c_int;
    }

    let n = n as usize;
    let kd = kd as usize;
    let ldab = ldab as usize;

    let expected_ldab = kd + 1;
    let ab_slice = slice::from_raw_parts(ab, ldab * n);

    let ab_work: Vec<f64> = if ldab == expected_ldab {
        ab_slice.to_vec()
    } else {
        let mut work = vec![0.0f64; expected_ldab * n];
        for j in 0..n {
            for i in 0..expected_ldab {
                work[i + j * expected_ldab] = ab_slice[i + j * ldab];
            }
        }
        work
    };

    match BandCholesky::compute(n, kd, &ab_work) {
        Ok(chol) => {
            let factored = chol.ab();
            let ab_out = slice::from_raw_parts_mut(ab, ldab * n);
            for j in 0..n {
                for i in 0..expected_ldab {
                    ab_out[i + j * ldab] = factored[i + j * expected_ldab];
                }
            }
            OblasReturn::Success as c_int
        }
        Err(e) => match e {
            oxiblas_lapack::cholesky::BandCholeskyError::NotPositiveDefinite { index } => {
                (index + 1) as c_int
            }
            _ => OblasReturn::InvalidArg as c_int,
        },
    }
}

// =============================================================================
// SPBTRS - Solve band SPD system from Cholesky factors (single precision)
// =============================================================================

/// Solves a system of linear equations A*X = B using band Cholesky factorization.
///
/// # Arguments
///
/// * `uplo` - 'L' for lower triangular storage
/// * `n` - Order of the matrix A
/// * `kd` - Number of sub-diagonals
/// * `nrhs` - Number of right-hand sides
/// * `ab` - Factored band matrix from PBTRF
/// * `ldab` - Leading dimension of ab
/// * `b` - On entry, RHS matrix B; on exit, solution X
/// * `ldb` - Leading dimension of b
///
/// # Safety
/// - `ab` must point to a valid factored band matrix
/// - `b` must point to a valid n × nrhs matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_spbtrs(
    uplo: CChar,
    n: c_int,
    kd: c_int,
    nrhs: c_int,
    ab: *const f32,
    ldab: c_int,
    b: *mut f32,
    ldb: c_int,
) -> c_int {
    let uplo_upper = (uplo as u8 as char).to_ascii_uppercase();
    if uplo_upper != 'L' && uplo_upper != 'U' {
        return -1;
    }
    if n < 0 {
        return -2;
    }
    if kd < 0 {
        return -3;
    }
    if nrhs < 0 {
        return -4;
    }
    if ab.is_null() {
        return -5;
    }
    if ldab < (kd + 1) {
        return -6;
    }
    if b.is_null() {
        return -7;
    }
    if ldb < n.max(1) {
        return -8;
    }

    if uplo_upper != 'L' {
        return OblasReturn::InvalidArg as c_int;
    }

    if n == 0 || nrhs == 0 {
        return OblasReturn::Success as c_int;
    }

    let n = n as usize;
    let kd = kd as usize;
    let nrhs = nrhs as usize;
    let ldab = ldab as usize;
    let ldb = ldb as usize;

    let expected_ldab = kd + 1;
    let ab_slice = slice::from_raw_parts(ab, ldab * n);

    let ab_work: Vec<f32> = if ldab == expected_ldab {
        ab_slice.to_vec()
    } else {
        let mut work = vec![0.0f32; expected_ldab * n];
        for j in 0..n {
            for i in 0..expected_ldab {
                work[i + j * expected_ldab] = ab_slice[i + j * ldab];
            }
        }
        work
    };

    let b_slice = slice::from_raw_parts_mut(b, ldb * nrhs);

    // Solve for each RHS
    for rhs in 0..nrhs {
        let mut x: Vec<f32> = (0..n).map(|i| b_slice[i + rhs * ldb]).collect();

        // Forward substitution: Ly = b
        for j in 0..n {
            x[j] = x[j] / ab_work[j * expected_ldab];
            let i_end = (j + kd).min(n - 1);
            for i in (j + 1)..=i_end {
                let l_ij = ab_work[(i - j) + j * expected_ldab];
                x[i] = x[i] - l_ij * x[j];
            }
        }

        // Back substitution: L^T x = y
        for j in (0..n).rev() {
            x[j] = x[j] / ab_work[j * expected_ldab];
            let i_start = j.saturating_sub(kd);
            for i in i_start..j {
                let l_ji = ab_work[(j - i) + i * expected_ldab];
                x[i] = x[i] - l_ji * x[j];
            }
        }

        // Write back
        for i in 0..n {
            b_slice[i + rhs * ldb] = x[i];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DPBTRS - Solve band SPD system from Cholesky factors (double precision)
// =============================================================================

/// Solves a system of linear equations A*X = B using band Cholesky factorization (double precision).
///
/// # Safety
/// - `ab` must point to a valid factored band matrix
/// - `b` must point to a valid n × nrhs matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dpbtrs(
    uplo: CChar,
    n: c_int,
    kd: c_int,
    nrhs: c_int,
    ab: *const f64,
    ldab: c_int,
    b: *mut f64,
    ldb: c_int,
) -> c_int {
    let uplo_upper = (uplo as u8 as char).to_ascii_uppercase();
    if uplo_upper != 'L' && uplo_upper != 'U' {
        return -1;
    }
    if n < 0 {
        return -2;
    }
    if kd < 0 {
        return -3;
    }
    if nrhs < 0 {
        return -4;
    }
    if ab.is_null() {
        return -5;
    }
    if ldab < (kd + 1) {
        return -6;
    }
    if b.is_null() {
        return -7;
    }
    if ldb < n.max(1) {
        return -8;
    }

    if uplo_upper != 'L' {
        return OblasReturn::InvalidArg as c_int;
    }

    if n == 0 || nrhs == 0 {
        return OblasReturn::Success as c_int;
    }

    let n = n as usize;
    let kd = kd as usize;
    let nrhs = nrhs as usize;
    let ldab = ldab as usize;
    let ldb = ldb as usize;

    let expected_ldab = kd + 1;
    let ab_slice = slice::from_raw_parts(ab, ldab * n);

    let ab_work: Vec<f64> = if ldab == expected_ldab {
        ab_slice.to_vec()
    } else {
        let mut work = vec![0.0f64; expected_ldab * n];
        for j in 0..n {
            for i in 0..expected_ldab {
                work[i + j * expected_ldab] = ab_slice[i + j * ldab];
            }
        }
        work
    };

    let b_slice = slice::from_raw_parts_mut(b, ldb * nrhs);

    for rhs in 0..nrhs {
        let mut x: Vec<f64> = (0..n).map(|i| b_slice[i + rhs * ldb]).collect();

        // Forward substitution
        for j in 0..n {
            x[j] = x[j] / ab_work[j * expected_ldab];
            let i_end = (j + kd).min(n - 1);
            for i in (j + 1)..=i_end {
                let l_ij = ab_work[(i - j) + j * expected_ldab];
                x[i] = x[i] - l_ij * x[j];
            }
        }

        // Back substitution
        for j in (0..n).rev() {
            x[j] = x[j] / ab_work[j * expected_ldab];
            let i_start = j.saturating_sub(kd);
            for i in i_start..j {
                let l_ji = ab_work[(j - i) + i * expected_ldab];
                x[i] = x[i] - l_ji * x[j];
            }
        }

        for i in 0..n {
            b_slice[i + rhs * ldb] = x[i];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// SPBSV - Direct band SPD system solve (single precision)
// =============================================================================

/// Solves a band SPD system A*X = B by computing Cholesky factorization and solving.
///
/// # Safety
/// - `ab` must point to a valid band storage array
/// - `b` must point to a valid n × nrhs matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_spbsv(
    uplo: CChar,
    n: c_int,
    kd: c_int,
    nrhs: c_int,
    ab: *mut f32,
    ldab: c_int,
    b: *mut f32,
    ldb: c_int,
) -> c_int {
    let info = oblas_spbtrf(uplo, n, kd, ab, ldab);
    if info != 0 {
        return info;
    }

    oblas_spbtrs(uplo, n, kd, nrhs, ab as *const f32, ldab, b, ldb)
}

// =============================================================================
// DPBSV - Direct band SPD system solve (double precision)
// =============================================================================

/// Solves a band SPD system A*X = B (double precision).
///
/// # Safety
/// - `ab` must point to a valid band storage array
/// - `b` must point to a valid n × nrhs matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dpbsv(
    uplo: CChar,
    n: c_int,
    kd: c_int,
    nrhs: c_int,
    ab: *mut f64,
    ldab: c_int,
    b: *mut f64,
    ldb: c_int,
) -> c_int {
    let info = oblas_dpbtrf(uplo, n, kd, ab, ldab);
    if info != 0 {
        return info;
    }

    oblas_dpbtrs(uplo, n, kd, nrhs, ab as *const f64, ldab, b, ldb)
}

// Type alias for C char
type CChar = i8;

#[cfg(test)]
mod tests {
    use super::*;
    use oxiblas_lapack::cholesky::dense_to_band_lower;
    use oxiblas_lapack::lu::dense_to_band;

    #[test]
    fn test_dgbtrf_tridiagonal() {
        // Tridiagonal matrix (kl=1, ku=1)
        // [4 -1  0  0]
        // [-1 4 -1  0]
        // [0 -1  4 -1]
        // [0  0 -1  4]
        let n = 4;
        let kl = 1;
        let ku = 1;
        let ldab = 2 * kl + ku + 1;

        #[rustfmt::skip]
        let a_dense: Vec<f64> = vec![
            4.0, -1.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, 0.0,
            0.0, -1.0, 4.0, -1.0,
            0.0, 0.0, -1.0, 4.0,
        ];

        let mut ab = dense_to_band(&a_dense, n, kl, ku);
        let mut ipiv = vec![0i32; n];

        unsafe {
            let info = oblas_dgbtrf(
                n as c_int,
                kl as c_int,
                ku as c_int,
                ab.as_mut_ptr(),
                ldab as c_int,
                ipiv.as_mut_ptr(),
            );
            assert_eq!(info, 0, "GBTRF failed with info = {info}");
        }
    }

    #[test]
    fn test_dgbsv_tridiagonal() {
        // Same tridiagonal matrix
        let n = 4;
        let kl = 1;
        let ku = 1;
        let ldab = 2 * kl + ku + 1;

        #[rustfmt::skip]
        let a_dense: Vec<f64> = vec![
            4.0, -1.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, 0.0,
            0.0, -1.0, 4.0, -1.0,
            0.0, 0.0, -1.0, 4.0,
        ];

        let mut ab = dense_to_band(&a_dense, n, kl, ku);
        let mut ipiv = vec![0i32; n];

        // RHS: b = [3, 2, 2, 3] -> x = [1, 1, 1, 1]
        let mut b = vec![3.0f64, 2.0, 2.0, 3.0];
        let nrhs = 1;
        let ldb = n;

        unsafe {
            let info = oblas_dgbsv(
                n as c_int,
                kl as c_int,
                ku as c_int,
                nrhs as c_int,
                ab.as_mut_ptr(),
                ldab as c_int,
                ipiv.as_mut_ptr(),
                b.as_mut_ptr(),
                ldb as c_int,
            );
            assert_eq!(info, 0, "GBSV failed with info = {info}");

            // Check solution
            for i in 0..n {
                assert!(
                    (b[i] - 1.0).abs() < 1e-10,
                    "x[{i}] = {}, expected 1.0",
                    b[i]
                );
            }
        }
    }

    #[test]
    fn test_dgbsv_pentadiagonal() {
        // Pentadiagonal matrix (kl=2, ku=2)
        let n = 5;
        let kl = 2;
        let ku = 2;
        let ldab = 2 * kl + ku + 1;

        #[rustfmt::skip]
        let a_dense: Vec<f64> = vec![
            10.0, -1.0, -2.0,  0.0,  0.0,
            -1.0, 10.0, -1.0, -2.0,  0.0,
            -2.0, -1.0, 10.0, -1.0, -2.0,
             0.0, -2.0, -1.0, 10.0, -1.0,
             0.0,  0.0, -2.0, -1.0, 10.0,
        ];

        let mut ab = dense_to_band(&a_dense, n, kl, ku);
        let mut ipiv = vec![0i32; n];

        // Create RHS such that x = [1, 1, 1, 1, 1]
        // b[i] = sum_j A[i,j] * 1 = row sum
        let b_true = vec![7.0f64, 6.0, 4.0, 6.0, 7.0];
        let mut b = b_true.clone();
        let nrhs = 1;
        let ldb = n;

        unsafe {
            let info = oblas_dgbsv(
                n as c_int,
                kl as c_int,
                ku as c_int,
                nrhs as c_int,
                ab.as_mut_ptr(),
                ldab as c_int,
                ipiv.as_mut_ptr(),
                b.as_mut_ptr(),
                ldb as c_int,
            );
            assert_eq!(info, 0, "GBSV failed with info = {info}");

            // Verify solution by computing Ax
            for i in 0..n {
                let mut ax_i = 0.0;
                for j in 0..n {
                    ax_i += a_dense[i * n + j] * b[j];
                }
                assert!(
                    (ax_i - b_true[i]).abs() < 1e-8,
                    "Ax[{i}] = {ax_i}, expected {}",
                    b_true[i]
                );
            }
        }
    }

    #[test]
    fn test_sgbtrf() {
        let n = 3;
        let kl = 1;
        let ku = 1;
        let ldab = 2 * kl + ku + 1;

        #[rustfmt::skip]
        let a_dense: Vec<f32> = vec![
            4.0, -1.0, 0.0,
            -1.0, 4.0, -1.0,
            0.0, -1.0, 4.0,
        ];

        let mut ab = dense_to_band(&a_dense, n, kl, ku);
        let mut ipiv = vec![0i32; n];

        unsafe {
            let info = oblas_sgbtrf(
                n as c_int,
                kl as c_int,
                ku as c_int,
                ab.as_mut_ptr(),
                ldab as c_int,
                ipiv.as_mut_ptr(),
            );
            assert_eq!(info, 0);
        }
    }

    #[test]
    fn test_dgbtrs_multiple_rhs() {
        let n = 4;
        let kl = 1;
        let ku = 1;
        let ldab = 2 * kl + ku + 1;

        #[rustfmt::skip]
        let a_dense: Vec<f64> = vec![
            4.0, -1.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, 0.0,
            0.0, -1.0, 4.0, -1.0,
            0.0, 0.0, -1.0, 4.0,
        ];

        let mut ab = dense_to_band(&a_dense, n, kl, ku);
        let mut ipiv = vec![0i32; n];

        // First factorize
        unsafe {
            let info = oblas_dgbtrf(
                n as c_int,
                kl as c_int,
                ku as c_int,
                ab.as_mut_ptr(),
                ldab as c_int,
                ipiv.as_mut_ptr(),
            );
            assert_eq!(info, 0);
        }

        // Two RHS (column-major)
        let nrhs = 2;
        let ldb = n;
        // b1 = [3, 2, 2, 3] -> x1 = [1, 1, 1, 1]
        // b2 = [6, 4, 4, 6] -> x2 = [2, 2, 2, 2]
        #[rustfmt::skip]
        let mut b = vec![
            3.0, 2.0, 2.0, 3.0,  // column 1
            6.0, 4.0, 4.0, 6.0,  // column 2
        ];

        unsafe {
            let info = oblas_dgbtrs(
                b'N' as CChar,
                n as c_int,
                kl as c_int,
                ku as c_int,
                nrhs as c_int,
                ab.as_ptr(),
                ldab as c_int,
                ipiv.as_ptr(),
                b.as_mut_ptr(),
                ldb as c_int,
            );
            assert_eq!(info, 0);

            // Check solutions
            for i in 0..n {
                assert!(
                    (b[i] - 1.0).abs() < 1e-10,
                    "x1[{i}] = {}, expected 1.0",
                    b[i]
                );
                assert!(
                    (b[i + ldb] - 2.0).abs() < 1e-10,
                    "x2[{i}] = {}, expected 2.0",
                    b[i + ldb]
                );
            }
        }
    }

    #[test]
    fn test_dgbtrf_singular() {
        // Singular matrix
        let n = 3;
        let kl = 1;
        let ku = 1;
        let ldab = 2 * kl + ku + 1;

        #[rustfmt::skip]
        let a_dense: Vec<f64> = vec![
            1.0, -1.0, 0.0,
            -1.0, 1.0, 0.0,  // Row 2 = -Row 1
            0.0, 0.0, 1.0,
        ];

        let mut ab = dense_to_band(&a_dense, n, kl, ku);
        let mut ipiv = vec![0i32; n];

        unsafe {
            let info = oblas_dgbtrf(
                n as c_int,
                kl as c_int,
                ku as c_int,
                ab.as_mut_ptr(),
                ldab as c_int,
                ipiv.as_mut_ptr(),
            );
            // Should return positive value indicating singular at some index
            assert!(info > 0, "Expected singular error, got {info}");
        }
    }

    // ==========================================================================
    // Band Cholesky tests
    // ==========================================================================

    #[test]
    fn test_dpbtrf_tridiagonal() {
        // SPD tridiagonal matrix (kd=1)
        let n = 4;
        let kd = 1;
        let ldab = kd + 1;

        #[rustfmt::skip]
        let a_dense: Vec<f64> = vec![
            4.0, -1.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, 0.0,
            0.0, -1.0, 4.0, -1.0,
            0.0, 0.0, -1.0, 4.0,
        ];

        let mut ab = dense_to_band_lower(&a_dense, n, kd);

        unsafe {
            let info = oblas_dpbtrf(
                b'L' as CChar,
                n as c_int,
                kd as c_int,
                ab.as_mut_ptr(),
                ldab as c_int,
            );
            assert_eq!(info, 0, "PBTRF failed with info = {info}");
        }
    }

    #[test]
    fn test_dpbsv_tridiagonal() {
        let n = 4;
        let kd = 1;
        let ldab = kd + 1;

        #[rustfmt::skip]
        let a_dense: Vec<f64> = vec![
            4.0, -1.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, 0.0,
            0.0, -1.0, 4.0, -1.0,
            0.0, 0.0, -1.0, 4.0,
        ];

        let mut ab = dense_to_band_lower(&a_dense, n, kd);

        // RHS: b = [3, 2, 2, 3] -> x = [1, 1, 1, 1]
        let mut b = vec![3.0f64, 2.0, 2.0, 3.0];
        let nrhs = 1;
        let ldb = n;

        unsafe {
            let info = oblas_dpbsv(
                b'L' as CChar,
                n as c_int,
                kd as c_int,
                nrhs as c_int,
                ab.as_mut_ptr(),
                ldab as c_int,
                b.as_mut_ptr(),
                ldb as c_int,
            );
            assert_eq!(info, 0, "PBSV failed with info = {info}");

            for i in 0..n {
                assert!(
                    (b[i] - 1.0).abs() < 1e-10,
                    "x[{i}] = {}, expected 1.0",
                    b[i]
                );
            }
        }
    }

    #[test]
    fn test_dpbsv_pentadiagonal() {
        let n = 5;
        let kd = 2;
        let ldab = kd + 1;

        #[rustfmt::skip]
        let a_dense: Vec<f64> = vec![
            10.0, -1.0, -2.0,  0.0,  0.0,
            -1.0, 10.0, -1.0, -2.0,  0.0,
            -2.0, -1.0, 10.0, -1.0, -2.0,
             0.0, -2.0, -1.0, 10.0, -1.0,
             0.0,  0.0, -2.0, -1.0, 10.0,
        ];

        let mut ab = dense_to_band_lower(&a_dense, n, kd);
        let b_true = vec![7.0f64, 6.0, 4.0, 6.0, 7.0];
        let mut b = b_true.clone();
        let nrhs = 1;
        let ldb = n;

        unsafe {
            let info = oblas_dpbsv(
                b'L' as CChar,
                n as c_int,
                kd as c_int,
                nrhs as c_int,
                ab.as_mut_ptr(),
                ldab as c_int,
                b.as_mut_ptr(),
                ldb as c_int,
            );
            assert_eq!(info, 0, "PBSV failed with info = {info}");

            // Verify by computing Ax
            for i in 0..n {
                let mut ax_i = 0.0;
                for j in 0..n {
                    ax_i += a_dense[i * n + j] * b[j];
                }
                assert!(
                    (ax_i - b_true[i]).abs() < 1e-8,
                    "Ax[{i}] = {ax_i}, expected {}",
                    b_true[i]
                );
            }
        }
    }

    #[test]
    fn test_spbtrf() {
        let n = 3;
        let kd = 1;
        let ldab = kd + 1;

        #[rustfmt::skip]
        let a_dense: Vec<f32> = vec![
            4.0, -1.0, 0.0,
            -1.0, 4.0, -1.0,
            0.0, -1.0, 4.0,
        ];

        let mut ab = dense_to_band_lower(&a_dense, n, kd);

        unsafe {
            let info = oblas_spbtrf(
                b'L' as CChar,
                n as c_int,
                kd as c_int,
                ab.as_mut_ptr(),
                ldab as c_int,
            );
            assert_eq!(info, 0);
        }
    }

    #[test]
    fn test_dpbtrs_multiple_rhs() {
        let n = 4;
        let kd = 1;
        let ldab = kd + 1;

        #[rustfmt::skip]
        let a_dense: Vec<f64> = vec![
            4.0, -1.0, 0.0, 0.0,
            -1.0, 4.0, -1.0, 0.0,
            0.0, -1.0, 4.0, -1.0,
            0.0, 0.0, -1.0, 4.0,
        ];

        let mut ab = dense_to_band_lower(&a_dense, n, kd);

        // First factorize
        unsafe {
            let info = oblas_dpbtrf(
                b'L' as CChar,
                n as c_int,
                kd as c_int,
                ab.as_mut_ptr(),
                ldab as c_int,
            );
            assert_eq!(info, 0);
        }

        // Two RHS (column-major)
        let nrhs = 2;
        let ldb = n;
        #[rustfmt::skip]
        let mut b = vec![
            3.0, 2.0, 2.0, 3.0,  // column 1 -> x = [1,1,1,1]
            6.0, 4.0, 4.0, 6.0,  // column 2 -> x = [2,2,2,2]
        ];

        unsafe {
            let info = oblas_dpbtrs(
                b'L' as CChar,
                n as c_int,
                kd as c_int,
                nrhs as c_int,
                ab.as_ptr(),
                ldab as c_int,
                b.as_mut_ptr(),
                ldb as c_int,
            );
            assert_eq!(info, 0);

            for i in 0..n {
                assert!(
                    (b[i] - 1.0).abs() < 1e-10,
                    "x1[{i}] = {}, expected 1.0",
                    b[i]
                );
                assert!(
                    (b[i + ldb] - 2.0).abs() < 1e-10,
                    "x2[{i}] = {}, expected 2.0",
                    b[i + ldb]
                );
            }
        }
    }

    #[test]
    fn test_dpbtrf_not_spd() {
        // Not positive definite
        let n = 3;
        let kd = 1;
        let ldab = kd + 1;

        #[rustfmt::skip]
        let a_dense: Vec<f64> = vec![
            1.0, -2.0, 0.0,
            -2.0, 1.0, -2.0,
            0.0, -2.0, 1.0,
        ];

        let mut ab = dense_to_band_lower(&a_dense, n, kd);

        unsafe {
            let info = oblas_dpbtrf(
                b'L' as CChar,
                n as c_int,
                kd as c_int,
                ab.as_mut_ptr(),
                ldab as c_int,
            );
            assert!(info > 0, "Expected not positive definite error, got {info}");
        }
    }
}
