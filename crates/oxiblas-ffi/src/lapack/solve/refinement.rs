//! LAPACK FFI - Linear system solve routines.

use crate::types::*;
use oxiblas_lapack::lu;
use oxiblas_matrix::Mat;
use std::ffi::c_int;
use std::slice;

// =============================================================================
// SGERFS - Iterative refinement for general systems (single precision)
// =============================================================================

/// Improves the computed solution to a general system and provides error bounds.
///
/// Uses iterative refinement to improve accuracy of a solution from LU factorization.
///
/// # Safety
/// - `a` must point to a valid n x n original matrix A
/// - `af` must point to a valid n x n LU factorization (from oblas_sgetrf)
/// - `ipiv` must point to an array of n pivot indices (from oblas_sgetrf)
/// - `b` must point to a valid n x nrhs right-hand side matrix B
/// - `x` must point to a valid n x nrhs solution matrix (input/output)
/// - `ferr` must point to an array of nrhs forward error bounds
/// - `berr` must point to an array of nrhs backward error bounds
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sgerfs(
    layout: OblasLayout,
    n: c_int,
    nrhs: c_int,
    a: *const f32,
    lda: c_int,
    af: *const f32,
    ldaf: c_int,
    ipiv: *const c_int,
    b: *const f32,
    ldb: c_int,
    x: *mut f32,
    ldx: c_int,
    ferr: *mut f32,
    berr: *mut f32,
) -> c_int {
    use oxiblas_lapack::solve::refine_solution;

    if n <= 0
        || nrhs <= 0
        || a.is_null()
        || af.is_null()
        || ipiv.is_null()
        || b.is_null()
        || x.is_null()
    {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let nrhs = nrhs as usize;
    let lda = lda as usize;
    let ldaf = ldaf as usize;
    let ldb = ldb as usize;
    let ldx = ldx as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert A to Mat
    let mut mat_a: Mat<f32> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    // Convert AF to Mat and reconstruct Lu
    let mut mat_af: Mat<f32> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major {
                i * ldaf + j
            } else {
                j * ldaf + i
            };
            mat_af[(i, j)] = *af.add(idx);
        }
    }

    // Convert pivot indices (1-based to 0-based)
    let ipiv_slice = slice::from_raw_parts(ipiv, n);
    let mut perm = vec![0usize; n];
    for i in 0..n {
        perm[i] = (ipiv_slice[i] - 1) as usize;
    }

    // Reconstruct LU from the packed format
    let lu = match lu::Lu::compute(mat_a.as_ref()) {
        Ok(lu) => lu,
        Err(_) => return OblasReturn::Singular as c_int,
    };

    // Convert B to Mat
    let mut mat_b: Mat<f32> = Mat::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            let idx = if row_major { i * ldb + j } else { j * ldb + i };
            mat_b[(i, j)] = *b.add(idx);
        }
    }

    // Convert X to Mat
    let mut mat_x: Mat<f32> = Mat::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            let idx = if row_major { i * ldx + j } else { j * ldx + i };
            mat_x[(i, j)] = *x.add(idx);
        }
    }

    // Perform iterative refinement
    match refine_solution(mat_a.as_ref(), &lu, mat_b.as_ref(), &mut mat_x) {
        Ok(result) => {
            // Copy refined solution back to x
            for i in 0..n {
                for j in 0..nrhs {
                    let idx = if row_major { i * ldx + j } else { j * ldx + i };
                    *x.add(idx) = result.solution[(i, j)];
                }
            }

            // Copy error bounds
            if !ferr.is_null() {
                let ferr_slice = slice::from_raw_parts_mut(ferr, nrhs);
                for j in 0..nrhs {
                    ferr_slice[j] = result.forward_error[j];
                }
            }
            if !berr.is_null() {
                let berr_slice = slice::from_raw_parts_mut(berr, nrhs);
                for j in 0..nrhs {
                    berr_slice[j] = result.backward_error[j];
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::Singular as c_int,
    }
}

// =============================================================================
// DGERFS - Iterative refinement for general systems (double precision)
// =============================================================================

/// Improves the computed solution to a general system and provides error bounds.
///
/// # Safety
/// Same as oblas_sgerfs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dgerfs(
    layout: OblasLayout,
    n: c_int,
    nrhs: c_int,
    a: *const f64,
    lda: c_int,
    af: *const f64,
    ldaf: c_int,
    ipiv: *const c_int,
    b: *const f64,
    ldb: c_int,
    x: *mut f64,
    ldx: c_int,
    ferr: *mut f64,
    berr: *mut f64,
) -> c_int {
    use oxiblas_lapack::solve::refine_solution;

    if n <= 0
        || nrhs <= 0
        || a.is_null()
        || af.is_null()
        || ipiv.is_null()
        || b.is_null()
        || x.is_null()
    {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let nrhs = nrhs as usize;
    let lda = lda as usize;
    let _ldaf = ldaf as usize;
    let ldb = ldb as usize;
    let ldx = ldx as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert A to Mat
    let mut mat_a: Mat<f64> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    // Reconstruct LU from the original matrix
    let lu = match lu::Lu::compute(mat_a.as_ref()) {
        Ok(lu) => lu,
        Err(_) => return OblasReturn::Singular as c_int,
    };

    // Convert B to Mat
    let mut mat_b: Mat<f64> = Mat::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            let idx = if row_major { i * ldb + j } else { j * ldb + i };
            mat_b[(i, j)] = *b.add(idx);
        }
    }

    // Convert X to Mat
    let mut mat_x: Mat<f64> = Mat::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            let idx = if row_major { i * ldx + j } else { j * ldx + i };
            mat_x[(i, j)] = *x.add(idx);
        }
    }

    // Perform iterative refinement
    match refine_solution(mat_a.as_ref(), &lu, mat_b.as_ref(), &mut mat_x) {
        Ok(result) => {
            // Copy refined solution back to x
            for i in 0..n {
                for j in 0..nrhs {
                    let idx = if row_major { i * ldx + j } else { j * ldx + i };
                    *x.add(idx) = result.solution[(i, j)];
                }
            }

            // Copy error bounds
            if !ferr.is_null() {
                let ferr_slice = slice::from_raw_parts_mut(ferr, nrhs);
                for j in 0..nrhs {
                    ferr_slice[j] = result.forward_error[j];
                }
            }
            if !berr.is_null() {
                let berr_slice = slice::from_raw_parts_mut(berr, nrhs);
                for j in 0..nrhs {
                    berr_slice[j] = result.backward_error[j];
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::Singular as c_int,
    }
}

// =============================================================================
// SPORFS - Iterative refinement for SPD systems (single precision)
// =============================================================================

/// Improves the computed solution to an SPD system and provides error bounds.
///
/// Uses iterative refinement with Cholesky factorization.
///
/// # Safety
/// - `a` must point to a valid n x n original SPD matrix A
/// - `af` must point to a valid n x n Cholesky factorization (from oblas_spotrf)
/// - `b` must point to a valid n x nrhs right-hand side matrix B
/// - `x` must point to a valid n x nrhs solution matrix (input/output)
/// - `ferr` must point to an array of nrhs forward error bounds
/// - `berr` must point to an array of nrhs backward error bounds
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sporfs(
    layout: OblasLayout,
    uplo: u8,
    n: c_int,
    nrhs: c_int,
    a: *const f32,
    lda: c_int,
    af: *const f32,
    ldaf: c_int,
    b: *const f32,
    ldb: c_int,
    x: *mut f32,
    ldx: c_int,
    ferr: *mut f32,
    berr: *mut f32,
) -> c_int {
    use oxiblas_lapack::cholesky::Cholesky;
    use oxiblas_lapack::solve::refine_solution_cholesky;

    let _ = (uplo, ldaf, af);

    if n <= 0 || nrhs <= 0 || a.is_null() || b.is_null() || x.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let nrhs = nrhs as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldx = ldx as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert A to symmetric Mat
    let mut mat_a: Mat<f32> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat_a[(i, j)] = *a.add(idx);
            mat_a[(j, i)] = *a.add(idx);
        }
    }

    // Compute Cholesky factorization
    let chol = match Cholesky::compute(mat_a.as_ref()) {
        Ok(c) => c,
        Err(_) => return OblasReturn::NotPosdef as c_int,
    };

    // Convert B to Mat
    let mut mat_b: Mat<f32> = Mat::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            let idx = if row_major { i * ldb + j } else { j * ldb + i };
            mat_b[(i, j)] = *b.add(idx);
        }
    }

    // Convert X to Mat
    let mut mat_x: Mat<f32> = Mat::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            let idx = if row_major { i * ldx + j } else { j * ldx + i };
            mat_x[(i, j)] = *x.add(idx);
        }
    }

    // Perform iterative refinement
    match refine_solution_cholesky(mat_a.as_ref(), &chol, mat_b.as_ref(), &mut mat_x) {
        Ok(result) => {
            // Copy refined solution back to x
            for i in 0..n {
                for j in 0..nrhs {
                    let idx = if row_major { i * ldx + j } else { j * ldx + i };
                    *x.add(idx) = result.solution[(i, j)];
                }
            }

            // Copy error bounds
            if !ferr.is_null() {
                let ferr_slice = slice::from_raw_parts_mut(ferr, nrhs);
                for j in 0..nrhs {
                    ferr_slice[j] = result.forward_error[j];
                }
            }
            if !berr.is_null() {
                let berr_slice = slice::from_raw_parts_mut(berr, nrhs);
                for j in 0..nrhs {
                    berr_slice[j] = result.backward_error[j];
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::Singular as c_int,
    }
}

// =============================================================================
// DPORFS - Iterative refinement for SPD systems (double precision)
// =============================================================================

/// Improves the computed solution to an SPD system and provides error bounds.
///
/// # Safety
/// Same as oblas_sporfs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dporfs(
    layout: OblasLayout,
    uplo: u8,
    n: c_int,
    nrhs: c_int,
    a: *const f64,
    lda: c_int,
    af: *const f64,
    ldaf: c_int,
    b: *const f64,
    ldb: c_int,
    x: *mut f64,
    ldx: c_int,
    ferr: *mut f64,
    berr: *mut f64,
) -> c_int {
    use oxiblas_lapack::cholesky::Cholesky;
    use oxiblas_lapack::solve::refine_solution_cholesky;

    let _ = (uplo, ldaf, af);

    if n <= 0 || nrhs <= 0 || a.is_null() || b.is_null() || x.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let nrhs = nrhs as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldx = ldx as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert A to symmetric Mat
    let mut mat_a: Mat<f64> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat_a[(i, j)] = *a.add(idx);
            mat_a[(j, i)] = *a.add(idx);
        }
    }

    // Compute Cholesky factorization
    let chol = match Cholesky::compute(mat_a.as_ref()) {
        Ok(c) => c,
        Err(_) => return OblasReturn::NotPosdef as c_int,
    };

    // Convert B to Mat
    let mut mat_b: Mat<f64> = Mat::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            let idx = if row_major { i * ldb + j } else { j * ldb + i };
            mat_b[(i, j)] = *b.add(idx);
        }
    }

    // Convert X to Mat
    let mut mat_x: Mat<f64> = Mat::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            let idx = if row_major { i * ldx + j } else { j * ldx + i };
            mat_x[(i, j)] = *x.add(idx);
        }
    }

    // Perform iterative refinement
    match refine_solution_cholesky(mat_a.as_ref(), &chol, mat_b.as_ref(), &mut mat_x) {
        Ok(result) => {
            // Copy refined solution back to x
            for i in 0..n {
                for j in 0..nrhs {
                    let idx = if row_major { i * ldx + j } else { j * ldx + i };
                    *x.add(idx) = result.solution[(i, j)];
                }
            }

            // Copy error bounds
            if !ferr.is_null() {
                let ferr_slice = slice::from_raw_parts_mut(ferr, nrhs);
                for j in 0..nrhs {
                    ferr_slice[j] = result.forward_error[j];
                }
            }
            if !berr.is_null() {
                let berr_slice = slice::from_raw_parts_mut(berr, nrhs);
                for j in 0..nrhs {
                    berr_slice[j] = result.backward_error[j];
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::Singular as c_int,
    }
}

// =============================================================================
// SSYRFS - Iterative refinement for symmetric systems (single precision)
// =============================================================================

/// Improves the computed solution to a symmetric system and provides error bounds.
///
/// Uses iterative refinement with LDL^T factorization.
///
/// # Safety
/// - `a` must point to a valid n x n original symmetric matrix A
/// - `af` must point to a valid n x n LDL^T factorization (from oblas_ssytrf)
/// - `b` must point to a valid n x nrhs right-hand side matrix B
/// - `x` must point to a valid n x nrhs solution matrix (input/output)
/// - `ferr` must point to an array of nrhs forward error bounds
/// - `berr` must point to an array of nrhs backward error bounds
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_ssyrfs(
    layout: OblasLayout,
    uplo: u8,
    n: c_int,
    nrhs: c_int,
    a: *const f32,
    lda: c_int,
    af: *const f32,
    ldaf: c_int,
    b: *const f32,
    ldb: c_int,
    x: *mut f32,
    ldx: c_int,
    ferr: *mut f32,
    berr: *mut f32,
) -> c_int {
    use oxiblas_lapack::cholesky::Ldlt;
    use oxiblas_lapack::solve::refine_solution_symmetric;

    let _ = (uplo, ldaf, af);

    if n <= 0 || nrhs <= 0 || a.is_null() || b.is_null() || x.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let nrhs = nrhs as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldx = ldx as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert A to symmetric Mat
    let mut mat_a: Mat<f32> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat_a[(i, j)] = *a.add(idx);
            mat_a[(j, i)] = *a.add(idx);
        }
    }

    // Compute LDL^T factorization
    let ldlt = match Ldlt::compute(mat_a.as_ref()) {
        Ok(l) => l,
        Err(_) => return OblasReturn::Singular as c_int,
    };

    // Convert B to Mat
    let mut mat_b: Mat<f32> = Mat::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            let idx = if row_major { i * ldb + j } else { j * ldb + i };
            mat_b[(i, j)] = *b.add(idx);
        }
    }

    // Convert X to Mat
    let mut mat_x: Mat<f32> = Mat::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            let idx = if row_major { i * ldx + j } else { j * ldx + i };
            mat_x[(i, j)] = *x.add(idx);
        }
    }

    // Perform iterative refinement
    match refine_solution_symmetric(mat_a.as_ref(), &ldlt, mat_b.as_ref(), &mut mat_x) {
        Ok(result) => {
            // Copy refined solution back to x
            for i in 0..n {
                for j in 0..nrhs {
                    let idx = if row_major { i * ldx + j } else { j * ldx + i };
                    *x.add(idx) = result.solution[(i, j)];
                }
            }

            // Copy error bounds
            if !ferr.is_null() {
                let ferr_slice = slice::from_raw_parts_mut(ferr, nrhs);
                for j in 0..nrhs {
                    ferr_slice[j] = result.forward_error[j];
                }
            }
            if !berr.is_null() {
                let berr_slice = slice::from_raw_parts_mut(berr, nrhs);
                for j in 0..nrhs {
                    berr_slice[j] = result.backward_error[j];
                }
            }

            OblasReturn::Success as c_int
        }
        Err(_) => OblasReturn::Singular as c_int,
    }
}

// =============================================================================
// DSYRFS - Iterative refinement for symmetric systems (double precision)
// =============================================================================

/// Improves the computed solution to a symmetric system and provides error bounds.
///
/// # Safety
/// Same as oblas_ssyrfs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dsyrfs(
    layout: OblasLayout,
    uplo: u8,
    n: c_int,
    nrhs: c_int,
    a: *const f64,
    lda: c_int,
    af: *const f64,
    ldaf: c_int,
    b: *const f64,
    ldb: c_int,
    x: *mut f64,
    ldx: c_int,
    ferr: *mut f64,
    berr: *mut f64,
) -> c_int {
    use oxiblas_lapack::cholesky::Ldlt;
    use oxiblas_lapack::solve::refine_solution_symmetric;

    let _ = (uplo, ldaf, af);

    if n <= 0 || nrhs <= 0 || a.is_null() || b.is_null() || x.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n = n as usize;
    let nrhs = nrhs as usize;
    let lda = lda as usize;
    let ldb = ldb as usize;
    let ldx = ldx as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert A to symmetric Mat
    let mut mat_a: Mat<f64> = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            mat_a[(i, j)] = *a.add(idx);
            mat_a[(j, i)] = *a.add(idx);
        }
    }

    // Compute LDL^T factorization
    let ldlt = match Ldlt::compute(mat_a.as_ref()) {
        Ok(l) => l,
        Err(_) => return OblasReturn::Singular as c_int,
    };

    // Convert B to Mat
    let mut mat_b: Mat<f64> = Mat::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            let idx = if row_major { i * ldb + j } else { j * ldb + i };
            mat_b[(i, j)] = *b.add(idx);
        }
    }

    // Convert X to Mat
    let mut mat_x: Mat<f64> = Mat::zeros(n, nrhs);
    for i in 0..n {
        for j in 0..nrhs {
            let idx = if row_major { i * ldx + j } else { j * ldx + i };
            mat_x[(i, j)] = *x.add(idx);
        }
    }

    // Perform iterative refinement
    match refine_solution_symmetric(mat_a.as_ref(), &ldlt, mat_b.as_ref(), &mut mat_x) {
        Ok(result) => {
            // Copy refined solution back to x
            for i in 0..n {
                for j in 0..nrhs {
                    let idx = if row_major { i * ldx + j } else { j * ldx + i };
                    *x.add(idx) = result.solution[(i, j)];
                }
            }

            // Copy error bounds
            if !ferr.is_null() {
                let ferr_slice = slice::from_raw_parts_mut(ferr, nrhs);
                for j in 0..nrhs {
                    ferr_slice[j] = result.forward_error[j];
                }
            }
            if !berr.is_null() {
                let berr_slice = slice::from_raw_parts_mut(berr, nrhs);
                for j in 0..nrhs {
                    berr_slice[j] = result.backward_error[j];
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
    use crate::lapack::{
        oblas_cgesv, oblas_dgesv, oblas_dgesvx, oblas_dposvx, oblas_dsysvx, oblas_sgesvx,
        oblas_sposvx, oblas_ssysvx, oblas_zgesv,
    };

    #[test]
    fn test_dgerfs() {
        // Test iterative refinement for general systems
        // A = [[4, 2], [2, 5]] (column-major)
        // B = [[8], [11]]
        let a = [4.0f64, 2.0, 2.0, 5.0];
        let af = [4.0f64, 2.0, 2.0, 5.0]; // Dummy AF (we recompute LU)
        let ipiv = [1i32, 2];
        let b = [8.0f64, 11.0];

        // Initial solution (could be from dgesv)
        let mut x = [1.5f64, 1.75]; // Approximate solution
        let mut ferr = [0.0f64];
        let mut berr = [0.0f64];

        unsafe {
            let result = oblas_dgerfs(
                OblasLayout::ColMajor,
                2,
                1,
                a.as_ptr(),
                2,
                af.as_ptr(),
                2,
                ipiv.as_ptr(),
                b.as_ptr(),
                2,
                x.as_mut_ptr(),
                2,
                ferr.as_mut_ptr(),
                berr.as_mut_ptr(),
            );
            assert_eq!(result, 0);

            // Verify Ax = b after refinement
            let ax0 = 4.0 * x[0] + 2.0 * x[1];
            let ax1 = 2.0 * x[0] + 5.0 * x[1];
            assert!((ax0 - 8.0).abs() < 1e-8);
            assert!((ax1 - 11.0).abs() < 1e-8);

            // Error bounds should be computed
            assert!(ferr[0] >= 0.0);
            assert!(berr[0] >= 0.0);
        }
    }

    #[test]
    fn test_sgerfs() {
        // Single precision test
        let a = [4.0f32, 2.0, 2.0, 5.0];
        let af = [4.0f32, 2.0, 2.0, 5.0];
        let ipiv = [1i32, 2];
        let b = [8.0f32, 11.0];
        let mut x = [1.5f32, 1.75];
        let mut ferr = [0.0f32];
        let mut berr = [0.0f32];

        unsafe {
            let result = oblas_sgerfs(
                OblasLayout::ColMajor,
                2,
                1,
                a.as_ptr(),
                2,
                af.as_ptr(),
                2,
                ipiv.as_ptr(),
                b.as_ptr(),
                2,
                x.as_mut_ptr(),
                2,
                ferr.as_mut_ptr(),
                berr.as_mut_ptr(),
            );
            assert_eq!(result, 0);

            // Verify Ax ≈ b
            let ax0 = 4.0 * x[0] + 2.0 * x[1];
            let ax1 = 2.0 * x[0] + 5.0 * x[1];
            assert!((ax0 - 8.0).abs() < 1e-5);
            assert!((ax1 - 11.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dporfs() {
        // Test iterative refinement for SPD systems
        let a = [4.0f64, 2.0, 2.0, 5.0]; // SPD matrix
        let af = [4.0f64, 2.0, 2.0, 5.0]; // Dummy AF
        let b = [8.0f64, 11.0];
        let mut x = [1.5f64, 1.75];
        let mut ferr = [0.0f64];
        let mut berr = [0.0f64];

        unsafe {
            let result = oblas_dporfs(
                OblasLayout::ColMajor,
                b'L',
                2,
                1,
                a.as_ptr(),
                2,
                af.as_ptr(),
                2,
                b.as_ptr(),
                2,
                x.as_mut_ptr(),
                2,
                ferr.as_mut_ptr(),
                berr.as_mut_ptr(),
            );
            assert_eq!(result, 0);

            // Verify Ax = b after refinement
            let ax0 = 4.0 * x[0] + 2.0 * x[1];
            let ax1 = 2.0 * x[0] + 5.0 * x[1];
            assert!((ax0 - 8.0).abs() < 1e-8);
            assert!((ax1 - 11.0).abs() < 1e-8);
        }
    }

    #[test]
    fn test_sporfs() {
        // Single precision test
        let a = [4.0f32, 2.0, 2.0, 5.0];
        let af = [4.0f32, 2.0, 2.0, 5.0];
        let b = [8.0f32, 11.0];
        let mut x = [1.5f32, 1.75];
        let mut ferr = [0.0f32];
        let mut berr = [0.0f32];

        unsafe {
            let result = oblas_sporfs(
                OblasLayout::ColMajor,
                b'L',
                2,
                1,
                a.as_ptr(),
                2,
                af.as_ptr(),
                2,
                b.as_ptr(),
                2,
                x.as_mut_ptr(),
                2,
                ferr.as_mut_ptr(),
                berr.as_mut_ptr(),
            );
            assert_eq!(result, 0);

            // Verify Ax ≈ b
            let ax0 = 4.0 * x[0] + 2.0 * x[1];
            let ax1 = 2.0 * x[0] + 5.0 * x[1];
            assert!((ax0 - 8.0).abs() < 1e-5);
            assert!((ax1 - 11.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dsyrfs() {
        // Test iterative refinement for symmetric systems
        let a = [4.0f64, 2.0, 2.0, 5.0]; // Symmetric matrix
        let af = [4.0f64, 2.0, 2.0, 5.0]; // Dummy AF
        let b = [8.0f64, 11.0];
        let mut x = [1.5f64, 1.75];
        let mut ferr = [0.0f64];
        let mut berr = [0.0f64];

        unsafe {
            let result = oblas_dsyrfs(
                OblasLayout::ColMajor,
                b'L',
                2,
                1,
                a.as_ptr(),
                2,
                af.as_ptr(),
                2,
                b.as_ptr(),
                2,
                x.as_mut_ptr(),
                2,
                ferr.as_mut_ptr(),
                berr.as_mut_ptr(),
            );
            assert_eq!(result, 0);

            // Verify Ax = b after refinement
            let ax0 = 4.0 * x[0] + 2.0 * x[1];
            let ax1 = 2.0 * x[0] + 5.0 * x[1];
            assert!((ax0 - 8.0).abs() < 1e-8);
            assert!((ax1 - 11.0).abs() < 1e-8);
        }
    }

    #[test]
    fn test_ssyrfs() {
        // Single precision test
        let a = [4.0f32, 2.0, 2.0, 5.0];
        let af = [4.0f32, 2.0, 2.0, 5.0];
        let b = [8.0f32, 11.0];
        let mut x = [1.5f32, 1.75];
        let mut ferr = [0.0f32];
        let mut berr = [0.0f32];

        unsafe {
            let result = oblas_ssyrfs(
                OblasLayout::ColMajor,
                b'L',
                2,
                1,
                a.as_ptr(),
                2,
                af.as_ptr(),
                2,
                b.as_ptr(),
                2,
                x.as_mut_ptr(),
                2,
                ferr.as_mut_ptr(),
                berr.as_mut_ptr(),
            );
            assert_eq!(result, 0);

            // Verify Ax ≈ b
            let ax0 = 4.0 * x[0] + 2.0 * x[1];
            let ax1 = 2.0 * x[0] + 5.0 * x[1];
            assert!((ax0 - 8.0).abs() < 1e-5);
            assert!((ax1 - 11.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dgesv() {
        // Solve A*X = B where:
        // A = [[2, 1], [1, 3]] (column-major)
        // B = [[5], [7]] (column-major, solution should be X = [[1.6], [1.8]])
        let mut a = [2.0f64, 1.0, 1.0, 3.0];
        let mut b = [5.0f64, 7.0];
        let mut ipiv = [0i32; 2];

        unsafe {
            let result = oblas_dgesv(
                OblasLayout::ColMajor,
                2,
                1,
                a.as_mut_ptr(),
                2,
                ipiv.as_mut_ptr(),
                b.as_mut_ptr(),
                2,
            );
            assert_eq!(result, 0);

            // Check solution
            assert!((b[0] - 1.6).abs() < 1e-8);
            assert!((b[1] - 1.8).abs() < 1e-8);
        }
    }

    #[test]
    fn test_dgesvx() {
        // Solve A*X = B with expert driver
        // A = [[2, 1], [1, 3]] (column-major)
        // B = [[5], [7]]
        let a = [2.0f64, 1.0, 1.0, 3.0];
        let b = [5.0f64, 7.0];
        let mut x = [0.0f64; 2];
        let mut rcond = 0.0f64;
        let mut ferr = [0.0f64];
        let mut berr = [0.0f64];

        unsafe {
            let result = oblas_dgesvx(
                OblasLayout::ColMajor,
                b'N', // No equilibration
                2,
                1,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                x.as_mut_ptr(),
                2,
                &mut rcond,
                ferr.as_mut_ptr(),
                berr.as_mut_ptr(),
                std::ptr::null_mut(), // No row scaling output
                std::ptr::null_mut(), // No column scaling output
            );
            assert_eq!(result, 0);

            // Check solution
            assert!((x[0] - 1.6).abs() < 1e-8);
            assert!((x[1] - 1.8).abs() < 1e-8);

            // Check rcond is reasonable (should be positive for well-conditioned matrix)
            assert!(rcond > 0.0);
            assert!(rcond <= 1.0);
        }
    }

    #[test]
    fn test_dgesvx_with_equilibration() {
        // Test with row equilibration
        // A = [[1000, 1], [1, 1000]] (column-major) - matrix with row imbalance
        // B = [[1001], [1001]]
        let a = [1000.0f64, 1.0, 1.0, 1000.0];
        let b = [1001.0f64, 1001.0];
        let mut x = [0.0f64; 2];
        let mut rcond = 0.0f64;
        let mut ferr = [0.0f64];
        let mut berr = [0.0f64];
        let mut r = [0.0f64; 2];

        unsafe {
            let result = oblas_dgesvx(
                OblasLayout::ColMajor,
                b'R', // Row equilibration
                2,
                1,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                x.as_mut_ptr(),
                2,
                &mut rcond,
                ferr.as_mut_ptr(),
                berr.as_mut_ptr(),
                r.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            assert_eq!(result, 0);

            // Check solution is approximately [1, 1]
            assert!((x[0] - 1.0).abs() < 1e-6);
            assert!((x[1] - 1.0).abs() < 1e-6);

            // Row scaling should have been applied
            assert!(r[0] > 0.0);
            assert!(r[1] > 0.0);
        }
    }

    #[test]
    fn test_sgesvx() {
        // Single precision test
        let a = [2.0f32, 1.0, 1.0, 3.0];
        let b = [5.0f32, 7.0];
        let mut x = [0.0f32; 2];
        let mut rcond = 0.0f32;
        let mut ferr = [0.0f32];
        let mut berr = [0.0f32];

        unsafe {
            let result = oblas_sgesvx(
                OblasLayout::ColMajor,
                b'N',
                2,
                1,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                x.as_mut_ptr(),
                2,
                &mut rcond,
                ferr.as_mut_ptr(),
                berr.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            assert_eq!(result, 0);

            // Check solution
            assert!((x[0] - 1.6).abs() < 1e-5);
            assert!((x[1] - 1.8).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dgesvx_row_major() {
        // Row-major test
        // A = [[2, 1], [1, 3]] (row-major)
        let a = [2.0f64, 1.0, 1.0, 3.0];
        let b = [5.0f64, 7.0];
        let mut x = [0.0f64; 2];
        let mut rcond = 0.0f64;
        let mut ferr = [0.0f64];
        let mut berr = [0.0f64];

        unsafe {
            let result = oblas_dgesvx(
                OblasLayout::RowMajor,
                b'N',
                2,
                1,
                a.as_ptr(),
                2,
                b.as_ptr(),
                1,
                x.as_mut_ptr(),
                1,
                &mut rcond,
                ferr.as_mut_ptr(),
                berr.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            assert_eq!(result, 0);

            // Check solution
            assert!((x[0] - 1.6).abs() < 1e-8);
            assert!((x[1] - 1.8).abs() < 1e-8);
        }
    }

    #[test]
    fn test_dposvx() {
        // Solve A*X = B with expert Cholesky
        // A = [[4, 2], [2, 5]] (column-major, SPD)
        // B = [[8], [11]]
        let a = [4.0f64, 2.0, 2.0, 5.0];
        let b = [8.0f64, 11.0];
        let mut x = [0.0f64; 2];
        let mut rcond = 0.0f64;
        let mut ferr = [0.0f64];
        let mut berr = [0.0f64];

        unsafe {
            let result = oblas_dposvx(
                OblasLayout::ColMajor,
                b'N', // No equilibration
                2,
                1,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                x.as_mut_ptr(),
                2,
                &mut rcond,
                ferr.as_mut_ptr(),
                berr.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            assert_eq!(result, 0);

            // Verify Ax = b
            let ax0 = 4.0 * x[0] + 2.0 * x[1];
            let ax1 = 2.0 * x[0] + 5.0 * x[1];
            assert!((ax0 - 8.0).abs() < 1e-8);
            assert!((ax1 - 11.0).abs() < 1e-8);

            // Check rcond is reasonable
            assert!(rcond > 0.0);
            assert!(rcond <= 1.0);
        }
    }

    #[test]
    fn test_dposvx_with_equilibration() {
        // A = [[100, 1], [1, 100]] (column-major, SPD with imbalance)
        // B = [[101], [101]]
        let a = [100.0f64, 1.0, 1.0, 100.0];
        let b = [101.0f64, 101.0];
        let mut x = [0.0f64; 2];
        let mut rcond = 0.0f64;
        let mut ferr = [0.0f64];
        let mut berr = [0.0f64];
        let mut s = [0.0f64; 2];

        unsafe {
            let result = oblas_dposvx(
                OblasLayout::ColMajor,
                b'Y', // With equilibration
                2,
                1,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                x.as_mut_ptr(),
                2,
                &mut rcond,
                ferr.as_mut_ptr(),
                berr.as_mut_ptr(),
                s.as_mut_ptr(),
            );
            assert_eq!(result, 0);

            // Verify Ax ≈ b
            let ax0 = 100.0 * x[0] + 1.0 * x[1];
            let ax1 = 1.0 * x[0] + 100.0 * x[1];
            assert!((ax0 - 101.0).abs() < 1e-6);
            assert!((ax1 - 101.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sposvx() {
        // Single precision test
        let a = [4.0f32, 2.0, 2.0, 5.0];
        let b = [8.0f32, 11.0];
        let mut x = [0.0f32; 2];
        let mut rcond = 0.0f32;
        let mut ferr = [0.0f32];
        let mut berr = [0.0f32];

        unsafe {
            let result = oblas_sposvx(
                OblasLayout::ColMajor,
                b'N',
                2,
                1,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                x.as_mut_ptr(),
                2,
                &mut rcond,
                ferr.as_mut_ptr(),
                berr.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            assert_eq!(result, 0);

            // Verify Ax ≈ b
            let ax0 = 4.0 * x[0] + 2.0 * x[1];
            let ax1 = 2.0 * x[0] + 5.0 * x[1];
            assert!((ax0 - 8.0).abs() < 1e-5);
            assert!((ax1 - 11.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dposvx_not_positive_definite() {
        // A = [[1, 2], [2, 1]] is NOT positive definite
        let a = [1.0f64, 2.0, 2.0, 1.0];
        let b = [1.0f64, 2.0];
        let mut x = [0.0f64; 2];
        let mut rcond = 0.0f64;
        let mut ferr = [0.0f64];
        let mut berr = [0.0f64];

        unsafe {
            let result = oblas_dposvx(
                OblasLayout::ColMajor,
                b'N',
                2,
                1,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                x.as_mut_ptr(),
                2,
                &mut rcond,
                ferr.as_mut_ptr(),
                berr.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            // Should fail because matrix is not positive definite
            assert_ne!(result, 0);
        }
    }

    #[test]
    fn test_dsysvx() {
        // Solve A*X = B with expert symmetric driver
        // A = [[4, 2], [2, 5]] (column-major, symmetric)
        // B = [[8], [11]]
        let a = [4.0f64, 2.0, 2.0, 5.0];
        let b = [8.0f64, 11.0];
        let mut x = [0.0f64; 2];
        let mut rcond = 0.0f64;
        let mut ferr = [0.0f64];
        let mut berr = [0.0f64];
        let mut inertia_pos = 0i32;
        let mut inertia_neg = 0i32;

        unsafe {
            let result = oblas_dsysvx(
                OblasLayout::ColMajor,
                b'N',
                2,
                1,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                x.as_mut_ptr(),
                2,
                &mut rcond,
                ferr.as_mut_ptr(),
                berr.as_mut_ptr(),
                &mut inertia_pos,
                &mut inertia_neg,
            );
            assert_eq!(result, 0);

            // Verify Ax = b
            let ax0 = 4.0 * x[0] + 2.0 * x[1];
            let ax1 = 2.0 * x[0] + 5.0 * x[1];
            assert!((ax0 - 8.0).abs() < 1e-8);
            assert!((ax1 - 11.0).abs() < 1e-8);

            // Inertia should be (2, 0) for SPD
            assert_eq!(inertia_pos, 2);
            assert_eq!(inertia_neg, 0);
        }
    }

    #[test]
    fn test_dsysvx_indefinite() {
        // A = [[1, 2], [2, 1]] (column-major, indefinite)
        // B = [[3], [3]]
        let a = [1.0f64, 2.0, 2.0, 1.0];
        let b = [3.0f64, 3.0];
        let mut x = [0.0f64; 2];
        let mut rcond = 0.0f64;
        let mut ferr = [0.0f64];
        let mut berr = [0.0f64];
        let mut inertia_pos = 0i32;
        let mut inertia_neg = 0i32;

        unsafe {
            let result = oblas_dsysvx(
                OblasLayout::ColMajor,
                b'N',
                2,
                1,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                x.as_mut_ptr(),
                2,
                &mut rcond,
                ferr.as_mut_ptr(),
                berr.as_mut_ptr(),
                &mut inertia_pos,
                &mut inertia_neg,
            );
            assert_eq!(result, 0);

            // Verify Ax = b
            let ax0 = 1.0 * x[0] + 2.0 * x[1];
            let ax1 = 2.0 * x[0] + 1.0 * x[1];
            assert!((ax0 - 3.0).abs() < 1e-8);
            assert!((ax1 - 3.0).abs() < 1e-8);

            // Inertia should be (1, 1) for indefinite
            assert_eq!(inertia_pos, 1);
            assert_eq!(inertia_neg, 1);
        }
    }

    #[test]
    fn test_ssysvx() {
        // Single precision test
        let a = [4.0f32, 2.0, 2.0, 5.0];
        let b = [8.0f32, 11.0];
        let mut x = [0.0f32; 2];
        let mut rcond = 0.0f32;
        let mut ferr = [0.0f32];
        let mut berr = [0.0f32];

        unsafe {
            let result = oblas_ssysvx(
                OblasLayout::ColMajor,
                b'N',
                2,
                1,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                x.as_mut_ptr(),
                2,
                &mut rcond,
                ferr.as_mut_ptr(),
                berr.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            assert_eq!(result, 0);

            // Verify Ax ≈ b
            let ax0 = 4.0 * x[0] + 2.0 * x[1];
            let ax1 = 2.0 * x[0] + 5.0 * x[1];
            assert!((ax0 - 8.0).abs() < 1e-5);
            assert!((ax1 - 11.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dsysvx_singular() {
        // Singular matrix: [1 2; 2 4]
        let a = [1.0f64, 2.0, 2.0, 4.0];
        let b = [1.0f64, 2.0];
        let mut x = [0.0f64; 2];
        let mut rcond = 0.0f64;
        let mut ferr = [0.0f64];
        let mut berr = [0.0f64];

        unsafe {
            let result = oblas_dsysvx(
                OblasLayout::ColMajor,
                b'N',
                2,
                1,
                a.as_ptr(),
                2,
                b.as_ptr(),
                2,
                x.as_mut_ptr(),
                2,
                &mut rcond,
                ferr.as_mut_ptr(),
                berr.as_mut_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            // Should fail because matrix is singular
            assert_ne!(result, 0);
        }
    }

    #[test]
    fn test_zgesv() {
        // Complex solve: A*X = B
        // A = [[2+i, 1], [1, 3-i]] (column-major)
        // B = [[3+i], [4]]
        let mut a = [
            OblasComplex64 { re: 2.0, im: 1.0 },
            OblasComplex64 { re: 1.0, im: 0.0 },
            OblasComplex64 { re: 1.0, im: 0.0 },
            OblasComplex64 { re: 3.0, im: -1.0 },
        ];
        let mut b = [
            OblasComplex64 { re: 3.0, im: 1.0 },
            OblasComplex64 { re: 4.0, im: 0.0 },
        ];
        let mut ipiv = [0i32; 2];

        unsafe {
            let result = oblas_zgesv(
                OblasLayout::ColMajor,
                2,
                1,
                a.as_mut_ptr(),
                2,
                ipiv.as_mut_ptr(),
                b.as_mut_ptr(),
                2,
            );
            assert_eq!(result, 0);

            // Verify solution by checking Ax = b (with original A)
            // We can't easily verify since A is overwritten, but the solve succeeded
        }
    }

    #[test]
    fn test_cgesv() {
        // Complex single precision solve
        let mut a = [
            OblasComplex32 { re: 4.0, im: 0.0 },
            OblasComplex32 { re: 1.0, im: -1.0 },
            OblasComplex32 { re: 1.0, im: 1.0 },
            OblasComplex32 { re: 3.0, im: 0.0 },
        ];
        let mut b = [
            OblasComplex32 { re: 5.0, im: 1.0 },
            OblasComplex32 { re: 4.0, im: -1.0 },
        ];
        let mut ipiv = [0i32; 2];

        unsafe {
            let result = oblas_cgesv(
                OblasLayout::ColMajor,
                2,
                1,
                a.as_mut_ptr(),
                2,
                ipiv.as_mut_ptr(),
                b.as_mut_ptr(),
                2,
            );
            assert_eq!(result, 0);
        }
    }
}
