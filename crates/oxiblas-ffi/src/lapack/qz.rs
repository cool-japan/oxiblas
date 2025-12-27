//! LAPACK FFI - QZ decomposition (Generalized Schur decomposition).
//!
//! Computes the generalized Schur form of matrix pencil (A, B):
//!   A = Q * S * Z^T
//!   B = Q * T * Z^T
//! where Q and Z are orthogonal, S is quasi-upper triangular, T is upper triangular.

use crate::types::*;
use oxiblas_lapack::evd::Qz;
use oxiblas_matrix::Mat;
use std::ffi::c_int;

// =============================================================================
// SGGES - QZ decomposition (single precision)
// =============================================================================

/// Computes the generalized Schur decomposition of matrix pencil (A, B).
///
/// Returns A = Q * S * Z^T and B = Q * T * Z^T where:
/// - Q, Z are orthogonal matrices
/// - S is quasi-upper triangular (real Schur form)
/// - T is upper triangular
///
/// # Arguments
/// * `layout` - Matrix layout
/// * `n` - Dimension of matrices A and B
/// * `a` - On input: matrix A. On output: S (quasi-upper triangular)
/// * `lda` - Leading dimension of A
/// * `b` - On input: matrix B. On output: T (upper triangular)
/// * `ldb` - Leading dimension of B
/// * `q` - Output: left orthogonal matrix Q
/// * `ldq` - Leading dimension of Q
/// * `z` - Output: right orthogonal matrix Z
/// * `ldz` - Leading dimension of Z
/// * `alphar` - Output: real parts of eigenvalue numerators (length n)
/// * `alphai` - Output: imaginary parts of eigenvalue numerators (length n)
/// * `beta` - Output: eigenvalue denominators (length n)
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
/// * 1 if algorithm did not converge
///
/// # Safety
/// - All matrices must be n × n
/// - All arrays must have length n
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sgges(
    layout: OblasLayout,
    n: c_int,
    a: *mut f32,
    lda: c_int,
    b: *mut f32,
    ldb: c_int,
    q: *mut f32,
    ldq: c_int,
    z: *mut f32,
    ldz: c_int,
    alphar: *mut f32,
    alphai: *mut f32,
    beta: *mut f32,
) -> c_int {
    if n <= 0
        || a.is_null()
        || b.is_null()
        || q.is_null()
        || z.is_null()
        || alphar.is_null()
        || alphai.is_null()
        || beta.is_null()
    {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let lda_val = lda as usize;
    let ldb_val = ldb as usize;
    let ldq_val = ldq as usize;
    let ldz_val = ldz as usize;
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

    // Convert B to Mat
    let mut mat_b: Mat<f32> = Mat::zeros(n_val, n_val);
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * ldb_val + j
            } else {
                j * ldb_val + i
            };
            mat_b[(i, j)] = *b.add(idx);
        }
    }

    // Compute QZ decomposition
    let qz = match Qz::compute(mat_a.as_ref(), mat_b.as_ref()) {
        Ok(q) => q,
        Err(_) => return 1,
    };

    // Copy S (quasi-upper triangular) to A
    let s = qz.s();
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            *a.add(idx) = s[(i, j)];
        }
    }

    // Copy T (upper triangular) to B
    let t = qz.t();
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * ldb_val + j
            } else {
                j * ldb_val + i
            };
            *b.add(idx) = t[(i, j)];
        }
    }

    // Copy Q
    let q_mat = qz.q();
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

    // Copy Z
    let z_mat = qz.z();
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * ldz_val + j
            } else {
                j * ldz_val + i
            };
            *z.add(idx) = z_mat[(i, j)];
        }
    }

    // Copy eigenvalue information
    let eigenvalues = qz.eigenvalues();
    for (i, ev) in eigenvalues.iter().enumerate() {
        *alphar.add(i) = ev.alpha_real;
        *alphai.add(i) = ev.alpha_imag;
        *beta.add(i) = ev.beta;
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DGGES - QZ decomposition (double precision)
// =============================================================================

/// Computes the generalized Schur decomposition (double precision).
///
/// # Safety
/// - All matrices must be n × n
/// - All arrays must have length n
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dgges(
    layout: OblasLayout,
    n: c_int,
    a: *mut f64,
    lda: c_int,
    b: *mut f64,
    ldb: c_int,
    q: *mut f64,
    ldq: c_int,
    z: *mut f64,
    ldz: c_int,
    alphar: *mut f64,
    alphai: *mut f64,
    beta: *mut f64,
) -> c_int {
    if n <= 0
        || a.is_null()
        || b.is_null()
        || q.is_null()
        || z.is_null()
        || alphar.is_null()
        || alphai.is_null()
        || beta.is_null()
    {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let lda_val = lda as usize;
    let ldb_val = ldb as usize;
    let ldq_val = ldq as usize;
    let ldz_val = ldz as usize;
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

    // Convert B to Mat
    let mut mat_b: Mat<f64> = Mat::zeros(n_val, n_val);
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * ldb_val + j
            } else {
                j * ldb_val + i
            };
            mat_b[(i, j)] = *b.add(idx);
        }
    }

    // Compute QZ decomposition
    let qz = match Qz::compute(mat_a.as_ref(), mat_b.as_ref()) {
        Ok(q) => q,
        Err(_) => return 1,
    };

    // Copy S (quasi-upper triangular) to A
    let s = qz.s();
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            *a.add(idx) = s[(i, j)];
        }
    }

    // Copy T (upper triangular) to B
    let t = qz.t();
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * ldb_val + j
            } else {
                j * ldb_val + i
            };
            *b.add(idx) = t[(i, j)];
        }
    }

    // Copy Q
    let q_mat = qz.q();
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

    // Copy Z
    let z_mat = qz.z();
    for i in 0..n_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * ldz_val + j
            } else {
                j * ldz_val + i
            };
            *z.add(idx) = z_mat[(i, j)];
        }
    }

    // Copy eigenvalue information
    let eigenvalues = qz.eigenvalues();
    for (i, ev) in eigenvalues.iter().enumerate() {
        *alphar.add(i) = ev.alpha_real;
        *alphai.add(i) = ev.alpha_imag;
        *beta.add(i) = ev.beta;
    }

    OblasReturn::Success as c_int
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dgges_diagonal() {
        // A = diag(2, 4), B = diag(1, 2) => eigenvalues 2, 2
        let mut a = [2.0f64, 0.0, 0.0, 4.0]; // column-major
        let mut b = [1.0f64, 0.0, 0.0, 2.0]; // column-major
        let mut q = [0.0f64; 4];
        let mut z = [0.0f64; 4];
        let mut alphar = [0.0f64; 2];
        let mut alphai = [0.0f64; 2];
        let mut beta = [0.0f64; 2];

        unsafe {
            let ret = oblas_dgges(
                OblasLayout::ColMajor,
                2,
                a.as_mut_ptr(),
                2,
                b.as_mut_ptr(),
                2,
                q.as_mut_ptr(),
                2,
                z.as_mut_ptr(),
                2,
                alphar.as_mut_ptr(),
                alphai.as_mut_ptr(),
                beta.as_mut_ptr(),
            );
            assert_eq!(ret, 0);

            // Compute eigenvalues = alpha / beta
            let mut eigs = Vec::new();
            for i in 0..2 {
                if beta[i].abs() > 1e-10 {
                    eigs.push(alphar[i] / beta[i]);
                }
            }
            eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // Eigenvalues should be 2 and 2
            for eig in &eigs {
                assert!(
                    (*eig - 2.0).abs() < 1e-6,
                    "Expected eigenvalue 2, got {}",
                    eig
                );
            }
        }
    }

    #[test]
    fn test_dgges_1x1() {
        let mut a = [5.0f64];
        let mut b = [2.0f64];
        let mut q = [0.0f64];
        let mut z = [0.0f64];
        let mut alphar = [0.0f64];
        let mut alphai = [0.0f64];
        let mut beta = [0.0f64];

        unsafe {
            let ret = oblas_dgges(
                OblasLayout::ColMajor,
                1,
                a.as_mut_ptr(),
                1,
                b.as_mut_ptr(),
                1,
                q.as_mut_ptr(),
                1,
                z.as_mut_ptr(),
                1,
                alphar.as_mut_ptr(),
                alphai.as_mut_ptr(),
                beta.as_mut_ptr(),
            );
            assert_eq!(ret, 0);

            // Eigenvalue should be 5/2 = 2.5
            let eig = alphar[0] / beta[0];
            assert!(
                (eig - 2.5).abs() < 1e-10,
                "Expected eigenvalue 2.5, got {}",
                eig
            );
        }
    }

    #[test]
    fn test_sgges_basic() {
        let mut a = [2.0f32, 0.0, 0.0, 3.0]; // column-major
        let mut b = [1.0f32, 0.0, 0.0, 1.0]; // column-major
        let mut q = [0.0f32; 4];
        let mut z = [0.0f32; 4];
        let mut alphar = [0.0f32; 2];
        let mut alphai = [0.0f32; 2];
        let mut beta = [0.0f32; 2];

        unsafe {
            let ret = oblas_sgges(
                OblasLayout::ColMajor,
                2,
                a.as_mut_ptr(),
                2,
                b.as_mut_ptr(),
                2,
                q.as_mut_ptr(),
                2,
                z.as_mut_ptr(),
                2,
                alphar.as_mut_ptr(),
                alphai.as_mut_ptr(),
                beta.as_mut_ptr(),
            );
            assert_eq!(ret, 0);

            // Eigenvalues should be 2 and 3
            let mut eigs = Vec::new();
            for i in 0..2 {
                if beta[i].abs() > 1e-5 {
                    eigs.push(alphar[i] / beta[i]);
                }
            }
            eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());

            assert!((eigs[0] - 2.0).abs() < 1e-4, "Expected 2, got {}", eigs[0]);
            assert!((eigs[1] - 3.0).abs() < 1e-4, "Expected 3, got {}", eigs[1]);
        }
    }

    #[test]
    fn test_dgges_upper_triangular() {
        // A = [[1, 2], [0, 3]], B = I => eigenvalues 1 and 3
        let mut a = [1.0f64, 0.0, 2.0, 3.0]; // column-major
        let mut b = [1.0f64, 0.0, 0.0, 1.0]; // column-major
        let mut q = [0.0f64; 4];
        let mut z = [0.0f64; 4];
        let mut alphar = [0.0f64; 2];
        let mut alphai = [0.0f64; 2];
        let mut beta = [0.0f64; 2];

        unsafe {
            let ret = oblas_dgges(
                OblasLayout::ColMajor,
                2,
                a.as_mut_ptr(),
                2,
                b.as_mut_ptr(),
                2,
                q.as_mut_ptr(),
                2,
                z.as_mut_ptr(),
                2,
                alphar.as_mut_ptr(),
                alphai.as_mut_ptr(),
                beta.as_mut_ptr(),
            );
            assert_eq!(ret, 0);

            // Eigenvalues should be 1 and 3
            let mut eigs = Vec::new();
            for i in 0..2 {
                if beta[i].abs() > 1e-10 {
                    eigs.push(alphar[i] / beta[i]);
                }
            }
            eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());

            assert!((eigs[0] - 1.0).abs() < 1e-6, "Expected 1, got {}", eigs[0]);
            assert!((eigs[1] - 3.0).abs() < 1e-6, "Expected 3, got {}", eigs[1]);
        }
    }

    #[test]
    fn test_dgges_q_orthogonal() {
        // Use diagonal matrices for simpler test case
        let mut a = [3.0f64, 0.0, 0.0, 4.0]; // column-major diagonal
        let mut b = [1.0f64, 0.0, 0.0, 1.0]; // column-major identity
        let mut q = [0.0f64; 4];
        let mut z = [0.0f64; 4];
        let mut alphar = [0.0f64; 2];
        let mut alphai = [0.0f64; 2];
        let mut beta = [0.0f64; 2];

        unsafe {
            let ret = oblas_dgges(
                OblasLayout::ColMajor,
                2,
                a.as_mut_ptr(),
                2,
                b.as_mut_ptr(),
                2,
                q.as_mut_ptr(),
                2,
                z.as_mut_ptr(),
                2,
                alphar.as_mut_ptr(),
                alphai.as_mut_ptr(),
                beta.as_mut_ptr(),
            );
            assert_eq!(ret, 0);

            // Verify Q is orthogonal: Q^T * Q = I
            for i in 0..2 {
                for j in 0..2 {
                    let mut dot = 0.0;
                    for k in 0..2 {
                        // Q[k,i] * Q[k,j] in column-major
                        dot += q[i * 2 + k] * q[j * 2 + k];
                    }
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (dot - expected).abs() < 1e-6,
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
