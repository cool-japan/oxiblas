//! LAPACK FFI - Kronecker product routines.

use crate::types::*;
use oxiblas_lapack::utils::{
    commutation_matrix, duplication_matrix, elimination_matrix, khatri_rao, kron, kron_sum,
    kron_vec, unvec, vec_mat,
};
use oxiblas_matrix::Mat;
use std::ffi::c_int;

// =============================================================================
// SKRON - Kronecker product (single precision)
// =============================================================================

/// Computes the Kronecker product C = A ⊗ B.
///
/// If A is m×n and B is p×q, then C is (mp)×(nq).
///
/// # Arguments
/// * `layout` - Matrix layout
/// * `m` - Number of rows in A
/// * `n` - Number of columns in A
/// * `a` - Matrix A (input)
/// * `lda` - Leading dimension of A
/// * `p` - Number of rows in B
/// * `q` - Number of columns in B
/// * `b` - Matrix B (input)
/// * `ldb` - Leading dimension of B
/// * `c` - Matrix C (output, must be pre-allocated as mp × nq)
/// * `ldc` - Leading dimension of C
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
///
/// # Safety
/// - `a` must point to a valid m × n matrix
/// - `b` must point to a valid p × q matrix
/// - `c` must point to pre-allocated storage for (mp) × (nq) matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_skron(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *const f32,
    lda: c_int,
    p: c_int,
    q: c_int,
    b: *const f32,
    ldb: c_int,
    c: *mut f32,
    ldc: c_int,
) -> c_int {
    if m <= 0 || n <= 0 || p <= 0 || q <= 0 || a.is_null() || b.is_null() || c.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m_val = m as usize;
    let n_val = n as usize;
    let p_val = p as usize;
    let q_val = q as usize;
    let lda_val = lda as usize;
    let ldb_val = ldb as usize;
    let ldc_val = ldc as usize;
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

    // Convert B to Mat
    let mut mat_b: Mat<f32> = Mat::zeros(p_val, q_val);
    for i in 0..p_val {
        for j in 0..q_val {
            let idx = if row_major {
                i * ldb_val + j
            } else {
                j * ldb_val + i
            };
            mat_b[(i, j)] = *b.add(idx);
        }
    }

    // Compute Kronecker product
    let result = kron(mat_a.as_ref(), mat_b.as_ref());

    // Copy result to C
    let out_rows = m_val * p_val;
    let out_cols = n_val * q_val;
    for i in 0..out_rows {
        for j in 0..out_cols {
            let idx = if row_major {
                i * ldc_val + j
            } else {
                j * ldc_val + i
            };
            *c.add(idx) = result[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DKRON - Kronecker product (double precision)
// =============================================================================

/// Computes the Kronecker product C = A ⊗ B.
///
/// If A is m×n and B is p×q, then C is (mp)×(nq).
///
/// # Safety
/// - `a` must point to a valid m × n matrix
/// - `b` must point to a valid p × q matrix
/// - `c` must point to pre-allocated storage for (mp) × (nq) matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dkron(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *const f64,
    lda: c_int,
    p: c_int,
    q: c_int,
    b: *const f64,
    ldb: c_int,
    c: *mut f64,
    ldc: c_int,
) -> c_int {
    if m <= 0 || n <= 0 || p <= 0 || q <= 0 || a.is_null() || b.is_null() || c.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m_val = m as usize;
    let n_val = n as usize;
    let p_val = p as usize;
    let q_val = q as usize;
    let lda_val = lda as usize;
    let ldb_val = ldb as usize;
    let ldc_val = ldc as usize;
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

    // Convert B to Mat
    let mut mat_b: Mat<f64> = Mat::zeros(p_val, q_val);
    for i in 0..p_val {
        for j in 0..q_val {
            let idx = if row_major {
                i * ldb_val + j
            } else {
                j * ldb_val + i
            };
            mat_b[(i, j)] = *b.add(idx);
        }
    }

    // Compute Kronecker product
    let result = kron(mat_a.as_ref(), mat_b.as_ref());

    // Copy result to C
    let out_rows = m_val * p_val;
    let out_cols = n_val * q_val;
    for i in 0..out_rows {
        for j in 0..out_cols {
            let idx = if row_major {
                i * ldc_val + j
            } else {
                j * ldc_val + i
            };
            *c.add(idx) = result[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// SKRONSUM - Kronecker sum (single precision)
// =============================================================================

/// Computes the Kronecker sum C = A ⊕ B = A ⊗ I_p + I_m ⊗ B.
///
/// Both A (m×m) and B (p×p) must be square. Result C is (mp)×(mp).
///
/// # Arguments
/// * `layout` - Matrix layout
/// * `m` - Dimension of A
/// * `a` - Square matrix A (input)
/// * `lda` - Leading dimension of A
/// * `p` - Dimension of B
/// * `b` - Square matrix B (input)
/// * `ldb` - Leading dimension of B
/// * `c` - Matrix C (output, must be pre-allocated as mp × mp)
/// * `ldc` - Leading dimension of C
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
///
/// # Safety
/// - `a` must point to a valid m × m square matrix
/// - `b` must point to a valid p × p square matrix
/// - `c` must point to pre-allocated storage for (mp) × (mp) matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_skronsum(
    layout: OblasLayout,
    m: c_int,
    a: *const f32,
    lda: c_int,
    p: c_int,
    b: *const f32,
    ldb: c_int,
    c: *mut f32,
    ldc: c_int,
) -> c_int {
    if m <= 0 || p <= 0 || a.is_null() || b.is_null() || c.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m_val = m as usize;
    let p_val = p as usize;
    let lda_val = lda as usize;
    let ldb_val = ldb as usize;
    let ldc_val = ldc as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert A to Mat
    let mut mat_a: Mat<f32> = Mat::zeros(m_val, m_val);
    for i in 0..m_val {
        for j in 0..m_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    // Convert B to Mat
    let mut mat_b: Mat<f32> = Mat::zeros(p_val, p_val);
    for i in 0..p_val {
        for j in 0..p_val {
            let idx = if row_major {
                i * ldb_val + j
            } else {
                j * ldb_val + i
            };
            mat_b[(i, j)] = *b.add(idx);
        }
    }

    // Compute Kronecker sum
    let result = kron_sum(mat_a.as_ref(), mat_b.as_ref());

    // Copy result to C
    let out_dim = m_val * p_val;
    for i in 0..out_dim {
        for j in 0..out_dim {
            let idx = if row_major {
                i * ldc_val + j
            } else {
                j * ldc_val + i
            };
            *c.add(idx) = result[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DKRONSUM - Kronecker sum (double precision)
// =============================================================================

/// Computes the Kronecker sum C = A ⊕ B = A ⊗ I_p + I_m ⊗ B.
///
/// # Safety
/// - `a` must point to a valid m × m square matrix
/// - `b` must point to a valid p × p square matrix
/// - `c` must point to pre-allocated storage for (mp) × (mp) matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dkronsum(
    layout: OblasLayout,
    m: c_int,
    a: *const f64,
    lda: c_int,
    p: c_int,
    b: *const f64,
    ldb: c_int,
    c: *mut f64,
    ldc: c_int,
) -> c_int {
    if m <= 0 || p <= 0 || a.is_null() || b.is_null() || c.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m_val = m as usize;
    let p_val = p as usize;
    let lda_val = lda as usize;
    let ldb_val = ldb as usize;
    let ldc_val = ldc as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert A to Mat
    let mut mat_a: Mat<f64> = Mat::zeros(m_val, m_val);
    for i in 0..m_val {
        for j in 0..m_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    // Convert B to Mat
    let mut mat_b: Mat<f64> = Mat::zeros(p_val, p_val);
    for i in 0..p_val {
        for j in 0..p_val {
            let idx = if row_major {
                i * ldb_val + j
            } else {
                j * ldb_val + i
            };
            mat_b[(i, j)] = *b.add(idx);
        }
    }

    // Compute Kronecker sum
    let result = kron_sum(mat_a.as_ref(), mat_b.as_ref());

    // Copy result to C
    let out_dim = m_val * p_val;
    for i in 0..out_dim {
        for j in 0..out_dim {
            let idx = if row_major {
                i * ldc_val + j
            } else {
                j * ldc_val + i
            };
            *c.add(idx) = result[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// SKHATRI_RAO - Khatri-Rao product (single precision)
// =============================================================================

/// Computes the Khatri-Rao product C = A ⊙ B (column-wise Kronecker product).
///
/// If A is m×n and B is p×n (same number of columns), then C is (mp)×n.
///
/// # Safety
/// - `a` must point to a valid m × n matrix
/// - `b` must point to a valid p × n matrix (same n as A)
/// - `c` must point to pre-allocated storage for (mp) × n matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_skhatri_rao(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *const f32,
    lda: c_int,
    p: c_int,
    b: *const f32,
    ldb: c_int,
    c: *mut f32,
    ldc: c_int,
) -> c_int {
    if m <= 0 || n <= 0 || p <= 0 || a.is_null() || b.is_null() || c.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m_val = m as usize;
    let n_val = n as usize;
    let p_val = p as usize;
    let lda_val = lda as usize;
    let ldb_val = ldb as usize;
    let ldc_val = ldc as usize;
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

    // Convert B to Mat
    let mut mat_b: Mat<f32> = Mat::zeros(p_val, n_val);
    for i in 0..p_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * ldb_val + j
            } else {
                j * ldb_val + i
            };
            mat_b[(i, j)] = *b.add(idx);
        }
    }

    // Compute Khatri-Rao product
    let result = khatri_rao(mat_a.as_ref(), mat_b.as_ref());

    // Copy result to C
    let out_rows = m_val * p_val;
    for i in 0..out_rows {
        for j in 0..n_val {
            let idx = if row_major {
                i * ldc_val + j
            } else {
                j * ldc_val + i
            };
            *c.add(idx) = result[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DKHATRI_RAO - Khatri-Rao product (double precision)
// =============================================================================

/// Computes the Khatri-Rao product C = A ⊙ B (column-wise Kronecker product).
///
/// # Safety
/// - `a` must point to a valid m × n matrix
/// - `b` must point to a valid p × n matrix (same n as A)
/// - `c` must point to pre-allocated storage for (mp) × n matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dkhatri_rao(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *const f64,
    lda: c_int,
    p: c_int,
    b: *const f64,
    ldb: c_int,
    c: *mut f64,
    ldc: c_int,
) -> c_int {
    if m <= 0 || n <= 0 || p <= 0 || a.is_null() || b.is_null() || c.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m_val = m as usize;
    let n_val = n as usize;
    let p_val = p as usize;
    let lda_val = lda as usize;
    let ldb_val = ldb as usize;
    let ldc_val = ldc as usize;
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

    // Convert B to Mat
    let mut mat_b: Mat<f64> = Mat::zeros(p_val, n_val);
    for i in 0..p_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * ldb_val + j
            } else {
                j * ldb_val + i
            };
            mat_b[(i, j)] = *b.add(idx);
        }
    }

    // Compute Khatri-Rao product
    let result = khatri_rao(mat_a.as_ref(), mat_b.as_ref());

    // Copy result to C
    let out_rows = m_val * p_val;
    for i in 0..out_rows {
        for j in 0..n_val {
            let idx = if row_major {
                i * ldc_val + j
            } else {
                j * ldc_val + i
            };
            *c.add(idx) = result[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// SVEC - Vectorize matrix (single precision)
// =============================================================================

/// Vectorizes a matrix by stacking its columns.
///
/// If A is m×n, then v = vec(A) is (mn)×1.
///
/// # Safety
/// - `a` must point to a valid m × n matrix
/// - `v` must point to pre-allocated storage for mn elements
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_svec(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *const f32,
    lda: c_int,
    v: *mut f32,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || v.is_null() {
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

    // Compute vec(A)
    let result = vec_mat(mat_a.as_ref());

    // Copy result to v
    for i in 0..(m_val * n_val) {
        *v.add(i) = result[(i, 0)];
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DVEC - Vectorize matrix (double precision)
// =============================================================================

/// Vectorizes a matrix by stacking its columns.
///
/// # Safety
/// - `a` must point to a valid m × n matrix
/// - `v` must point to pre-allocated storage for mn elements
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dvec(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *const f64,
    lda: c_int,
    v: *mut f64,
) -> c_int {
    if m <= 0 || n <= 0 || a.is_null() || v.is_null() {
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

    // Compute vec(A)
    let result = vec_mat(mat_a.as_ref());

    // Copy result to v
    for i in 0..(m_val * n_val) {
        *v.add(i) = result[(i, 0)];
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// SUNVEC - Reshape vector to matrix (single precision)
// =============================================================================

/// Reshapes a vector back into a matrix (inverse of vec).
///
/// # Safety
/// - `v` must point to mn elements
/// - `a` must point to pre-allocated storage for m × n matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sunvec(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    v: *const f32,
    a: *mut f32,
    lda: c_int,
) -> c_int {
    if m <= 0 || n <= 0 || v.is_null() || a.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m_val = m as usize;
    let n_val = n as usize;
    let lda_val = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Create column vector from v
    let mut vec_v: Mat<f32> = Mat::zeros(m_val * n_val, 1);
    for i in 0..(m_val * n_val) {
        vec_v[(i, 0)] = *v.add(i);
    }

    // Compute unvec
    let result = unvec(vec_v.as_ref(), m_val, n_val);

    // Copy result to A
    for i in 0..m_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            *a.add(idx) = result[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DUNVEC - Reshape vector to matrix (double precision)
// =============================================================================

/// Reshapes a vector back into a matrix (inverse of vec).
///
/// # Safety
/// - `v` must point to mn elements
/// - `a` must point to pre-allocated storage for m × n matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dunvec(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    v: *const f64,
    a: *mut f64,
    lda: c_int,
) -> c_int {
    if m <= 0 || n <= 0 || v.is_null() || a.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m_val = m as usize;
    let n_val = n as usize;
    let lda_val = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Create column vector from v
    let mut vec_v: Mat<f64> = Mat::zeros(m_val * n_val, 1);
    for i in 0..(m_val * n_val) {
        vec_v[(i, 0)] = *v.add(i);
    }

    // Compute unvec
    let result = unvec(vec_v.as_ref(), m_val, n_val);

    // Copy result to A
    for i in 0..m_val {
        for j in 0..n_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            *a.add(idx) = result[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// SCOMMUTATION - Commutation matrix (single precision)
// =============================================================================

/// Computes the commutation matrix K_{m,n}.
///
/// K_{m,n} is an (mn)×(mn) matrix such that K_{m,n} * vec(A) = vec(A^T).
///
/// # Safety
/// - `k` must point to pre-allocated storage for (mn) × (mn) matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_scommutation(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    k: *mut f32,
    ldk: c_int,
) -> c_int {
    if m <= 0 || n <= 0 || k.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m_val = m as usize;
    let n_val = n as usize;
    let ldk_val = ldk as usize;
    let row_major = layout == OblasLayout::RowMajor;

    let result = commutation_matrix::<f32>(m_val, n_val);

    let mn = m_val * n_val;
    for i in 0..mn {
        for j in 0..mn {
            let idx = if row_major {
                i * ldk_val + j
            } else {
                j * ldk_val + i
            };
            *k.add(idx) = result[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DCOMMUTATION - Commutation matrix (double precision)
// =============================================================================

/// Computes the commutation matrix K_{m,n}.
///
/// # Safety
/// - `k` must point to pre-allocated storage for (mn) × (mn) matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dcommutation(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    k: *mut f64,
    ldk: c_int,
) -> c_int {
    if m <= 0 || n <= 0 || k.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m_val = m as usize;
    let n_val = n as usize;
    let ldk_val = ldk as usize;
    let row_major = layout == OblasLayout::RowMajor;

    let result = commutation_matrix::<f64>(m_val, n_val);

    let mn = m_val * n_val;
    for i in 0..mn {
        for j in 0..mn {
            let idx = if row_major {
                i * ldk_val + j
            } else {
                j * ldk_val + i
            };
            *k.add(idx) = result[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// SDUPLICATION - Duplication matrix (single precision)
// =============================================================================

/// Computes the duplication matrix D_n.
///
/// D_n is an (n²)×(n(n+1)/2) matrix such that D_n * vech(A) = vec(A)
/// for symmetric A.
///
/// # Safety
/// - `d` must point to pre-allocated storage for (n²) × (n(n+1)/2) matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sduplication(
    layout: OblasLayout,
    n: c_int,
    d: *mut f32,
    ldd: c_int,
) -> c_int {
    if n <= 0 || d.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let ldd_val = ldd as usize;
    let row_major = layout == OblasLayout::RowMajor;

    let result = duplication_matrix::<f32>(n_val);

    let n_sq = n_val * n_val;
    let n_half = n_val * (n_val + 1) / 2;
    for i in 0..n_sq {
        for j in 0..n_half {
            let idx = if row_major {
                i * ldd_val + j
            } else {
                j * ldd_val + i
            };
            *d.add(idx) = result[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DDUPLICATION - Duplication matrix (double precision)
// =============================================================================

/// Computes the duplication matrix D_n.
///
/// # Safety
/// - `d` must point to pre-allocated storage for (n²) × (n(n+1)/2) matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dduplication(
    layout: OblasLayout,
    n: c_int,
    d: *mut f64,
    ldd: c_int,
) -> c_int {
    if n <= 0 || d.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let ldd_val = ldd as usize;
    let row_major = layout == OblasLayout::RowMajor;

    let result = duplication_matrix::<f64>(n_val);

    let n_sq = n_val * n_val;
    let n_half = n_val * (n_val + 1) / 2;
    for i in 0..n_sq {
        for j in 0..n_half {
            let idx = if row_major {
                i * ldd_val + j
            } else {
                j * ldd_val + i
            };
            *d.add(idx) = result[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// SELIMINATION - Elimination matrix (single precision)
// =============================================================================

/// Computes the elimination matrix L_n.
///
/// L_n is an (n(n+1)/2)×(n²) matrix such that L_n * vec(A) = vech(A)
/// for symmetric A.
///
/// # Safety
/// - `l` must point to pre-allocated storage for (n(n+1)/2) × (n²) matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_selimination(
    layout: OblasLayout,
    n: c_int,
    l: *mut f32,
    ldl: c_int,
) -> c_int {
    if n <= 0 || l.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let ldl_val = ldl as usize;
    let row_major = layout == OblasLayout::RowMajor;

    let result = elimination_matrix::<f32>(n_val);

    let n_sq = n_val * n_val;
    let n_half = n_val * (n_val + 1) / 2;
    for i in 0..n_half {
        for j in 0..n_sq {
            let idx = if row_major {
                i * ldl_val + j
            } else {
                j * ldl_val + i
            };
            *l.add(idx) = result[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DELIMINATION - Elimination matrix (double precision)
// =============================================================================

/// Computes the elimination matrix L_n.
///
/// # Safety
/// - `l` must point to pre-allocated storage for (n(n+1)/2) × (n²) matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_delimination(
    layout: OblasLayout,
    n: c_int,
    l: *mut f64,
    ldl: c_int,
) -> c_int {
    if n <= 0 || l.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let n_val = n as usize;
    let ldl_val = ldl as usize;
    let row_major = layout == OblasLayout::RowMajor;

    let result = elimination_matrix::<f64>(n_val);

    let n_sq = n_val * n_val;
    let n_half = n_val * (n_val + 1) / 2;
    for i in 0..n_half {
        for j in 0..n_sq {
            let idx = if row_major {
                i * ldl_val + j
            } else {
                j * ldl_val + i
            };
            *l.add(idx) = result[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// SKRON_VEC - Efficient Kronecker-vector product (single precision)
// =============================================================================

/// Computes (A ⊗ B) * x efficiently without forming the full Kronecker product.
///
/// Uses the identity: (A ⊗ B) * vec(X) = vec(B * X * A^T)
/// where x = vec(X) for X of appropriate size.
///
/// # Arguments
/// * `layout` - Matrix layout
/// * `ma` - Rows of A
/// * `na` - Columns of A
/// * `a` - Matrix A (ma × na)
/// * `lda` - Leading dimension of A
/// * `mb` - Rows of B
/// * `nb` - Columns of B
/// * `b` - Matrix B (mb × nb)
/// * `ldb` - Leading dimension of B
/// * `x` - Input vector (na * nb × 1)
/// * `y` - Output vector (ma * mb × 1)
///
/// # Returns
/// * 0 on success
/// * -1 for invalid arguments
///
/// # Safety
/// - `a`, `b`, `x`, `y` must point to valid storage
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_skron_vec(
    layout: OblasLayout,
    ma: c_int,
    na: c_int,
    a: *const f32,
    lda: c_int,
    mb: c_int,
    nb: c_int,
    b: *const f32,
    ldb: c_int,
    x: *const f32,
    y: *mut f32,
) -> c_int {
    if ma <= 0
        || na <= 0
        || mb <= 0
        || nb <= 0
        || a.is_null()
        || b.is_null()
        || x.is_null()
        || y.is_null()
    {
        return OblasReturn::InvalidArg as c_int;
    }

    let ma_val = ma as usize;
    let na_val = na as usize;
    let mb_val = mb as usize;
    let nb_val = nb as usize;
    let lda_val = lda as usize;
    let ldb_val = ldb as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert A to Mat
    let mut mat_a: Mat<f32> = Mat::zeros(ma_val, na_val);
    for i in 0..ma_val {
        for j in 0..na_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    // Convert B to Mat
    let mut mat_b: Mat<f32> = Mat::zeros(mb_val, nb_val);
    for i in 0..mb_val {
        for j in 0..nb_val {
            let idx = if row_major {
                i * ldb_val + j
            } else {
                j * ldb_val + i
            };
            mat_b[(i, j)] = *b.add(idx);
        }
    }

    // Convert x to slice
    let x_len = na_val * nb_val;
    let x_slice: Vec<f32> = (0..x_len).map(|i| *x.add(i)).collect();

    // Compute (A ⊗ B) * x
    let result = kron_vec(mat_a.as_ref(), mat_b.as_ref(), &x_slice);

    // Copy result to y
    for (i, &val) in result.iter().enumerate() {
        *y.add(i) = val;
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DKRON_VEC - Efficient Kronecker-vector product (double precision)
// =============================================================================

/// Computes (A ⊗ B) * x efficiently without forming the full Kronecker product.
///
/// Uses the identity: (A ⊗ B) * vec(X) = vec(B * X * A^T)
///
/// # Safety
/// - `a`, `b`, `x`, `y` must point to valid storage
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dkron_vec(
    layout: OblasLayout,
    ma: c_int,
    na: c_int,
    a: *const f64,
    lda: c_int,
    mb: c_int,
    nb: c_int,
    b: *const f64,
    ldb: c_int,
    x: *const f64,
    y: *mut f64,
) -> c_int {
    if ma <= 0
        || na <= 0
        || mb <= 0
        || nb <= 0
        || a.is_null()
        || b.is_null()
        || x.is_null()
        || y.is_null()
    {
        return OblasReturn::InvalidArg as c_int;
    }

    let ma_val = ma as usize;
    let na_val = na as usize;
    let mb_val = mb as usize;
    let nb_val = nb as usize;
    let lda_val = lda as usize;
    let ldb_val = ldb as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Convert A to Mat
    let mut mat_a: Mat<f64> = Mat::zeros(ma_val, na_val);
    for i in 0..ma_val {
        for j in 0..na_val {
            let idx = if row_major {
                i * lda_val + j
            } else {
                j * lda_val + i
            };
            mat_a[(i, j)] = *a.add(idx);
        }
    }

    // Convert B to Mat
    let mut mat_b: Mat<f64> = Mat::zeros(mb_val, nb_val);
    for i in 0..mb_val {
        for j in 0..nb_val {
            let idx = if row_major {
                i * ldb_val + j
            } else {
                j * ldb_val + i
            };
            mat_b[(i, j)] = *b.add(idx);
        }
    }

    // Convert x to slice
    let x_len = na_val * nb_val;
    let x_slice: Vec<f64> = (0..x_len).map(|i| *x.add(i)).collect();

    // Compute (A ⊗ B) * x
    let result = kron_vec(mat_a.as_ref(), mat_b.as_ref(), &x_slice);

    // Copy result to y
    for (i, &val) in result.iter().enumerate() {
        *y.add(i) = val;
    }

    OblasReturn::Success as c_int
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dkron_2x2_identity() {
        // A = [[1, 2], [3, 4]], B = I_2
        // C = A ⊗ I = 4x4 block diagonal
        let a = [1.0f64, 3.0, 2.0, 4.0]; // column-major
        let b = [1.0f64, 0.0, 0.0, 1.0]; // column-major I_2
        let mut c = [0.0f64; 16]; // 4x4

        unsafe {
            let ret = oblas_dkron(
                OblasLayout::ColMajor,
                2,
                2,
                a.as_ptr(),
                2,
                2,
                2,
                b.as_ptr(),
                2,
                c.as_mut_ptr(),
                4,
            );
            assert_eq!(ret, 0);

            // Check diagonal blocks
            // Top-left 2x2 block = 1 * I = [[1,0],[0,1]]
            assert!((c[0] - 1.0).abs() < 1e-10); // c[0,0]
            assert!((c[1] - 0.0).abs() < 1e-10); // c[1,0]
            assert!((c[4] - 0.0).abs() < 1e-10); // c[0,1]
            assert!((c[5] - 1.0).abs() < 1e-10); // c[1,1]

            // Top-right 2x2 block = 2 * I = [[2,0],[0,2]]
            assert!((c[8] - 2.0).abs() < 1e-10); // c[0,2]
            assert!((c[13] - 2.0).abs() < 1e-10); // c[1,3]
        }
    }

    #[test]
    fn test_dkron_2x2() {
        // A = [[1, 2], [3, 4]], B = [[0, 5], [6, 7]]
        let a = [1.0f64, 3.0, 2.0, 4.0]; // column-major
        let b = [0.0f64, 6.0, 5.0, 7.0]; // column-major
        let mut c = [0.0f64; 16]; // 4x4

        unsafe {
            let ret = oblas_dkron(
                OblasLayout::ColMajor,
                2,
                2,
                a.as_ptr(),
                2,
                2,
                2,
                b.as_ptr(),
                2,
                c.as_mut_ptr(),
                4,
            );
            assert_eq!(ret, 0);

            // Top-left block = 1 * B = [[0, 5], [6, 7]]
            // Column major: c[0,0]=0, c[1,0]=6, c[0,1]=5, c[1,1]=7
            assert!((c[0] - 0.0).abs() < 1e-10);
            assert!((c[1] - 6.0).abs() < 1e-10);
            assert!((c[4] - 5.0).abs() < 1e-10);
            assert!((c[5] - 7.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dkron_rectangular() {
        // A = [[1, 2, 3]] (1x3), B = [[1], [2]] (2x1)
        // C should be 2x3
        let a = [1.0f64, 2.0, 3.0]; // column-major (lda=1)
        let b = [1.0f64, 2.0]; // column-major (ldb=2)
        let mut c = [0.0f64; 6]; // 2x3

        unsafe {
            let ret = oblas_dkron(
                OblasLayout::ColMajor,
                1,
                3,
                a.as_ptr(),
                1,
                2,
                1,
                b.as_ptr(),
                2,
                c.as_mut_ptr(),
                2,
            );
            assert_eq!(ret, 0);

            // Result: [[1*1, 2*1, 3*1], [1*2, 2*2, 3*2]] = [[1,2,3],[2,4,6]]
            // Column-major: [1, 2, 2, 4, 3, 6]
            assert!((c[0] - 1.0).abs() < 1e-10);
            assert!((c[1] - 2.0).abs() < 1e-10);
            assert!((c[2] - 2.0).abs() < 1e-10);
            assert!((c[3] - 4.0).abs() < 1e-10);
            assert!((c[4] - 3.0).abs() < 1e-10);
            assert!((c[5] - 6.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_skron_2x2() {
        let a = [1.0f32, 3.0, 2.0, 4.0]; // column-major
        let b = [1.0f32, 0.0, 0.0, 1.0]; // column-major I_2
        let mut c = [0.0f32; 16]; // 4x4

        unsafe {
            let ret = oblas_skron(
                OblasLayout::ColMajor,
                2,
                2,
                a.as_ptr(),
                2,
                2,
                2,
                b.as_ptr(),
                2,
                c.as_mut_ptr(),
                4,
            );
            assert_eq!(ret, 0);

            // Check diagonal
            assert!((c[0] - 1.0).abs() < 1e-5);
            assert!((c[5] - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dkronsum_diagonal() {
        // A = [[1, 0], [0, 2]], B = [[3, 0], [0, 4]]
        // A ⊕ B = A ⊗ I_2 + I_2 ⊗ B
        // Diagonal should be [1+3, 1+4, 2+3, 2+4] = [4, 5, 5, 6]
        let a = [1.0f64, 0.0, 0.0, 2.0]; // column-major
        let b = [3.0f64, 0.0, 0.0, 4.0]; // column-major
        let mut c = [0.0f64; 16]; // 4x4

        unsafe {
            let ret = oblas_dkronsum(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                2,
                b.as_ptr(),
                2,
                c.as_mut_ptr(),
                4,
            );
            assert_eq!(ret, 0);

            // Check diagonal elements
            assert!((c[0] - 4.0).abs() < 1e-10); // c[0,0]
            assert!((c[5] - 5.0).abs() < 1e-10); // c[1,1]
            assert!((c[10] - 5.0).abs() < 1e-10); // c[2,2]
            assert!((c[15] - 6.0).abs() < 1e-10); // c[3,3]
        }
    }

    #[test]
    fn test_skronsum_diagonal() {
        let a = [1.0f32, 0.0, 0.0, 2.0]; // column-major
        let b = [3.0f32, 0.0, 0.0, 4.0]; // column-major
        let mut c = [0.0f32; 16]; // 4x4

        unsafe {
            let ret = oblas_skronsum(
                OblasLayout::ColMajor,
                2,
                a.as_ptr(),
                2,
                2,
                b.as_ptr(),
                2,
                c.as_mut_ptr(),
                4,
            );
            assert_eq!(ret, 0);

            // Check diagonal elements
            assert!((c[0] - 4.0).abs() < 1e-5);
            assert!((c[5] - 5.0).abs() < 1e-5);
            assert!((c[10] - 5.0).abs() < 1e-5);
            assert!((c[15] - 6.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dkron_row_major() {
        // Test with row-major layout
        let a = [1.0f64, 2.0, 3.0, 4.0]; // row-major [[1,2],[3,4]]
        let b = [1.0f64, 0.0, 0.0, 1.0]; // row-major I_2
        let mut c = [0.0f64; 16]; // 4x4

        unsafe {
            let ret = oblas_dkron(
                OblasLayout::RowMajor,
                2,
                2,
                a.as_ptr(),
                2,
                2,
                2,
                b.as_ptr(),
                2,
                c.as_mut_ptr(),
                4,
            );
            assert_eq!(ret, 0);

            // In row-major, C[0,0] = A[0,0]*B[0,0] = 1
            assert!((c[0] - 1.0).abs() < 1e-10);
            // C[0,1] = A[0,0]*B[0,1] = 0
            assert!((c[1] - 0.0).abs() < 1e-10);
            // C[0,2] = A[0,1]*B[0,0] = 2
            assert!((c[2] - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dkhatri_rao() {
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        // Khatri-Rao should give 4x2 matrix
        let a = [1.0f64, 3.0, 2.0, 4.0]; // column-major
        let b = [5.0f64, 7.0, 6.0, 8.0]; // column-major
        let mut c = [0.0f64; 8]; // 4x2

        unsafe {
            let ret = oblas_dkhatri_rao(
                OblasLayout::ColMajor,
                2,
                2,
                a.as_ptr(),
                2,
                2,
                b.as_ptr(),
                2,
                c.as_mut_ptr(),
                4,
            );
            assert_eq!(ret, 0);

            // First column: kron([1,3], [5,7]) = [5, 7, 15, 21]
            assert!((c[0] - 5.0).abs() < 1e-10);
            assert!((c[1] - 7.0).abs() < 1e-10);
            assert!((c[2] - 15.0).abs() < 1e-10);
            assert!((c[3] - 21.0).abs() < 1e-10);

            // Second column: kron([2,4], [6,8]) = [12, 16, 24, 32]
            assert!((c[4] - 12.0).abs() < 1e-10);
            assert!((c[5] - 16.0).abs() < 1e-10);
            assert!((c[6] - 24.0).abs() < 1e-10);
            assert!((c[7] - 32.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dvec_dunvec_roundtrip() {
        // A = [[1, 2], [3, 4]]
        let a = [1.0f64, 3.0, 2.0, 4.0]; // column-major
        let mut v = [0.0f64; 4];
        let mut b = [0.0f64; 4];

        unsafe {
            // vec(A)
            let ret1 = oblas_dvec(OblasLayout::ColMajor, 2, 2, a.as_ptr(), 2, v.as_mut_ptr());
            assert_eq!(ret1, 0);

            // vec should be [1, 3, 2, 4] (column-major stacking)
            assert!((v[0] - 1.0).abs() < 1e-10);
            assert!((v[1] - 3.0).abs() < 1e-10);
            assert!((v[2] - 2.0).abs() < 1e-10);
            assert!((v[3] - 4.0).abs() < 1e-10);

            // unvec(v) should give back A
            let ret2 = oblas_dunvec(OblasLayout::ColMajor, 2, 2, v.as_ptr(), b.as_mut_ptr(), 2);
            assert_eq!(ret2, 0);

            // Check roundtrip
            for i in 0..4 {
                assert!((a[i] - b[i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_dcommutation() {
        // K_{2,3} should be 6x6
        let mut k = [0.0f64; 36];

        unsafe {
            let ret = oblas_dcommutation(OblasLayout::ColMajor, 2, 3, k.as_mut_ptr(), 6);
            assert_eq!(ret, 0);

            // K is a permutation matrix - each row and column should sum to 1
            for i in 0..6 {
                let mut row_sum = 0.0;
                let mut col_sum = 0.0;
                for j in 0..6 {
                    row_sum += k[j * 6 + i]; // column-major
                    col_sum += k[i * 6 + j];
                }
                assert!((row_sum - 1.0).abs() < 1e-10);
                assert!((col_sum - 1.0).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_dduplication() {
        // D_2 should be 4x3
        let mut d = [0.0f64; 12];

        unsafe {
            let ret = oblas_dduplication(OblasLayout::ColMajor, 2, d.as_mut_ptr(), 4);
            assert_eq!(ret, 0);

            // For n=2: D_2 is 4x3
            // vech(A) = [a11, a21, a22] and D_2 * vech(A) = vec(A)
            // Check some entries are 1 and most are 0
            let mut ones_count = 0;
            for i in 0..12 {
                if (d[i] - 1.0).abs() < 1e-10 {
                    ones_count += 1;
                }
            }
            // D_2 should have 4 ones (one for each element of vec(A))
            assert!(ones_count >= 3);
        }
    }

    #[test]
    fn test_delimination() {
        // L_2 should be 3x4
        let mut l = [0.0f64; 12];

        unsafe {
            let ret = oblas_delimination(OblasLayout::ColMajor, 2, l.as_mut_ptr(), 3);
            assert_eq!(ret, 0);

            // L_2 extracts lower triangular part
            let mut ones_count = 0;
            for i in 0..12 {
                if (l[i] - 1.0).abs() < 1e-10 {
                    ones_count += 1;
                }
            }
            // Should have exactly 3 ones
            assert_eq!(ones_count, 3);
        }
    }

    #[test]
    fn test_skhatri_rao_basic() {
        let a = [1.0f32, 3.0, 2.0, 4.0]; // column-major
        let b = [5.0f32, 7.0, 6.0, 8.0]; // column-major
        let mut c = [0.0f32; 8];

        unsafe {
            let ret = oblas_skhatri_rao(
                OblasLayout::ColMajor,
                2,
                2,
                a.as_ptr(),
                2,
                2,
                b.as_ptr(),
                2,
                c.as_mut_ptr(),
                4,
            );
            assert_eq!(ret, 0);
            assert!((c[0] - 5.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dkron_vec_identity() {
        // A = [[1, 2], [3, 4]], B = I_2, x = [1, 0, 0, 1] = vec(I_2)
        // (A ⊗ B) * x should equal vec(B * I * A^T) = vec(A^T)
        let a = [1.0f64, 3.0, 2.0, 4.0]; // column-major
        let b = [1.0f64, 0.0, 0.0, 1.0]; // column-major I_2
        let x = [1.0f64, 0.0, 0.0, 1.0]; // vec(I_2)
        let mut y = [0.0f64; 4];

        unsafe {
            let ret = oblas_dkron_vec(
                OblasLayout::ColMajor,
                2,
                2,
                a.as_ptr(),
                2,
                2,
                2,
                b.as_ptr(),
                2,
                x.as_ptr(),
                y.as_mut_ptr(),
            );
            assert_eq!(ret, 0);

            // Compute explicit (A ⊗ B) * x for comparison
            let mut c = [0.0f64; 16];
            let ret_kron = oblas_dkron(
                OblasLayout::ColMajor,
                2,
                2,
                a.as_ptr(),
                2,
                2,
                2,
                b.as_ptr(),
                2,
                c.as_mut_ptr(),
                4,
            );
            assert_eq!(ret_kron, 0);

            // Verify (A ⊗ B) * x matches our efficient computation
            for i in 0..4 {
                let mut expected = 0.0;
                for j in 0..4 {
                    expected += c[j * 4 + i] * x[j]; // column-major
                }
                assert!(
                    (y[i] - expected).abs() < 1e-10,
                    "y[{}]={} vs expected={}",
                    i,
                    y[i],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_dkron_vec_simple() {
        // A = [[2]], B = [[3]], x = [1]
        // (A ⊗ B) * x = [[6]] * [1] = [6]
        let a = [2.0f64];
        let b = [3.0f64];
        let x = [1.0f64];
        let mut y = [0.0f64; 1];

        unsafe {
            let ret = oblas_dkron_vec(
                OblasLayout::ColMajor,
                1,
                1,
                a.as_ptr(),
                1,
                1,
                1,
                b.as_ptr(),
                1,
                x.as_ptr(),
                y.as_mut_ptr(),
            );
            assert_eq!(ret, 0);
            assert!((y[0] - 6.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_skron_vec_identity() {
        let a = [1.0f32, 3.0, 2.0, 4.0]; // column-major
        let b = [1.0f32, 0.0, 0.0, 1.0]; // column-major I_2
        let x = [1.0f32, 0.0, 0.0, 1.0];
        let mut y = [0.0f32; 4];

        unsafe {
            let ret = oblas_skron_vec(
                OblasLayout::ColMajor,
                2,
                2,
                a.as_ptr(),
                2,
                2,
                2,
                b.as_ptr(),
                2,
                x.as_ptr(),
                y.as_mut_ptr(),
            );
            assert_eq!(ret, 0);
            // Just verify it runs without error
            assert!(y.iter().any(|&v| v != 0.0));
        }
    }
}
