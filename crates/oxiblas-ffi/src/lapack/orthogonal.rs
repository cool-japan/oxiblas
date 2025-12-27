//! LAPACK FFI - Orthogonal/unitary matrix routines (Q operations from QR).

use crate::types::*;
use num_complex::{Complex32, Complex64};
use std::ffi::c_int;
use std::slice;

// =============================================================================
// SORGQR - Generate Q from QR factorization (single precision)
// =============================================================================

/// Generates an m x n orthogonal matrix Q from the QR factorization.
///
/// The QR factorization should have been computed by oblas_sgeqrf.
/// The matrix Q is represented as a product of elementary reflectors:
///   Q = H(1) * H(2) * ... * H(k), where k = min(m, n)
///
/// # Safety
/// - `a` must point to a valid m x n matrix containing the Householder vectors
/// - `tau` must point to an array of k elements containing Householder scalars
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sorgqr(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    k: c_int,
    a: *mut f32,
    lda: c_int,
    tau: *const f32,
) -> c_int {
    if m <= 0 || n <= 0 || k <= 0 || a.is_null() || tau.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }
    if n > m || k > n {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let tau_slice = slice::from_raw_parts(tau, k);

    // Read Householder vectors from A
    let mut v = vec![vec![0.0f32; m]; k];
    for j in 0..k {
        for i in 0..m {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            v[j][i] = if i < j {
                0.0
            } else if i == j {
                1.0
            } else {
                *a.add(idx)
            };
        }
    }

    // Initialize Q to identity
    let mut q = vec![0.0f32; m * n];
    for i in 0..m.min(n) {
        q[i * n + i] = 1.0;
    }

    // Apply Householder reflections in reverse order: Q = H(k) * ... * H(2) * H(1)
    for j in (0..k).rev() {
        let tau_j = tau_slice[j];

        // Apply H(j) = I - tau * v * v^T to Q
        // Q = Q - tau * Q * v * v^T = Q - tau * (Q * v) * v^T

        // Compute w = Q * v[j]
        let mut w = vec![0.0f32; m];
        for i in 0..m {
            let mut sum = 0.0f32;
            for l in j..m {
                sum += q[i * n + l.min(n - 1)] * v[j][l];
            }
            w[i] = sum;
        }

        // Q = Q - tau * w * v^T
        for i in 0..m {
            for l in j..n.min(m) {
                q[i * n + l] -= tau_j * w[i] * v[j][l];
            }
        }
    }

    // Copy Q back to a
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            *a.add(idx) = q[i * n + j];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DORGQR - Generate Q from QR factorization (double precision)
// =============================================================================

/// Generates an m x n orthogonal matrix Q from the QR factorization.
///
/// The QR factorization should have been computed by oblas_dgeqrf.
/// The matrix Q is represented as a product of elementary reflectors:
///   Q = H(1) * H(2) * ... * H(k), where k = min(m, n)
///
/// # Safety
/// - `a` must point to a valid m x n matrix containing the Householder vectors
/// - `tau` must point to an array of k elements containing Householder scalars
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dorgqr(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    k: c_int,
    a: *mut f64,
    lda: c_int,
    tau: *const f64,
) -> c_int {
    if m <= 0 || n <= 0 || k <= 0 || a.is_null() || tau.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }
    if n > m || k > n {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let tau_slice = slice::from_raw_parts(tau, k);

    // Read Householder vectors from A
    let mut v = vec![vec![0.0f64; m]; k];
    for j in 0..k {
        for i in 0..m {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            v[j][i] = if i < j {
                0.0
            } else if i == j {
                1.0
            } else {
                *a.add(idx)
            };
        }
    }

    // Initialize Q to identity
    let mut q = vec![0.0f64; m * n];
    for i in 0..m.min(n) {
        q[i * n + i] = 1.0;
    }

    // Apply Householder reflections in reverse order: Q = H(k) * ... * H(2) * H(1)
    for j in (0..k).rev() {
        let tau_j = tau_slice[j];

        // Apply H(j) = I - tau * v * v^T to Q
        // Q = Q - tau * Q * v * v^T = Q - tau * (Q * v) * v^T

        // Compute w = Q * v[j]
        let mut w = vec![0.0f64; m];
        for i in 0..m {
            let mut sum = 0.0f64;
            for l in j..m {
                sum += q[i * n + l.min(n - 1)] * v[j][l];
            }
            w[i] = sum;
        }

        // Q = Q - tau * w * v^T
        for i in 0..m {
            for l in j..n.min(m) {
                q[i * n + l] -= tau_j * w[i] * v[j][l];
            }
        }
    }

    // Copy Q back to a
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            *a.add(idx) = q[i * n + j];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// SORMQR - Multiply by Q from QR (single precision)
// =============================================================================

/// Multiplies a matrix C by the orthogonal matrix Q from QR factorization.
///
/// Computes one of: Q*C, Q^T*C, C*Q, or C*Q^T
///
/// # Safety
/// - `a` must contain the Householder vectors from sgeqrf
/// - `tau` must contain the Householder scalars
/// - `c` must point to a valid matrix (overwritten with result)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sormqr(
    layout: OblasLayout,
    side: OblasSide,
    trans: OblasTranspose,
    m: c_int,
    n: c_int,
    k: c_int,
    a: *const f32,
    lda: c_int,
    tau: *const f32,
    c: *mut f32,
    ldc: c_int,
) -> c_int {
    if m <= 0 || n <= 0 || k <= 0 || a.is_null() || tau.is_null() || c.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldc = ldc as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let left = side == OblasSide::Left;
    let no_trans = trans == OblasTranspose::NoTrans;
    let tau_slice = slice::from_raw_parts(tau, k);

    // Determine dimensions of Q
    let nq = if left { m } else { n };

    // Read Householder vectors from A
    let mut v = vec![vec![0.0f32; nq]; k];
    for j in 0..k {
        for i in 0..nq {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            v[j][i] = if i < j {
                0.0
            } else if i == j {
                1.0
            } else {
                *a.add(idx)
            };
        }
    }

    // Read C matrix
    let mut c_mat = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            c_mat[i * n + j] = *c.add(idx);
        }
    }

    // Apply Householder reflections
    // For left side: C = Q*C or C = Q^T*C
    // For right side: C = C*Q or C = C*Q^T
    if left {
        // Q*C or Q^T*C
        let range: Box<dyn Iterator<Item = usize>> = if no_trans {
            Box::new((0..k).rev()) // Q = H(k)...H(2)H(1), apply in reverse
        } else {
            Box::new(0..k) // Q^T = H(1)H(2)...H(k), apply in order
        };

        for j in range {
            let tau_j = tau_slice[j];
            if tau_j == 0.0 {
                continue;
            }

            // Apply H(j) = I - tau * v * v^T to C from left
            // C = C - tau * v * (v^T * C)

            // Compute w = v^T * C (row vector of length n)
            let mut w = vec![0.0f32; n];
            for col in 0..n {
                let mut sum = 0.0f32;
                for row in j..m {
                    sum += v[j][row] * c_mat[row * n + col];
                }
                w[col] = sum;
            }

            // C = C - tau * v * w^T
            for row in j..m {
                for col in 0..n {
                    c_mat[row * n + col] -= tau_j * v[j][row] * w[col];
                }
            }
        }
    } else {
        // C*Q or C*Q^T
        let range: Box<dyn Iterator<Item = usize>> = if no_trans {
            Box::new(0..k) // C*Q = C*H(1)H(2)...H(k), apply in order
        } else {
            Box::new((0..k).rev()) // C*Q^T = C*H(k)...H(2)H(1), apply in reverse
        };

        for j in range {
            let tau_j = tau_slice[j];
            if tau_j == 0.0 {
                continue;
            }

            // Apply H(j) from right: C = C - tau * (C * v) * v^T

            // Compute w = C * v (column vector of length m)
            let mut w = vec![0.0f32; m];
            for row in 0..m {
                let mut sum = 0.0f32;
                for col in j..n {
                    sum += c_mat[row * n + col] * v[j][col];
                }
                w[row] = sum;
            }

            // C = C - tau * w * v^T
            for row in 0..m {
                for col in j..n {
                    c_mat[row * n + col] -= tau_j * w[row] * v[j][col];
                }
            }
        }
    }

    // Copy result back to C
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            *c.add(idx) = c_mat[i * n + j];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DORMQR - Multiply by Q from QR (double precision)
// =============================================================================

/// Multiplies a matrix C by the orthogonal matrix Q from QR factorization.
///
/// Computes one of: Q*C, Q^T*C, C*Q, or C*Q^T
///
/// # Safety
/// - `a` must contain the Householder vectors from dgeqrf
/// - `tau` must contain the Householder scalars
/// - `c` must point to a valid matrix (overwritten with result)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dormqr(
    layout: OblasLayout,
    side: OblasSide,
    trans: OblasTranspose,
    m: c_int,
    n: c_int,
    k: c_int,
    a: *const f64,
    lda: c_int,
    tau: *const f64,
    c: *mut f64,
    ldc: c_int,
) -> c_int {
    if m <= 0 || n <= 0 || k <= 0 || a.is_null() || tau.is_null() || c.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldc = ldc as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let left = side == OblasSide::Left;
    let no_trans = trans == OblasTranspose::NoTrans;
    let tau_slice = slice::from_raw_parts(tau, k);

    // Determine dimensions of Q
    let nq = if left { m } else { n };

    // Read Householder vectors from A
    let mut v = vec![vec![0.0f64; nq]; k];
    for j in 0..k {
        for i in 0..nq {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            v[j][i] = if i < j {
                0.0
            } else if i == j {
                1.0
            } else {
                *a.add(idx)
            };
        }
    }

    // Read C matrix
    let mut c_mat = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            c_mat[i * n + j] = *c.add(idx);
        }
    }

    // Apply Householder reflections
    if left {
        // Q*C or Q^T*C
        let range: Box<dyn Iterator<Item = usize>> = if no_trans {
            Box::new((0..k).rev())
        } else {
            Box::new(0..k)
        };

        for j in range {
            let tau_j = tau_slice[j];
            if tau_j == 0.0 {
                continue;
            }

            // Apply H(j) = I - tau * v * v^T to C from left

            // Compute w = v^T * C
            let mut w = vec![0.0f64; n];
            for col in 0..n {
                let mut sum = 0.0f64;
                for row in j..m {
                    sum += v[j][row] * c_mat[row * n + col];
                }
                w[col] = sum;
            }

            // C = C - tau * v * w^T
            for row in j..m {
                for col in 0..n {
                    c_mat[row * n + col] -= tau_j * v[j][row] * w[col];
                }
            }
        }
    } else {
        // C*Q or C*Q^T
        let range: Box<dyn Iterator<Item = usize>> = if no_trans {
            Box::new(0..k)
        } else {
            Box::new((0..k).rev())
        };

        for j in range {
            let tau_j = tau_slice[j];
            if tau_j == 0.0 {
                continue;
            }

            // Apply H(j) from right

            // Compute w = C * v
            let mut w = vec![0.0f64; m];
            for row in 0..m {
                let mut sum = 0.0f64;
                for col in j..n {
                    sum += c_mat[row * n + col] * v[j][col];
                }
                w[row] = sum;
            }

            // C = C - tau * w * v^T
            for row in 0..m {
                for col in j..n {
                    c_mat[row * n + col] -= tau_j * w[row] * v[j][col];
                }
            }
        }
    }

    // Copy result back to C (dormqr)
    for i in 0..m {
        for j in 0..n {
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            *c.add(idx) = c_mat[i * n + j];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// ZUNGQR - Generate unitary Q from QR factorization (double precision complex)
// =============================================================================

/// Generates an m x n unitary matrix Q from the complex QR factorization.
///
/// The QR factorization should have been computed by oblas_zgeqrf.
/// The matrix Q is represented as a product of elementary reflectors:
///   Q = H(1) * H(2) * ... * H(k), where k = min(m, n)
///   and H(i) = I - tau(i) * v(i) * v(i)^H
///
/// # Safety
/// - `a` must point to a valid m x n complex matrix containing the Householder vectors
/// - `tau` must point to an array of k complex elements containing Householder scalars
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zungqr(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    k: c_int,
    a: *mut OblasComplex64,
    lda: c_int,
    tau: *const OblasComplex64,
) -> c_int {
    if m <= 0 || n <= 0 || k <= 0 || a.is_null() || tau.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }
    if n > m || k > n {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Read tau values
    let tau_slice: Vec<Complex64> = (0..k)
        .map(|i| {
            let t = *tau.add(i);
            Complex64::new(t.re, t.im)
        })
        .collect();

    // Read Householder vectors from A
    let mut v = vec![vec![Complex64::new(0.0, 0.0); m]; k];
    for jcol in 0..k {
        for i in 0..m {
            let idx = if row_major {
                i * lda + jcol
            } else {
                jcol * lda + i
            };
            v[jcol][i] = if i < jcol {
                Complex64::new(0.0, 0.0)
            } else if i == jcol {
                Complex64::new(1.0, 0.0)
            } else {
                let c = *a.add(idx);
                Complex64::new(c.re, c.im)
            };
        }
    }

    // Initialize Q to identity
    let mut q = vec![Complex64::new(0.0, 0.0); m * n];
    for i in 0..m.min(n) {
        q[i * n + i] = Complex64::new(1.0, 0.0);
    }

    // Apply Householder reflections in reverse order: Q = H(k) * ... * H(2) * H(1)
    // H(j) = I - tau * v * v^H
    for jcol in (0..k).rev() {
        let tau_j = tau_slice[jcol];

        if tau_j.norm() < 1e-15 {
            continue;
        }

        // Apply H(j) = I - tau * v * v^H to Q
        // Q = Q - tau * Q * v * v^H = Q - tau * (Q * v) * v^H

        // Compute w = Q * v[j]
        let mut w = vec![Complex64::new(0.0, 0.0); m];
        for i in 0..m {
            let mut sum = Complex64::new(0.0, 0.0);
            for l in jcol..m {
                sum += q[i * n + l.min(n - 1)] * v[jcol][l];
            }
            w[i] = sum;
        }

        // Q = Q - tau * w * v^H
        for i in 0..m {
            for l in jcol..n.min(m) {
                q[i * n + l] -= tau_j * w[i] * v[jcol][l].conj();
            }
        }
    }

    // Copy Q back to a
    for i in 0..m {
        for jcol in 0..n {
            let idx = if row_major {
                i * lda + jcol
            } else {
                jcol * lda + i
            };
            *a.add(idx) = OblasComplex64 {
                re: q[i * n + jcol].re,
                im: q[i * n + jcol].im,
            };
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// CUNGQR - Generate unitary Q from QR factorization (single precision complex)
// =============================================================================

/// Generates an m x n unitary matrix Q from the complex QR factorization.
///
/// Single precision complex version.
///
/// # Safety
/// - `a` must point to a valid m x n complex matrix containing the Householder vectors
/// - `tau` must point to an array of k complex elements containing Householder scalars
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_cungqr(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    k: c_int,
    a: *mut OblasComplex32,
    lda: c_int,
    tau: *const OblasComplex32,
) -> c_int {
    if m <= 0 || n <= 0 || k <= 0 || a.is_null() || tau.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }
    if n > m || k > n {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Read tau values
    let tau_slice: Vec<Complex32> = (0..k)
        .map(|i| {
            let t = *tau.add(i);
            Complex32::new(t.re, t.im)
        })
        .collect();

    // Read Householder vectors from A
    let mut v = vec![vec![Complex32::new(0.0, 0.0); m]; k];
    for jcol in 0..k {
        for i in 0..m {
            let idx = if row_major {
                i * lda + jcol
            } else {
                jcol * lda + i
            };
            v[jcol][i] = if i < jcol {
                Complex32::new(0.0, 0.0)
            } else if i == jcol {
                Complex32::new(1.0, 0.0)
            } else {
                let c = *a.add(idx);
                Complex32::new(c.re, c.im)
            };
        }
    }

    // Initialize Q to identity
    let mut q = vec![Complex32::new(0.0, 0.0); m * n];
    for i in 0..m.min(n) {
        q[i * n + i] = Complex32::new(1.0, 0.0);
    }

    // Apply Householder reflections in reverse order
    for jcol in (0..k).rev() {
        let tau_j = tau_slice[jcol];

        if tau_j.norm() < 1e-7 {
            continue;
        }

        // Compute w = Q * v[j]
        let mut w = vec![Complex32::new(0.0, 0.0); m];
        for i in 0..m {
            let mut sum = Complex32::new(0.0, 0.0);
            for l in jcol..m {
                sum += q[i * n + l.min(n - 1)] * v[jcol][l];
            }
            w[i] = sum;
        }

        // Q = Q - tau * w * v^H
        for i in 0..m {
            for l in jcol..n.min(m) {
                q[i * n + l] -= tau_j * w[i] * v[jcol][l].conj();
            }
        }
    }

    // Copy Q back to a
    for i in 0..m {
        for jcol in 0..n {
            let idx = if row_major {
                i * lda + jcol
            } else {
                jcol * lda + i
            };
            *a.add(idx) = OblasComplex32 {
                re: q[i * n + jcol].re,
                im: q[i * n + jcol].im,
            };
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// ZUNMQR - Multiply by unitary Q from QR (double precision complex)
// =============================================================================

/// Multiplies a complex matrix C by the unitary matrix Q from QR factorization.
///
/// Computes one of: Q*C, Q^H*C, C*Q, or C*Q^H
///
/// For complex matrices, H(i) = I - tau(i) * v(i) * v(i)^H
///
/// # Safety
/// - `a` must contain the Householder vectors from zgeqrf
/// - `tau` must contain the Householder scalars
/// - `c` must point to a valid matrix (overwritten with result)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_zunmqr(
    layout: OblasLayout,
    side: OblasSide,
    trans: OblasTranspose,
    m: c_int,
    n: c_int,
    k: c_int,
    a: *const OblasComplex64,
    lda: c_int,
    tau: *const OblasComplex64,
    c: *mut OblasComplex64,
    ldc: c_int,
) -> c_int {
    if m <= 0 || n <= 0 || k <= 0 || a.is_null() || tau.is_null() || c.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldc = ldc as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let left = side == OblasSide::Left;
    // For complex, NoTrans means Q, ConjTrans means Q^H
    let no_trans = trans == OblasTranspose::NoTrans;

    // Read tau values
    let tau_slice: Vec<Complex64> = (0..k)
        .map(|i| {
            let t = *tau.add(i);
            Complex64::new(t.re, t.im)
        })
        .collect();

    // Determine dimensions of Q
    let nq = if left { m } else { n };

    // Read Householder vectors from A
    let mut v = vec![vec![Complex64::new(0.0, 0.0); nq]; k];
    for jcol in 0..k {
        for i in 0..nq {
            let idx = if row_major {
                i * lda + jcol
            } else {
                jcol * lda + i
            };
            v[jcol][i] = if i < jcol {
                Complex64::new(0.0, 0.0)
            } else if i == jcol {
                Complex64::new(1.0, 0.0)
            } else {
                let cplx = *a.add(idx);
                Complex64::new(cplx.re, cplx.im)
            };
        }
    }

    // Read C matrix
    let mut c_mat = vec![Complex64::new(0.0, 0.0); m * n];
    for i in 0..m {
        for jcol in 0..n {
            let idx = if row_major {
                i * ldc + jcol
            } else {
                jcol * ldc + i
            };
            let cplx = *c.add(idx);
            c_mat[i * n + jcol] = Complex64::new(cplx.re, cplx.im);
        }
    }

    // Apply Householder reflections
    // For complex: H = I - tau * v * v^H, H^H = I - conj(tau) * v * v^H
    if left {
        // Q*C or Q^H*C
        let range: Box<dyn Iterator<Item = usize>> = if no_trans {
            Box::new((0..k).rev()) // Q = H(k)...H(2)H(1), apply in reverse
        } else {
            Box::new(0..k) // Q^H = H(1)^H H(2)^H...H(k)^H, apply in order
        };

        for jcol in range {
            let tau_j = if no_trans {
                tau_slice[jcol]
            } else {
                tau_slice[jcol].conj() // For Q^H, use conjugate of tau
            };

            if tau_j.norm() < 1e-15 {
                continue;
            }

            // Apply H(j) = I - tau * v * v^H to C from left
            // C = C - tau * v * (v^H * C)

            // Compute w = v^H * C (row vector of length n)
            let mut w = vec![Complex64::new(0.0, 0.0); n];
            for col in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for row in jcol..m {
                    sum += v[jcol][row].conj() * c_mat[row * n + col];
                }
                w[col] = sum;
            }

            // C = C - tau * v * w^T (w is already v^H * C)
            for row in jcol..m {
                for col in 0..n {
                    c_mat[row * n + col] -= tau_j * v[jcol][row] * w[col];
                }
            }
        }
    } else {
        // C*Q or C*Q^H
        let range: Box<dyn Iterator<Item = usize>> = if no_trans {
            Box::new(0..k) // C*Q = C*H(1)H(2)...H(k), apply in order
        } else {
            Box::new((0..k).rev()) // C*Q^H = C*H(k)^H...H(2)^H H(1)^H, apply in reverse
        };

        for jcol in range {
            let tau_j = if no_trans {
                tau_slice[jcol]
            } else {
                tau_slice[jcol].conj()
            };

            if tau_j.norm() < 1e-15 {
                continue;
            }

            // Apply H(j) from right: C = C - tau * (C * v) * v^H

            // Compute w = C * v (column vector of length m)
            let mut w = vec![Complex64::new(0.0, 0.0); m];
            for row in 0..m {
                let mut sum = Complex64::new(0.0, 0.0);
                for col in jcol..n {
                    sum += c_mat[row * n + col] * v[jcol][col];
                }
                w[row] = sum;
            }

            // C = C - tau * w * v^H
            for row in 0..m {
                for col in jcol..n {
                    c_mat[row * n + col] -= tau_j * w[row] * v[jcol][col].conj();
                }
            }
        }
    }

    // Copy result back to C (zunmqr)
    for i in 0..m {
        for jcol in 0..n {
            let idx = if row_major {
                i * ldc + jcol
            } else {
                jcol * ldc + i
            };
            *c.add(idx) = OblasComplex64 {
                re: c_mat[i * n + jcol].re,
                im: c_mat[i * n + jcol].im,
            };
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// CUNMQR - Multiply by unitary Q from QR (single precision complex)
// =============================================================================

/// Multiplies a complex matrix C by the unitary matrix Q from QR factorization.
///
/// Single precision complex version.
///
/// # Safety
/// - `a` must contain the Householder vectors from cgeqrf
/// - `tau` must contain the Householder scalars
/// - `c` must point to a valid matrix (overwritten with result)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_cunmqr(
    layout: OblasLayout,
    side: OblasSide,
    trans: OblasTranspose,
    m: c_int,
    n: c_int,
    k: c_int,
    a: *const OblasComplex32,
    lda: c_int,
    tau: *const OblasComplex32,
    c: *mut OblasComplex32,
    ldc: c_int,
) -> c_int {
    if m <= 0 || n <= 0 || k <= 0 || a.is_null() || tau.is_null() || c.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m = m as usize;
    let n = n as usize;
    let k = k as usize;
    let lda = lda as usize;
    let ldc = ldc as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let left = side == OblasSide::Left;
    let no_trans = trans == OblasTranspose::NoTrans;

    // Read tau values
    let tau_slice: Vec<Complex32> = (0..k)
        .map(|i| {
            let t = *tau.add(i);
            Complex32::new(t.re, t.im)
        })
        .collect();

    // Determine dimensions of Q
    let nq = if left { m } else { n };

    // Read Householder vectors from A
    let mut v = vec![vec![Complex32::new(0.0, 0.0); nq]; k];
    for jcol in 0..k {
        for i in 0..nq {
            let idx = if row_major {
                i * lda + jcol
            } else {
                jcol * lda + i
            };
            v[jcol][i] = if i < jcol {
                Complex32::new(0.0, 0.0)
            } else if i == jcol {
                Complex32::new(1.0, 0.0)
            } else {
                let cplx = *a.add(idx);
                Complex32::new(cplx.re, cplx.im)
            };
        }
    }

    // Read C matrix
    let mut c_mat = vec![Complex32::new(0.0, 0.0); m * n];
    for i in 0..m {
        for jcol in 0..n {
            let idx = if row_major {
                i * ldc + jcol
            } else {
                jcol * ldc + i
            };
            let cplx = *c.add(idx);
            c_mat[i * n + jcol] = Complex32::new(cplx.re, cplx.im);
        }
    }

    // Apply Householder reflections
    if left {
        let range: Box<dyn Iterator<Item = usize>> = if no_trans {
            Box::new((0..k).rev())
        } else {
            Box::new(0..k)
        };

        for jcol in range {
            let tau_j = if no_trans {
                tau_slice[jcol]
            } else {
                tau_slice[jcol].conj()
            };

            if tau_j.norm() < 1e-7 {
                continue;
            }

            // Compute w = v^H * C
            let mut w = vec![Complex32::new(0.0, 0.0); n];
            for col in 0..n {
                let mut sum = Complex32::new(0.0, 0.0);
                for row in jcol..m {
                    sum += v[jcol][row].conj() * c_mat[row * n + col];
                }
                w[col] = sum;
            }

            // C = C - tau * v * w
            for row in jcol..m {
                for col in 0..n {
                    c_mat[row * n + col] -= tau_j * v[jcol][row] * w[col];
                }
            }
        }
    } else {
        let range: Box<dyn Iterator<Item = usize>> = if no_trans {
            Box::new(0..k)
        } else {
            Box::new((0..k).rev())
        };

        for jcol in range {
            let tau_j = if no_trans {
                tau_slice[jcol]
            } else {
                tau_slice[jcol].conj()
            };

            if tau_j.norm() < 1e-7 {
                continue;
            }

            // Compute w = C * v
            let mut w = vec![Complex32::new(0.0, 0.0); m];
            for row in 0..m {
                let mut sum = Complex32::new(0.0, 0.0);
                for col in jcol..n {
                    sum += c_mat[row * n + col] * v[jcol][col];
                }
                w[row] = sum;
            }

            // C = C - tau * w * v^H
            for row in 0..m {
                for col in jcol..n {
                    c_mat[row * n + col] -= tau_j * w[row] * v[jcol][col].conj();
                }
            }
        }
    }

    // Copy result back to C (cunmqr)
    for i in 0..m {
        for jcol in 0..n {
            let idx = if row_major {
                i * ldc + jcol
            } else {
                jcol * ldc + i
            };
            *c.add(idx) = OblasComplex32 {
                re: c_mat[i * n + jcol].re,
                im: c_mat[i * n + jcol].im,
            };
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// Bidiagonal Reduction and Transforms (GEBRD, ORGBR, ORMBR)
// =============================================================================

/// Which matrix to work with in bidiagonal transforms.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OblasVect {
    /// Work with Q (left orthogonal matrix from column operations).
    Q = 0,
    /// Work with P (right orthogonal matrix from row operations).
    P = 1,
}

// =============================================================================
// SGEBRD - Bidiagonal reduction (single precision)
// =============================================================================

/// Reduces a general m x n matrix A to bidiagonal form B = Q^T * A * P.
///
/// The orthogonal matrices Q and P are represented as products of Householder
/// reflectors. For m >= n: B has diagonal d and superdiagonal e.
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `d` must point to an array of min(m,n) elements
/// - `e` must point to an array of min(m,n)-1 elements
/// - `tauq` must point to an array of min(m,n) elements
/// - `taup` must point to an array of min(m,n) elements
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sgebrd(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *mut f32,
    lda: c_int,
    d: *mut f32,
    e: *mut f32,
    tauq: *mut f32,
    taup: *mut f32,
) -> c_int {
    use oxiblas_lapack::svd::gebrd;
    use oxiblas_matrix::Mat;

    if m <= 0 || n <= 0 || a.is_null() || d.is_null() || tauq.is_null() || taup.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m_usize = m as usize;
    let n_usize = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Read A matrix
    let mut a_mat = Mat::zeros(m_usize, n_usize);
    for i in 0..m_usize {
        for j in 0..n_usize {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            a_mat[(i, j)] = *a.add(idx);
        }
    }

    // Compute bidiagonal reduction
    let factors = match gebrd(a_mat.as_ref()) {
        Ok(f) => f,
        Err(_) => return OblasReturn::InvalidArg as c_int,
    };

    // Copy results back
    for i in 0..factors.d.len() {
        *d.add(i) = factors.d[i];
    }
    if !e.is_null() {
        for i in 0..factors.e.len() {
            *e.add(i) = factors.e[i];
        }
    }
    for i in 0..factors.tauq.len() {
        *tauq.add(i) = factors.tauq[i];
    }
    for i in 0..factors.taup.len() {
        *taup.add(i) = factors.taup[i];
    }

    // Store the working matrix back to A
    for i in 0..m_usize {
        for j in 0..n_usize {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            *a.add(idx) = factors.work[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DGEBRD - Bidiagonal reduction (double precision)
// =============================================================================

/// Reduces a general m x n matrix A to bidiagonal form B = Q^T * A * P.
///
/// Double precision version.
///
/// # Safety
/// - `a` must point to a valid m x n matrix
/// - `d` must point to an array of min(m,n) elements
/// - `e` must point to an array of min(m,n)-1 elements
/// - `tauq` must point to an array of min(m,n) elements
/// - `taup` must point to an array of min(m,n) elements
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dgebrd(
    layout: OblasLayout,
    m: c_int,
    n: c_int,
    a: *mut f64,
    lda: c_int,
    d: *mut f64,
    e: *mut f64,
    tauq: *mut f64,
    taup: *mut f64,
) -> c_int {
    use oxiblas_lapack::svd::gebrd;
    use oxiblas_matrix::Mat;

    if m <= 0 || n <= 0 || a.is_null() || d.is_null() || tauq.is_null() || taup.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m_usize = m as usize;
    let n_usize = n as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;

    // Read A matrix
    let mut a_mat = Mat::zeros(m_usize, n_usize);
    for i in 0..m_usize {
        for j in 0..n_usize {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            a_mat[(i, j)] = *a.add(idx);
        }
    }

    // Compute bidiagonal reduction
    let factors = match gebrd(a_mat.as_ref()) {
        Ok(f) => f,
        Err(_) => return OblasReturn::InvalidArg as c_int,
    };

    // Copy results back
    for i in 0..factors.d.len() {
        *d.add(i) = factors.d[i];
    }
    if !e.is_null() {
        for i in 0..factors.e.len() {
            *e.add(i) = factors.e[i];
        }
    }
    for i in 0..factors.tauq.len() {
        *tauq.add(i) = factors.tauq[i];
    }
    for i in 0..factors.taup.len() {
        *taup.add(i) = factors.taup[i];
    }

    // Store the working matrix back to A
    for i in 0..m_usize {
        for j in 0..n_usize {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            *a.add(idx) = factors.work[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// SORGBR - Generate Q or P from bidiagonal reduction (single precision)
// =============================================================================

/// Generates one of the orthogonal matrices Q or P determined by SGEBRD.
///
/// If vect = Q: generates m x min(m,n) matrix Q
/// If vect = P: generates min(m,n) x n matrix P
///
/// # Safety
/// - `a` must contain the Householder vectors from sgebrd
/// - `tau` must contain the appropriate tau values (tauq for Q, taup for P)
/// - `a` is overwritten with the generated matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sorgbr(
    layout: OblasLayout,
    vect: OblasVect,
    m: c_int,
    n: c_int,
    k: c_int,
    a: *mut f32,
    lda: c_int,
    tau: *const f32,
) -> c_int {
    use oxiblas_lapack::svd::{BidiagFactors, BidiagVect, orgbr};
    use oxiblas_matrix::Mat;

    if m <= 0 || n <= 0 || k <= 0 || a.is_null() || tau.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m_usize = m as usize;
    let n_usize = n as usize;
    let k_usize = k as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let gen_q = vect == OblasVect::Q;

    // For ORGBR, the input depends on whether we're generating Q or P
    // This is a simplified implementation - full LAPACK behavior would need
    // the original GEBRD output structure

    // Read the matrix with Householder vectors
    let rows_in = if gen_q { m_usize } else { k_usize };
    let cols_in = if gen_q { k_usize } else { n_usize };

    let mut work = Mat::zeros(rows_in, cols_in);
    for i in 0..rows_in {
        for j in 0..cols_in {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            work[(i, j)] = *a.add(idx);
        }
    }

    // Read tau values
    let tau_len = k_usize;
    let tau_slice = slice::from_raw_parts(tau, tau_len);

    // Create factors structure manually (simplified)
    let factors = BidiagFactors {
        work: work.clone(),
        d: vec![0.0f32; k_usize],
        e: vec![0.0f32; if k_usize > 1 { k_usize - 1 } else { 0 }],
        tauq: if gen_q {
            tau_slice.to_vec()
        } else {
            vec![0.0f32; k_usize]
        },
        taup: if gen_q {
            vec![0.0f32; if k_usize > 1 { k_usize - 1 } else { 0 }]
        } else {
            tau_slice.to_vec()
        },
        m: rows_in,
        n: cols_in,
    };

    // Generate the orthogonal matrix
    let bidiag_vect = if gen_q { BidiagVect::Q } else { BidiagVect::P };
    let result = match orgbr(&factors, bidiag_vect) {
        Ok(r) => r,
        Err(_) => return OblasReturn::InvalidArg as c_int,
    };

    // Write result back to a
    let out_rows = result.nrows();
    let out_cols = result.ncols();
    for i in 0..out_rows.min(m_usize) {
        for j in 0..out_cols.min(n_usize) {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            *a.add(idx) = result[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DORGBR - Generate Q or P from bidiagonal reduction (double precision)
// =============================================================================

/// Generates one of the orthogonal matrices Q or P determined by DGEBRD.
///
/// Double precision version.
///
/// # Safety
/// - `a` must contain the Householder vectors from dgebrd
/// - `tau` must contain the appropriate tau values (tauq for Q, taup for P)
/// - `a` is overwritten with the generated matrix
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dorgbr(
    layout: OblasLayout,
    vect: OblasVect,
    m: c_int,
    n: c_int,
    k: c_int,
    a: *mut f64,
    lda: c_int,
    tau: *const f64,
) -> c_int {
    use oxiblas_lapack::svd::{BidiagFactors, BidiagVect, orgbr};
    use oxiblas_matrix::Mat;

    if m <= 0 || n <= 0 || k <= 0 || a.is_null() || tau.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m_usize = m as usize;
    let n_usize = n as usize;
    let k_usize = k as usize;
    let lda = lda as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let gen_q = vect == OblasVect::Q;

    let rows_in = if gen_q { m_usize } else { k_usize };
    let cols_in = if gen_q { k_usize } else { n_usize };

    let mut work = Mat::zeros(rows_in, cols_in);
    for i in 0..rows_in {
        for j in 0..cols_in {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            work[(i, j)] = *a.add(idx);
        }
    }

    let tau_len = k_usize;
    let tau_slice = slice::from_raw_parts(tau, tau_len);

    let factors = BidiagFactors {
        work: work.clone(),
        d: vec![0.0f64; k_usize],
        e: vec![0.0f64; if k_usize > 1 { k_usize - 1 } else { 0 }],
        tauq: if gen_q {
            tau_slice.to_vec()
        } else {
            vec![0.0f64; k_usize]
        },
        taup: if gen_q {
            vec![0.0f64; if k_usize > 1 { k_usize - 1 } else { 0 }]
        } else {
            tau_slice.to_vec()
        },
        m: rows_in,
        n: cols_in,
    };

    let bidiag_vect = if gen_q { BidiagVect::Q } else { BidiagVect::P };
    let result = match orgbr(&factors, bidiag_vect) {
        Ok(r) => r,
        Err(_) => return OblasReturn::InvalidArg as c_int,
    };

    let out_rows = result.nrows();
    let out_cols = result.ncols();
    for i in 0..out_rows.min(m_usize) {
        for j in 0..out_cols.min(n_usize) {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            *a.add(idx) = result[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// SORMBR - Multiply by Q or P from bidiagonal reduction (single precision)
// =============================================================================

/// Multiplies a matrix C by one of the orthogonal matrices Q or P from SGEBRD.
///
/// Computes one of:
/// - side=Left,  trans=N: Q*C   or P*C
/// - side=Left,  trans=T: Q^T*C or P^T*C
/// - side=Right, trans=N: C*Q   or C*P
/// - side=Right, trans=T: C*Q^T or C*P^T
///
/// # Safety
/// - `a` must contain Householder vectors from sgebrd
/// - `tau` must contain tau values (tauq for Q, taup for P)
/// - `c` must point to valid m x n matrix (overwritten with result)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_sormbr(
    layout: OblasLayout,
    vect: OblasVect,
    side: OblasSide,
    trans: OblasTranspose,
    m: c_int,
    n: c_int,
    k: c_int,
    a: *const f32,
    lda: c_int,
    tau: *const f32,
    c: *mut f32,
    ldc: c_int,
) -> c_int {
    use oxiblas_lapack::svd::{BidiagFactors, BidiagVect, Side, Trans, ormbr};
    use oxiblas_matrix::Mat;

    if m <= 0 || n <= 0 || k <= 0 || a.is_null() || tau.is_null() || c.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m_usize = m as usize;
    let n_usize = n as usize;
    let k_usize = k as usize;
    let lda = lda as usize;
    let ldc = ldc as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let apply_q = vect == OblasVect::Q;
    let left = side == OblasSide::Left;
    let no_trans = trans == OblasTranspose::NoTrans;

    // Determine dimensions of A based on vect and side
    let nq = if left { m_usize } else { n_usize };
    let rows_a = if apply_q { nq } else { k_usize.min(nq) };
    let cols_a = if apply_q { k_usize.min(nq) } else { nq };

    // Read A (Householder vectors)
    let mut work = Mat::zeros(rows_a, cols_a);
    for i in 0..rows_a {
        for j in 0..cols_a {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            work[(i, j)] = *a.add(idx);
        }
    }

    // Read tau
    let tau_len = k_usize;
    let tau_slice = slice::from_raw_parts(tau, tau_len);

    // Read C matrix
    let mut c_mat = Mat::zeros(m_usize, n_usize);
    for i in 0..m_usize {
        for j in 0..n_usize {
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            c_mat[(i, j)] = *c.add(idx);
        }
    }

    // Build factors
    let min_dim = rows_a.min(cols_a);
    let factors = BidiagFactors {
        work: work.clone(),
        d: vec![0.0f32; min_dim],
        e: vec![0.0f32; if min_dim > 1 { min_dim - 1 } else { 0 }],
        tauq: if apply_q {
            tau_slice.to_vec()
        } else {
            vec![0.0f32; min_dim]
        },
        taup: if apply_q {
            vec![0.0f32; if min_dim > 1 { min_dim - 1 } else { 0 }]
        } else {
            tau_slice.to_vec()
        },
        m: rows_a,
        n: cols_a,
    };

    let bidiag_vect = if apply_q {
        BidiagVect::Q
    } else {
        BidiagVect::P
    };
    let side_enum = if left { Side::Left } else { Side::Right };
    let trans_enum = if no_trans {
        Trans::NoTrans
    } else {
        Trans::Trans
    };

    let result = match ormbr(&factors, bidiag_vect, side_enum, trans_enum, c_mat.as_ref()) {
        Ok(r) => r,
        Err(_) => return OblasReturn::InvalidArg as c_int,
    };

    // Write result back to C
    for i in 0..m_usize {
        for j in 0..n_usize {
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            *c.add(idx) = result[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// DORMBR - Multiply by Q or P from bidiagonal reduction (double precision)
// =============================================================================

/// Multiplies a matrix C by one of the orthogonal matrices Q or P from DGEBRD.
///
/// Double precision version.
///
/// # Safety
/// - `a` must contain Householder vectors from dgebrd
/// - `tau` must contain tau values (tauq for Q, taup for P)
/// - `c` must point to valid m x n matrix (overwritten with result)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oblas_dormbr(
    layout: OblasLayout,
    vect: OblasVect,
    side: OblasSide,
    trans: OblasTranspose,
    m: c_int,
    n: c_int,
    k: c_int,
    a: *const f64,
    lda: c_int,
    tau: *const f64,
    c: *mut f64,
    ldc: c_int,
) -> c_int {
    use oxiblas_lapack::svd::{BidiagFactors, BidiagVect, Side, Trans, ormbr};
    use oxiblas_matrix::Mat;

    if m <= 0 || n <= 0 || k <= 0 || a.is_null() || tau.is_null() || c.is_null() {
        return OblasReturn::InvalidArg as c_int;
    }

    let m_usize = m as usize;
    let n_usize = n as usize;
    let k_usize = k as usize;
    let lda = lda as usize;
    let ldc = ldc as usize;
    let row_major = layout == OblasLayout::RowMajor;
    let apply_q = vect == OblasVect::Q;
    let left = side == OblasSide::Left;
    let no_trans = trans == OblasTranspose::NoTrans;

    let nq = if left { m_usize } else { n_usize };
    let rows_a = if apply_q { nq } else { k_usize.min(nq) };
    let cols_a = if apply_q { k_usize.min(nq) } else { nq };

    let mut work = Mat::zeros(rows_a, cols_a);
    for i in 0..rows_a {
        for j in 0..cols_a {
            let idx = if row_major { i * lda + j } else { j * lda + i };
            work[(i, j)] = *a.add(idx);
        }
    }

    let tau_len = k_usize;
    let tau_slice = slice::from_raw_parts(tau, tau_len);

    let mut c_mat = Mat::zeros(m_usize, n_usize);
    for i in 0..m_usize {
        for j in 0..n_usize {
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            c_mat[(i, j)] = *c.add(idx);
        }
    }

    let min_dim = rows_a.min(cols_a);
    let factors = BidiagFactors {
        work: work.clone(),
        d: vec![0.0f64; min_dim],
        e: vec![0.0f64; if min_dim > 1 { min_dim - 1 } else { 0 }],
        tauq: if apply_q {
            tau_slice.to_vec()
        } else {
            vec![0.0f64; min_dim]
        },
        taup: if apply_q {
            vec![0.0f64; if min_dim > 1 { min_dim - 1 } else { 0 }]
        } else {
            tau_slice.to_vec()
        },
        m: rows_a,
        n: cols_a,
    };

    let bidiag_vect = if apply_q {
        BidiagVect::Q
    } else {
        BidiagVect::P
    };
    let side_enum = if left { Side::Left } else { Side::Right };
    let trans_enum = if no_trans {
        Trans::NoTrans
    } else {
        Trans::Trans
    };

    let result = match ormbr(&factors, bidiag_vect, side_enum, trans_enum, c_mat.as_ref()) {
        Ok(r) => r,
        Err(_) => return OblasReturn::InvalidArg as c_int,
    };

    for i in 0..m_usize {
        for j in 0..n_usize {
            let idx = if row_major { i * ldc + j } else { j * ldc + i };
            *c.add(idx) = result[(i, j)];
        }
    }

    OblasReturn::Success as c_int
}

// =============================================================================
// Tests for bidiagonal functions
// =============================================================================

#[cfg(test)]
mod tests_bidiag {
    use super::*;

    #[test]
    fn test_dgebrd_basic() {
        // 4x3 matrix
        let mut a: Vec<f64> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let mut d = vec![0.0f64; 3];
        let mut e = vec![0.0f64; 2];
        let mut tauq = vec![0.0f64; 3];
        let mut taup = vec![0.0f64; 2];

        let ret = unsafe {
            oblas_dgebrd(
                OblasLayout::RowMajor,
                4,
                3,
                a.as_mut_ptr(),
                3,
                d.as_mut_ptr(),
                e.as_mut_ptr(),
                tauq.as_mut_ptr(),
                taup.as_mut_ptr(),
            )
        };

        assert_eq!(ret, OblasReturn::Success as c_int);
        // Check that d has non-zero values (diagonal of bidiagonal)
        assert!(d.iter().any(|&x| x.abs() > 1e-10));
    }

    #[test]
    fn test_sgebrd_basic() {
        let mut a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut d = vec![0.0f32; 2];
        let mut e = vec![0.0f32; 1];
        let mut tauq = vec![0.0f32; 2];
        let mut taup = vec![0.0f32; 1];

        let ret = unsafe {
            oblas_sgebrd(
                OblasLayout::RowMajor,
                3,
                2,
                a.as_mut_ptr(),
                2,
                d.as_mut_ptr(),
                e.as_mut_ptr(),
                tauq.as_mut_ptr(),
                taup.as_mut_ptr(),
            )
        };

        assert_eq!(ret, OblasReturn::Success as c_int);
    }
}
