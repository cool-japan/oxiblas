//! Orthogonal/Unitary transformations for QR factorization.
//!
//! This module provides LAPACK-style routines for working with Q matrices
//! from QR factorization without forming them explicitly.
//!
//! - `orgqr` / `ungqr`: Generate explicit Q matrix from Householder reflectors
//! - `ormqr` / `unmqr`: Multiply a matrix by Q (or Q^T) without forming Q
//!
//! These routines are more efficient than forming Q explicitly when Q is only
//! needed for multiplication.

use crate::qr::Qr;
use oxiblas_core::scalar::{ComplexScalar, Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatMut, MatRef};

/// Side for applying Q transformation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// Apply Q from the left: C = op(Q) * C
    Left,
    /// Apply Q from the right: C = C * op(Q)
    Right,
}

/// Transpose operation for Q.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trans {
    /// No transpose: use Q
    NoTrans,
    /// Transpose: use Q^T (or Q^H for complex)
    Trans,
}

/// Error type for orthogonal transformation operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrthoError {
    /// Dimension mismatch.
    DimensionMismatch,
    /// Invalid parameter.
    InvalidParameter,
    /// Empty matrix.
    EmptyMatrix,
}

impl core::fmt::Display for OrthoError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DimensionMismatch => write!(f, "Dimension mismatch"),
            Self::InvalidParameter => write!(f, "Invalid parameter"),
            Self::EmptyMatrix => write!(f, "Empty matrix"),
        }
    }
}

impl std::error::Error for OrthoError {}

/// Generates the orthogonal matrix Q from QR factorization.
///
/// Given the Householder reflectors stored in the QR factorization,
/// this generates the first `k` columns of the full orthogonal matrix Q.
///
/// This is equivalent to LAPACK's DORGQR/ZUNGQR.
///
/// # Arguments
///
/// * `qr` - The QR factorization containing Householder reflectors
/// * `k` - Number of columns of Q to generate (1 <= k <= min(m, n))
///
/// # Returns
///
/// An m×k matrix containing the first k columns of Q.
///
/// # Example
///
/// ```
/// use oxiblas_lapack::qr::{Qr, orgqr};
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[1.0f64, 2.0],
///     &[3.0, 4.0],
///     &[5.0, 6.0],
/// ]);
///
/// let qr = Qr::compute(a.as_ref()).unwrap();
/// let q = orgqr(&qr, 2).unwrap();
///
/// // Q is 3×2 orthonormal
/// assert_eq!(q.nrows(), 3);
/// assert_eq!(q.ncols(), 2);
/// ```
pub fn orgqr<T: Field + Real + bytemuck::Zeroable>(
    qr: &Qr<T>,
    k: usize,
) -> Result<Mat<T>, OrthoError> {
    let m = qr.nrows();
    let n = qr.ncols();
    let min_mn = m.min(n);

    if k == 0 || k > min_mn {
        return Err(OrthoError::InvalidParameter);
    }

    // Start with identity in the first k columns
    let mut q = Mat::zeros(m, k);
    for i in 0..k {
        q[(i, i)] = T::one();
    }

    // Get internal storage
    let qr_mat = qr.qr_factors();
    let tau = qr.tau();

    // Apply Householder reflections in reverse order
    for j in (0..k).rev() {
        if tau[j] == T::zero() {
            continue;
        }

        // Apply H_j = I - tau * v * v^T
        // v[j] = 1, v[j+1:m] stored in qr[j+1:m, j]
        for col in j..k {
            // Compute w = v^T * q[:, col]
            let mut w = q[(j, col)]; // v[j] = 1
            for i in (j + 1)..m {
                w = w + qr_mat[(i, j)] * q[(i, col)];
            }

            // q[:, col] -= tau * w * v
            let tw = tau[j] * w;
            q[(j, col)] = q[(j, col)] - tw;
            for i in (j + 1)..m {
                q[(i, col)] = q[(i, col)] - tw * qr_mat[(i, j)];
            }
        }
    }

    Ok(q)
}

/// Multiplies a matrix by Q from QR factorization.
///
/// Computes one of:
/// - C = Q * C (side=Left, trans=NoTrans)
/// - C = Q^T * C (side=Left, trans=Trans)
/// - C = C * Q (side=Right, trans=NoTrans)
/// - C = C * Q^T (side=Right, trans=Trans)
///
/// This is equivalent to LAPACK's DORMQR/ZUNMQR.
///
/// # Arguments
///
/// * `qr` - The QR factorization containing Householder reflectors
/// * `side` - Apply Q from left or right
/// * `trans` - Use Q or Q^T
/// * `c` - The matrix to multiply (modified in place)
///
/// # Example
///
/// ```
/// use oxiblas_lapack::qr::{Qr, ormqr, Side, Trans};
/// use oxiblas_matrix::Mat;
///
/// let a = Mat::from_rows(&[
///     &[1.0f64, 2.0],
///     &[3.0, 4.0],
///     &[5.0, 6.0],
/// ]);
///
/// let qr = Qr::compute(a.as_ref()).unwrap();
///
/// // Apply Q^T to a vector
/// let mut b = Mat::from_rows(&[&[1.0f64], &[2.0], &[3.0]]);
/// ormqr(&qr, Side::Left, Trans::Trans, b.as_mut()).unwrap();
/// ```
pub fn ormqr<T: Field + Real + bytemuck::Zeroable>(
    qr: &Qr<T>,
    side: Side,
    trans: Trans,
    mut c: MatMut<'_, T>,
) -> Result<(), OrthoError> {
    let m = qr.nrows();
    let n = qr.ncols();
    let k = m.min(n);
    let c_rows = c.nrows();
    let c_cols = c.ncols();

    // Dimension checks
    match side {
        Side::Left => {
            if c_rows != m {
                return Err(OrthoError::DimensionMismatch);
            }
        }
        Side::Right => {
            if c_cols != m {
                return Err(OrthoError::DimensionMismatch);
            }
        }
    }

    let qr_mat = qr.qr_factors();
    let tau = qr.tau();

    // Apply Householder reflections
    // The order depends on both side and trans:
    //
    // LEFT side (C = op(Q) * C):
    //   - Q * C: apply H_{k-1}, ..., H_0 (backward) since Q = H_0 * ... * H_{k-1}
    //   - Q^T * C: apply H_0, ..., H_{k-1} (forward) since Q^T = H_{k-1}^T * ... * H_0^T
    //
    // RIGHT side (C = C * op(Q)):
    //   - C * Q: apply H_0, ..., H_{k-1} (forward) since Q = H_0 * ... * H_{k-1}
    //   - C * Q^T: apply H_{k-1}, ..., H_0 (backward) since Q^T = H_{k-1}^T * ... * H_0^T
    let forward = match (side, trans) {
        (Side::Left, Trans::Trans) => true,
        (Side::Left, Trans::NoTrans) => false,
        (Side::Right, Trans::NoTrans) => true,
        (Side::Right, Trans::Trans) => false,
    };

    let (start, end, step): (isize, isize, isize) = if forward {
        (0, k as isize, 1)
    } else {
        ((k - 1) as isize, -1, -1)
    };

    let mut j = start;
    while (step > 0 && j < end) || (step < 0 && j > end) {
        let jj = j as usize;
        if tau[jj] != T::zero() {
            match side {
                Side::Left => {
                    // C = H_j * C where H_j = I - tau * v * v^T
                    for col in 0..c_cols {
                        // w = v^T * C[:, col]
                        let mut w = c[(jj, col)]; // v[j] = 1
                        for i in (jj + 1)..m {
                            w = w + qr_mat[(i, jj)] * c[(i, col)];
                        }

                        // C[:, col] -= tau * w * v
                        let tw = tau[jj] * w;
                        c[(jj, col)] = c[(jj, col)] - tw;
                        for i in (jj + 1)..m {
                            c[(i, col)] = c[(i, col)] - tw * qr_mat[(i, jj)];
                        }
                    }
                }
                Side::Right => {
                    // C = C * H_j where H_j = I - tau * v * v^T
                    for row in 0..c_rows {
                        // w = C[row, :] * v
                        let mut w = c[(row, jj)]; // v[j] = 1
                        for i in (jj + 1)..m {
                            w = w + c[(row, i)] * qr_mat[(i, jj)];
                        }

                        // C[row, :] -= tau * w * v^T
                        let tw = tau[jj] * w;
                        c[(row, jj)] = c[(row, jj)] - tw;
                        for i in (jj + 1)..m {
                            c[(row, i)] = c[(row, i)] - tw * qr_mat[(i, jj)];
                        }
                    }
                }
            }
        }
        j += step;
    }

    Ok(())
}

/// Generates the unitary matrix Q from QR factorization (complex version).
///
/// This is equivalent to LAPACK's ZUNGQR.
pub fn ungqr<T: Field + ComplexScalar + bytemuck::Zeroable>(
    qr_factors: MatRef<'_, T>,
    tau: &[T],
    k: usize,
) -> Result<Mat<T>, OrthoError>
where
    T::Real: Real,
{
    let m = qr_factors.nrows();
    let n = qr_factors.ncols();
    let min_mn = m.min(n);

    if k == 0 || k > min_mn {
        return Err(OrthoError::InvalidParameter);
    }

    if tau.len() < k {
        return Err(OrthoError::InvalidParameter);
    }

    // Start with identity in the first k columns
    let mut q: Mat<T> = Mat::zeros(m, k);
    for i in 0..k {
        q[(i, i)] = T::one();
    }

    // Apply Householder reflections in reverse order
    for j in (0..k).rev() {
        if tau[j].abs() < T::Real::epsilon() {
            continue;
        }

        // Apply H_j = I - tau * v * v^H
        for col in j..k {
            // Compute w = v^H * q[:, col]
            let mut w = q[(j, col)]; // v[j] = 1
            for i in (j + 1)..m {
                w = w + qr_factors[(i, j)].conj() * q[(i, col)];
            }

            // q[:, col] -= tau * w * v
            let tw = tau[j] * w;
            q[(j, col)] = q[(j, col)] - tw;
            for i in (j + 1)..m {
                q[(i, col)] = q[(i, col)] - tw * qr_factors[(i, j)];
            }
        }
    }

    Ok(q)
}

/// Multiplies a matrix by unitary Q from QR factorization (complex version).
///
/// This is equivalent to LAPACK's ZUNMQR.
pub fn unmqr<T: Field + ComplexScalar + bytemuck::Zeroable>(
    qr_factors: MatRef<'_, T>,
    tau: &[T],
    side: Side,
    trans: Trans,
    mut c: MatMut<'_, T>,
) -> Result<(), OrthoError>
where
    T::Real: Real,
{
    let m = qr_factors.nrows();
    let n = qr_factors.ncols();
    let k = m.min(n);
    let c_rows = c.nrows();
    let c_cols = c.ncols();

    // Dimension checks
    match side {
        Side::Left => {
            if c_rows != m {
                return Err(OrthoError::DimensionMismatch);
            }
        }
        Side::Right => {
            if c_cols != m {
                return Err(OrthoError::DimensionMismatch);
            }
        }
    }

    if tau.len() < k {
        return Err(OrthoError::InvalidParameter);
    }

    // Apply Householder reflections
    // For Q: apply H_0, H_1, ..., H_{k-1} (forward)
    // For Q^H: apply H_{k-1}, ..., H_1, H_0 (backward)
    let (start, end, step): (isize, isize, isize) = match trans {
        Trans::NoTrans => (0, k as isize, 1),
        Trans::Trans => ((k - 1) as isize, -1, -1), // Trans means conjugate transpose for complex
    };

    let mut j = start;
    while (step > 0 && j < end) || (step < 0 && j > end) {
        let jj = j as usize;
        if tau[jj].abs() >= T::Real::epsilon() {
            // Use conjugate tau for Q^H
            let tau_eff = if trans == Trans::Trans {
                tau[jj].conj()
            } else {
                tau[jj]
            };

            match side {
                Side::Left => {
                    for col in 0..c_cols {
                        // w = v^H * C[:, col]
                        let mut w = c[(jj, col)];
                        for i in (jj + 1)..m {
                            w = w + qr_factors[(i, jj)].conj() * c[(i, col)];
                        }

                        let tw = tau_eff * w;
                        c[(jj, col)] = c[(jj, col)] - tw;
                        for i in (jj + 1)..m {
                            c[(i, col)] = c[(i, col)] - tw * qr_factors[(i, jj)];
                        }
                    }
                }
                Side::Right => {
                    for row in 0..c_rows {
                        // w = C[row, :] * v
                        let mut w = c[(row, jj)];
                        for i in (jj + 1)..m {
                            w = w + c[(row, i)] * qr_factors[(i, jj)];
                        }

                        let tw = tau_eff * w;
                        c[(row, jj)] = c[(row, jj)] - tw;
                        for i in (jj + 1)..m {
                            c[(row, i)] = c[(row, i)] - tw * qr_factors[(i, jj)].conj();
                        }
                    }
                }
            }
        }
        j += step;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_orgqr_basic() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let qr = Qr::compute(a.as_ref()).unwrap();
        let q = orgqr(&qr, 2).unwrap();

        // Q should be 3×2
        assert_eq!(q.nrows(), 3);
        assert_eq!(q.ncols(), 2);

        // Columns should be orthonormal
        for i in 0..2 {
            for j in 0..2 {
                let mut dot = 0.0;
                for k in 0..3 {
                    dot += q[(k, i)] * q[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(dot, expected, 1e-10),
                    "Column orthonormality failed: Q[:,{}]^T * Q[:,{}] = {}, expected {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_orgqr_matches_q() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let qr = Qr::compute(a.as_ref()).unwrap();
        let q_thin = qr.q_thin();
        let q_orgqr = orgqr(&qr, 2).unwrap();

        // Should match q_thin
        for i in 0..3 {
            for j in 0..2 {
                assert!(
                    approx_eq(q_thin[(i, j)], q_orgqr[(i, j)], 1e-10),
                    "Mismatch at ({}, {}): q_thin={}, orgqr={}",
                    i,
                    j,
                    q_thin[(i, j)],
                    q_orgqr[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_ormqr_left_notrans() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let qr = Qr::compute(a.as_ref()).unwrap();
        let q = qr.q();

        // Compute Q * b using ormqr
        let mut c = Mat::from_rows(&[&[1.0f64], &[2.0], &[3.0]]);
        ormqr(&qr, Side::Left, Trans::NoTrans, c.as_mut()).unwrap();

        // Compute Q * b directly
        let b = Mat::from_rows(&[&[1.0f64], &[2.0], &[3.0]]);
        let mut expected = Mat::zeros(3, 1);
        for i in 0..3 {
            for k in 0..3 {
                expected[(i, 0)] = expected[(i, 0)] + q[(i, k)] * b[(k, 0)];
            }
        }

        // Should match
        for i in 0..3 {
            assert!(
                approx_eq(c[(i, 0)], expected[(i, 0)], 1e-10),
                "Mismatch at {}: ormqr={}, expected={}",
                i,
                c[(i, 0)],
                expected[(i, 0)]
            );
        }
    }

    #[test]
    fn test_ormqr_left_trans() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let qr = Qr::compute(a.as_ref()).unwrap();
        let q = qr.q();

        // Compute Q^T * b using ormqr
        let mut c = Mat::from_rows(&[&[1.0f64], &[2.0], &[3.0]]);
        ormqr(&qr, Side::Left, Trans::Trans, c.as_mut()).unwrap();

        // Compute Q^T * b directly
        let b = Mat::from_rows(&[&[1.0f64], &[2.0], &[3.0]]);
        let mut expected = Mat::zeros(3, 1);
        for i in 0..3 {
            for k in 0..3 {
                expected[(i, 0)] = expected[(i, 0)] + q[(k, i)] * b[(k, 0)];
            }
        }

        // Should match
        for i in 0..3 {
            assert!(
                approx_eq(c[(i, 0)], expected[(i, 0)], 1e-10),
                "Mismatch at {}: ormqr={}, expected={}",
                i,
                c[(i, 0)],
                expected[(i, 0)]
            );
        }
    }

    #[test]
    fn test_ormqr_right_notrans() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let qr = Qr::compute(a.as_ref()).unwrap();
        let q = qr.q();

        // Compute C * Q using ormqr
        let mut c = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        ormqr(&qr, Side::Right, Trans::NoTrans, c.as_mut()).unwrap();

        // Compute C * Q directly
        let b = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let mut expected = Mat::zeros(2, 3);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..3 {
                    expected[(i, j)] = expected[(i, j)] + b[(i, k)] * q[(k, j)];
                }
            }
        }

        // Should match
        for i in 0..2 {
            for j in 0..3 {
                assert!(
                    approx_eq(c[(i, j)], expected[(i, j)], 1e-10),
                    "Mismatch at ({}, {}): ormqr={}, expected={}",
                    i,
                    j,
                    c[(i, j)],
                    expected[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_orgqr_full() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 10.0]]);

        let qr = Qr::compute(a.as_ref()).unwrap();
        let q = orgqr(&qr, 3).unwrap();

        // Should match the full Q
        let q_full = qr.q();

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    approx_eq(q[(i, j)], q_full[(i, j)], 1e-10),
                    "Mismatch at ({}, {})",
                    i,
                    j
                );
            }
        }
    }
}
