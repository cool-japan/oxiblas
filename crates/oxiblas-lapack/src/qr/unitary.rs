//! Unitary QR decomposition for complex matrices.
//!
//! Computes A = Q·R where Q is unitary (Q^H·Q = I) and R is upper triangular.
//! Uses Householder reflections with complex conjugates.

use num_traits::Zero;
use oxiblas_core::scalar::{ComplexScalar, Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Error type for unitary QR decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitaryQrError {
    /// Matrix is empty.
    EmptyMatrix,
}

impl core::fmt::Display for UnitaryQrError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
        }
    }
}

impl std::error::Error for UnitaryQrError {}

/// Unitary QR decomposition of a complex matrix.
///
/// Stores the decomposition in a compact form:
/// - The upper triangular part contains R
/// - The lower triangular part (below diagonal) contains the Householder vectors
/// - The tau vector contains the Householder scalars
#[derive(Debug, Clone)]
pub struct UnitaryQr<T: Scalar> {
    /// QR factors (compact storage)
    qr: Mat<T>,
    /// Householder scalars
    tau: Vec<T>,
    /// Number of rows
    m: usize,
    /// Number of columns
    n: usize,
}

impl<T: Field + ComplexScalar + bytemuck::Zeroable> UnitaryQr<T>
where
    T::Real: Real,
{
    /// Computes the unitary QR decomposition of a complex matrix A.
    ///
    /// Returns Q and R such that A = QR, where Q is unitary (Q^H Q = I)
    /// and R is upper triangular.
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, UnitaryQrError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(UnitaryQrError::EmptyMatrix);
        }

        // Copy A to working matrix
        let mut qr: Mat<T> = Mat::zeros(m, n);
        for j in 0..n {
            for i in 0..m {
                qr[(i, j)] = a[(i, j)];
            }
        }

        let k = m.min(n);
        let mut tau = vec![T::zero(); k];

        // Apply Householder reflections
        for j in 0..k {
            // Compute Householder vector for column j
            let (tau_j, beta) = complex_householder_vector(&mut qr, j, m);
            tau[j] = tau_j;

            // Update the diagonal element
            qr[(j, j)] = beta;

            // Apply Householder reflection to trailing submatrix
            if j < n - 1 {
                apply_complex_householder_left(&mut qr, j, m, n, tau_j);
            }
        }

        Ok(Self { qr, tau, m, n })
    }

    /// Returns the number of rows in the original matrix.
    pub fn nrows(&self) -> usize {
        self.m
    }

    /// Returns the number of columns in the original matrix.
    pub fn ncols(&self) -> usize {
        self.n
    }

    /// Extracts the R matrix (upper triangular).
    pub fn r(&self) -> Mat<T> {
        let k = self.m.min(self.n);
        let mut r: Mat<T> = Mat::zeros(self.m, self.n);

        for j in 0..self.n {
            for i in 0..=j.min(self.m - 1) {
                r[(i, j)] = self.qr[(i, j)];
            }
        }

        // Zero out below diagonal
        for j in 0..k {
            for i in (j + 1)..self.m {
                r[(i, j)] = T::zero();
            }
        }

        r
    }

    /// Extracts the thin R matrix (k×n where k = min(m, n)).
    pub fn r_thin(&self) -> Mat<T> {
        let k = self.m.min(self.n);
        let mut r: Mat<T> = Mat::zeros(k, self.n);

        for j in 0..self.n {
            for i in 0..=j.min(k - 1) {
                r[(i, j)] = self.qr[(i, j)];
            }
        }

        r
    }

    /// Computes and returns the Q matrix (unitary).
    ///
    /// Returns an m×m unitary matrix where Q^H Q = I.
    pub fn q(&self) -> Mat<T> {
        let k = self.m.min(self.n);

        // Start with identity matrix
        let mut q: Mat<T> = Mat::zeros(self.m, self.m);
        for i in 0..self.m {
            q[(i, i)] = T::one();
        }

        // Apply Householder reflections in reverse order
        for j in (0..k).rev() {
            apply_complex_householder_to_q(&mut q, &self.qr, j, self.m, self.tau[j]);
        }

        q
    }

    /// Computes and returns the thin Q matrix (m×k where k = min(m, n)).
    pub fn q_thin(&self) -> Mat<T> {
        let k = self.m.min(self.n);

        // Start with the first k columns of identity
        let mut q: Mat<T> = Mat::zeros(self.m, k);
        for i in 0..k {
            q[(i, i)] = T::one();
        }

        // Apply Householder reflections in reverse order
        for j in (0..k).rev() {
            apply_complex_householder_to_q_thin(&mut q, &self.qr, j, self.m, k, self.tau[j]);
        }

        q
    }

    /// Returns the Householder scalars (tau values).
    pub fn tau(&self) -> &[T] {
        &self.tau
    }
}

/// Computes the complex Householder vector for column j.
/// Returns (tau, beta) where beta is the new diagonal element.
fn complex_householder_vector<T: Field + ComplexScalar>(
    qr: &mut Mat<T>,
    j: usize,
    m: usize,
) -> (T, T)
where
    T::Real: Real,
{
    // Compute the norm of the column below the diagonal
    // ||x|| = sqrt(sum |x_i|^2)
    let mut norm_sq = T::Real::zero();
    for i in j..m {
        norm_sq = norm_sq + qr[(i, j)].abs_sq();
    }
    let norm = <T::Real as Real>::sqrt(norm_sq);

    if norm == T::Real::zero() {
        return (T::zero(), T::zero());
    }

    let x_j = qr[(j, j)];
    let x_j_abs = x_j.abs();

    // Compute beta = -sign(x_j) * ||x|| where sign is complex unit or 1
    // For numerical stability, we use: beta = -exp(i*arg(x_j)) * ||x|| if x_j != 0
    let beta = if x_j_abs > T::Real::zero() {
        // -sign(x_j) * norm = -(x_j / |x_j|) * norm
        let sign = T::from_real_imag(x_j.real() / x_j_abs, x_j.imag() / x_j_abs);
        T::zero() - sign * T::from_real(norm)
    } else {
        T::from_real(-norm)
    };

    // Compute tau = (beta - x_j) / beta
    let diff = beta - x_j;
    let tau = diff / beta;

    // Scale the Householder vector: v = x / (x_j - beta)
    // v[j] is implicitly 1, store v[j+1:] in qr[j+1:, j]
    let scale_denom = x_j - beta;
    if scale_denom.abs() > T::Real::zero() {
        let scale = T::one() / scale_denom;
        for i in (j + 1)..m {
            qr[(i, j)] = qr[(i, j)] * scale;
        }
    }

    (tau, beta)
}

/// Applies complex Householder reflection to trailing submatrix.
/// H = I - tau * v * v^H
fn apply_complex_householder_left<T: Field + ComplexScalar>(
    qr: &mut Mat<T>,
    j: usize,
    m: usize,
    n: usize,
    tau: T,
) where
    T::Real: Real,
{
    if tau.abs() < T::Real::epsilon() {
        return;
    }

    // Apply H = I - tau * v * v^H to columns j+1..n
    // v[j] = 1, v[j+1:] stored in qr[j+1:, j]
    for k in (j + 1)..n {
        // Compute w = v^H * qr[:, k] = conj(v)^T * qr[:, k]
        let mut w = qr[(j, k)]; // v[j] = 1, conj(1) = 1
        for i in (j + 1)..m {
            // w += conj(v[i]) * qr[i, k]
            w = w + qr[(i, j)].conj() * qr[(i, k)];
        }

        // Update qr[:, k] -= tau * w * v
        let tw = tau * w;
        qr[(j, k)] = qr[(j, k)] - tw; // v[j] = 1
        for i in (j + 1)..m {
            qr[(i, k)] = qr[(i, k)] - tw * qr[(i, j)];
        }
    }
}

/// Applies complex Householder reflection to Q matrix.
fn apply_complex_householder_to_q<T: Field + ComplexScalar>(
    q: &mut Mat<T>,
    qr: &Mat<T>,
    j: usize,
    m: usize,
    tau: T,
) where
    T::Real: Real,
{
    if tau.abs() < T::Real::epsilon() {
        return;
    }

    // Apply H = I - tau * v * v^H to all columns of Q
    for k in 0..m {
        // Compute w = v^H * q[:, k]
        let mut w = q[(j, k)]; // v[j] = 1
        for i in (j + 1)..m {
            w = w + qr[(i, j)].conj() * q[(i, k)];
        }

        // Update q[:, k] -= tau * w * v
        let tw = tau * w;
        q[(j, k)] = q[(j, k)] - tw;
        for i in (j + 1)..m {
            q[(i, k)] = q[(i, k)] - tw * qr[(i, j)];
        }
    }
}

/// Applies complex Householder reflection to thin Q matrix.
fn apply_complex_householder_to_q_thin<T: Field + ComplexScalar>(
    q: &mut Mat<T>,
    qr: &Mat<T>,
    j: usize,
    m: usize,
    ncols: usize,
    tau: T,
) where
    T::Real: Real,
{
    if tau.abs() < T::Real::epsilon() {
        return;
    }

    for k in 0..ncols {
        let mut w = q[(j, k)];
        for i in (j + 1)..m {
            w = w + qr[(i, j)].conj() * q[(i, k)];
        }

        let tw = tau * w;
        q[(j, k)] = q[(j, k)] - tw;
        for i in (j + 1)..m {
            q[(i, k)] = q[(i, k)] - tw * qr[(i, j)];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::{Complex32, Complex64};

    #[test]
    fn test_unitary_qr_simple() {
        // Simple complex matrix
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 1.0)],
            &[Complex64::new(3.0, -1.0), Complex64::new(4.0, 0.0)],
        ]);

        let qr = UnitaryQr::compute(a.as_ref()).expect("Should compute");
        let q = qr.q();
        let r = qr.r();

        // Verify Q is unitary: Q^H * Q = I
        let n = q.nrows();
        for i in 0..n {
            for j in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    // Q^H[i,k] * Q[k,j] = conj(Q[k,i]) * Q[k,j]
                    sum = sum + q[(k, i)].conj() * q[(k, j)];
                }
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                let diff = (sum - expected).norm();
                assert!(
                    diff < 1e-10,
                    "Q^H*Q[{},{}] = {:?}, expected {:?}",
                    i,
                    j,
                    sum,
                    expected
                );
            }
        }

        // Verify R is upper triangular
        assert!(r[(1, 0)].norm() < 1e-10, "R[1,0] = {:?}", r[(1, 0)]);

        // Verify Q * R = A
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..2 {
                    sum = sum + q[(i, k)] * r[(k, j)];
                }
                let diff = (sum - a[(i, j)]).norm();
                assert!(
                    diff < 1e-10,
                    "QR[{},{}] = {:?}, A = {:?}",
                    i,
                    j,
                    sum,
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_unitary_qr_tall() {
        // 3x2 complex matrix
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(1.0, 1.0), Complex64::new(2.0, 0.0)],
            &[Complex64::new(3.0, 0.0), Complex64::new(4.0, -1.0)],
            &[Complex64::new(5.0, -1.0), Complex64::new(6.0, 1.0)],
        ]);

        let qr = UnitaryQr::compute(a.as_ref()).expect("Should compute");
        let q = qr.q();
        let r = qr.r();

        // Q should be 3×3
        assert_eq!(q.nrows(), 3);
        assert_eq!(q.ncols(), 3);

        // Verify Q is unitary
        let n = q.nrows();
        for i in 0..n {
            for j in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    sum = sum + q[(k, i)].conj() * q[(k, j)];
                }
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                let diff = (sum - expected).norm();
                assert!(diff < 1e-10);
            }
        }

        // Verify Q * R = A
        for i in 0..3 {
            for j in 0..2 {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..3 {
                    sum = sum + q[(i, k)] * r[(k, j)];
                }
                let diff = (sum - a[(i, j)]).norm();
                assert!(diff < 1e-10);
            }
        }
    }

    #[test]
    fn test_unitary_qr_hermitian() {
        // Hermitian matrix (should still work)
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(4.0, 0.0), Complex64::new(1.0, -1.0)],
            &[Complex64::new(1.0, 1.0), Complex64::new(3.0, 0.0)],
        ]);

        let qr = UnitaryQr::compute(a.as_ref()).expect("Should compute");
        let q = qr.q();
        let r = qr.r();

        // Verify Q * R = A
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..2 {
                    sum = sum + q[(i, k)] * r[(k, j)];
                }
                let diff = (sum - a[(i, j)]).norm();
                assert!(diff < 1e-10);
            }
        }
    }

    #[test]
    fn test_unitary_qr_complex32() {
        let a: Mat<Complex32> = Mat::from_rows(&[
            &[Complex32::new(1.0, 0.0), Complex32::new(2.0, 1.0)],
            &[Complex32::new(3.0, -1.0), Complex32::new(4.0, 0.0)],
        ]);

        let qr = UnitaryQr::compute(a.as_ref()).expect("Should compute");
        let q = qr.q();
        let r = qr.r();

        // Verify Q * R = A
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = Complex32::new(0.0, 0.0);
                for k in 0..2 {
                    sum = sum + q[(i, k)] * r[(k, j)];
                }
                let diff = (sum - a[(i, j)]).norm();
                assert!(
                    diff < 1e-5,
                    "QR[{},{}] = {:?}, A = {:?}",
                    i,
                    j,
                    sum,
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_unitary_qr_identity() {
        // Complex identity matrix
        let eye: Mat<Complex64> = Mat::eye(3);

        let qr = UnitaryQr::compute(eye.as_ref()).expect("Should compute");
        let q = qr.q();
        let r = qr.r();

        // Q and R should both be close to identity
        for i in 0..3 {
            for j in 0..3 {
                // Allow sign flips on diagonal
                if i == j {
                    assert!(q[(i, j)].norm() > 0.99);
                    assert!(r[(i, j)].norm() > 0.99);
                } else {
                    assert!(q[(i, j)].norm() < 1e-10);
                    assert!(r[(i, j)].norm() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_unitary_qr_thin() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(1.0, 1.0), Complex64::new(2.0, 0.0)],
            &[Complex64::new(3.0, 0.0), Complex64::new(4.0, -1.0)],
            &[Complex64::new(5.0, -1.0), Complex64::new(6.0, 1.0)],
        ]);

        let qr = UnitaryQr::compute(a.as_ref()).expect("Should compute");
        let q_thin = qr.q_thin();
        let r_thin = qr.r_thin();

        // Q_thin should be 3×2
        assert_eq!(q_thin.nrows(), 3);
        assert_eq!(q_thin.ncols(), 2);

        // R_thin should be 2×2
        assert_eq!(r_thin.nrows(), 2);
        assert_eq!(r_thin.ncols(), 2);

        // Verify Q_thin * R_thin = A
        for i in 0..3 {
            for j in 0..2 {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..2 {
                    sum = sum + q_thin[(i, k)] * r_thin[(k, j)];
                }
                let diff = (sum - a[(i, j)]).norm();
                assert!(diff < 1e-10);
            }
        }
    }
}
