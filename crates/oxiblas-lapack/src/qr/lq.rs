//! LQ factorization.
//!
//! The LQ factorization decomposes a matrix A (m×n) into:
//! A = L * Q
//!
//! where:
//! - L is an m×n lower trapezoidal matrix
//! - Q is an n×n orthogonal matrix
//!
//! This is similar to QR but factors into L and Q instead of Q and R.

use crate::error::LapackError;
use oxiblas_core::scalar::{Field, Real};
use oxiblas_matrix::{Mat, MatRef};

/// Compute Householder reflection for a vector.
///
/// Returns (v, tau, alpha) where:
/// - v is the Householder vector (with v[0] = 1 implicitly)
/// - tau is the scalar factor
/// - alpha is the resulting first element after reflection (what becomes L[i,i])
fn compute_householder<T: Field + Real>(x: &[T]) -> (Vec<T>, T, T) {
    let n = x.len();
    if n == 0 {
        return (vec![], T::zero(), T::zero());
    }

    let mut v = x.to_vec();
    let norm_x = Real::sqrt(
        x.iter()
            .map(|&xi| xi * xi.conj())
            .fold(T::zero(), |a, b| a + b),
    );

    if norm_x == T::zero() {
        return (v, T::zero(), x[0]);
    }

    // Compute alpha = -sign(x[0]) * ||x||
    // This is the value that x[0] transforms to after applying H
    let sign = if x[0].real() >= T::Real::zero() {
        T::one()
    } else {
        -T::one()
    };
    let alpha = -sign * norm_x;

    // beta = x[0] - alpha
    let beta = x[0] - alpha;

    if beta == T::zero() {
        return (v, T::zero(), alpha);
    }

    // Compute tau = 1 - x[0]/alpha = (alpha - x[0])/alpha = -beta/alpha
    let tau = -beta / alpha;

    // Normalize v so that v[0] = 1 (stored implicitly)
    // v[j] = x[j] / beta for j > 0
    v[0] = T::one();
    for i in 1..n {
        v[i] = x[i] / beta;
    }

    (v, tau, alpha)
}

/// LQ factorization result.
///
/// Contains the L factor and Householder reflector data to reconstruct Q.
#[derive(Debug, Clone)]
pub struct Lq<T: Field> {
    /// Combined L and Householder reflectors in compact storage.
    /// - Lower trapezoidal part contains L
    /// - Above diagonal contains Householder vectors
    pub(crate) factors: Mat<T>,
    /// Scalar factors (tau) for Householder reflectors.
    pub(crate) tau: Vec<T>,
}

impl<T: Field + Real + bytemuck::Zeroable> Lq<T> {
    /// Computes the LQ factorization of a matrix.
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix (m×n)
    ///
    /// # Returns
    ///
    /// LQ factorization containing L factor and Q reflectors.
    ///
    /// # Errors
    ///
    /// Returns error if the factorization fails.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::qr::Lq;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[1.0f64, 2.0, 3.0],
    ///     &[4.0, 5.0, 6.0],
    /// ]);
    ///
    /// let lq = Lq::compute(a.as_ref()).unwrap();
    /// let l = lq.l_factor();
    /// ```
    pub fn compute(a: MatRef<T>) -> Result<Self, LapackError> {
        let m = a.nrows();
        let n = a.ncols();
        let k = m.min(n);

        // Copy A to working matrix
        let mut factors = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                factors[(i, j)] = a[(i, j)];
            }
        }

        let mut tau = vec![T::zero(); k];

        // Perform Householder reflections row by row
        for i in 0..k {
            // Extract row starting from diagonal
            let row_len = n - i;
            let mut x = vec![T::zero(); row_len];
            for j in 0..row_len {
                x[j] = factors[(i, i + j)];
            }

            // Compute Householder reflection
            let (v, tau_val, alpha) = compute_householder(&x);
            tau[i] = tau_val;

            // Store the transformed diagonal element (L[i,i])
            factors[(i, i)] = alpha;

            // Apply reflection to current and remaining rows
            if tau_val != T::zero() {
                // Store Householder vector (v[0] = 1 implicitly, store v[1..])
                for j in 1..row_len {
                    factors[(i, i + j)] = v[j];
                }

                // Apply reflection to rows below: H = I - tau * v * v^H
                for r in (i + 1)..m {
                    let mut dot = T::zero();
                    for j in 0..row_len {
                        dot = dot + factors[(r, i + j)] * v[j].conj();
                    }

                    let scale = tau_val * dot;
                    for j in 0..row_len {
                        factors[(r, i + j)] = factors[(r, i + j)] - scale * v[j];
                    }
                }
            }
        }

        Ok(Self { factors, tau })
    }

    /// Extracts the L factor (lower trapezoidal).
    #[must_use]
    pub fn l_factor(&self) -> Mat<T> {
        let m = self.factors.nrows();
        let n = self.factors.ncols();

        let mut l = Mat::zeros(m, n);

        for i in 0..m {
            for j in 0..=i.min(n - 1) {
                l[(i, j)] = self.factors[(i, j)];
            }
        }

        l
    }

    /// Returns the dimensions of the factorization.
    #[must_use]
    pub fn dims(&self) -> (usize, usize) {
        (self.factors.nrows(), self.factors.ncols())
    }

    /// Extracts Q as an explicit matrix.
    ///
    /// This generates the full orthogonal matrix Q by applying the stored
    /// Householder reflections.
    #[must_use]
    pub fn q_factor(&self) -> Mat<T> {
        let m = self.factors.nrows();
        let n = self.factors.ncols();
        let k = m.min(n);

        // Start with identity
        let mut q = Mat::zeros(n, n);
        for i in 0..n {
            q[(i, i)] = T::one();
        }

        // Apply Householder reflections in forward order (k-1 down to 0)
        // This builds Q = H_{k-1} * ... * H_1 * H_0
        // which gives A = L * Q correctly since A * H_0 * H_1 * ... = L
        for i in (0..k).rev() {
            if self.tau[i] == T::zero() {
                continue;
            }

            let row_len = n - i;
            let mut v = vec![T::one(); row_len];
            for j in 1..row_len {
                v[j] = self.factors[(i, i + j)];
            }

            // Apply H = I - tau * v * v^H to Q from the right (not left!)
            // Q_new = Q_old * H = Q_old - tau * Q_old * v * v^H
            for c in 0..n {
                let mut dot = T::zero();
                for j in 0..row_len {
                    dot = dot + v[j].conj() * q[(c, i + j)];
                }

                let scale = self.tau[i] * dot;
                for j in 0..row_len {
                    q[(c, i + j)] = q[(c, i + j)] - scale * v[j];
                }
            }
        }

        q
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lq_square() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let lq = Lq::compute(a.as_ref()).unwrap();
        let l = lq.l_factor();

        // L should be lower triangular
        assert!(l[(0, 1)].abs() < 1e-10);
    }

    #[test]
    fn test_lq_wide() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let lq = Lq::compute(a.as_ref()).unwrap();
        let l = lq.l_factor();
        let q = lq.q_factor();

        assert_eq!(l.nrows(), 2);
        assert_eq!(l.ncols(), 3);
        assert_eq!(q.nrows(), 3);
        assert_eq!(q.ncols(), 3);

        // Verify L is lower trapezoidal
        assert!(l[(0, 1)].abs() < 1e-10);
        assert!(l[(0, 2)].abs() < 1e-10);
        assert!(l[(1, 2)].abs() < 1e-10);
    }

    #[test]
    fn test_lq_reconstruction() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let lq = Lq::compute(a.as_ref()).unwrap();
        let l = lq.l_factor();
        let q = lq.q_factor();

        // Reconstruct A = L * Q
        let mut reconstructed = Mat::zeros(2, 3);
        for i in 0..2 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += l[(i, k)] * q[(k, j)];
                }
                reconstructed[(i, j)] = sum;
            }
        }

        // Check reconstruction
        for i in 0..2 {
            for j in 0..3 {
                assert!((reconstructed[(i, j)] - a[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_lq_q_orthogonal() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let lq = Lq::compute(a.as_ref()).unwrap();
        let q = lq.q_factor();

        // Q^T * Q should be identity
        let mut qtq = Mat::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += q[(k, i)] * q[(k, j)];
                }
                qtq[(i, j)] = sum;
            }
        }

        // Check orthogonality
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((qtq[(i, j)] - expected).abs() < 1e-10);
            }
        }
    }
}
