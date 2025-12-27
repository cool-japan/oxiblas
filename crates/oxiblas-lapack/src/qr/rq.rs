//! RQ factorization (reversed QR).
//!
//! The RQ factorization decomposes a matrix A (m×n) into:
//! A = R * Q
//!
//! where:
//! - R is an m×n upper trapezoidal matrix
//! - Q is an n×n orthogonal matrix
//!
//! This is the "reversed" version of QR, computed using Householder reflections
//! applied from the right (column operations) from the bottom-right corner.

use crate::error::LapackError;
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// RQ factorization result.
///
/// Contains the R factor and Householder reflector data to reconstruct Q.
#[derive(Debug, Clone)]
pub struct Rq<T: Field> {
    /// Combined R and Householder reflectors in compact storage.
    /// - Upper trapezoidal part contains R
    /// - Below/left contains Householder vectors
    pub(crate) factors: Mat<T>,
    /// Scalar factors (tau) for Householder reflectors.
    pub(crate) tau: Vec<T>,
}

impl<T: Field + Real + bytemuck::Zeroable> Rq<T> {
    /// Computes the RQ factorization of a matrix.
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix (m×n)
    ///
    /// # Returns
    ///
    /// RQ factorization containing R factor and Q reflectors.
    ///
    /// # Errors
    ///
    /// Returns error if the factorization fails.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::qr::Rq;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[1.0f64, 2.0, 3.0],
    ///     &[4.0, 5.0, 6.0],
    /// ]);
    ///
    /// let rq = Rq::compute(a.as_ref()).unwrap();
    /// let r = rq.r_factor();
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

        // RQ factorization: A * H_1 * H_2 * ... * H_k = R
        // where each H_i zeros out elements in row (m - k + i - 1) to the left
        // Apply Householder reflections from the RIGHT (column operations)
        for step in 0..k {
            // Work on row (m - k + step) counting from top
            // But we iterate from bottom, so step 0 works on last row
            let row = m - 1 - step;

            // Householder vector operates on columns [0, n - step)
            // Pivot is at column (n - 1 - step)
            let vec_len = n - step;
            if vec_len == 0 {
                continue;
            }
            let pivot_col = vec_len - 1; // Column index within the segment

            // Compute Householder reflection for this row segment
            // zeros out columns [0, pivot_col) keeping pivot_col
            let (tau_val, beta) = compute_householder_right(&mut factors, row, vec_len);
            tau[step] = tau_val;

            // Store beta at the pivot position (will be part of R)
            factors[(row, pivot_col)] = beta;

            // Apply reflection to all rows (column operation from the right)
            // A <- A * H = A * (I - tau * v * v^H)
            // For each row i: A[i,:] <- A[i,:] - tau * (A[i,:] * v) * v^H
            if tau_val != T::zero() {
                for i in 0..m {
                    if i == row {
                        continue; // Skip the defining row
                    }

                    // Compute w = A[i, 0:vec_len] * v (where v[pivot_col] = 1)
                    let mut w = factors[(i, pivot_col)]; // v[pivot_col] = 1
                    for j in 0..pivot_col {
                        w = w + factors[(i, j)] * factors[(row, j)];
                    }

                    // A[i, 0:vec_len] -= tau * w * v^H
                    let tw = tau_val * w;
                    factors[(i, pivot_col)] = factors[(i, pivot_col)] - tw;
                    for j in 0..pivot_col {
                        factors[(i, j)] = factors[(i, j)] - tw * factors[(row, j)].conj();
                    }
                }
            }
        }

        Ok(Self { factors, tau })
    }

    /// Extracts the R factor (upper trapezoidal).
    #[must_use]
    pub fn r_factor(&self) -> Mat<T> {
        let m = self.factors.nrows();
        let n = self.factors.ncols();

        let mut r = Mat::zeros(m, n);

        for i in 0..m {
            // For RQ factorization, R is upper trapezoidal
            // Wide (m <= n): R in rightmost columns, row i starts at column (n - m + i)
            // Tall (m > n): R in bottom rows
            if m <= n {
                let start_col = n - m + i;
                for j in start_col..n {
                    r[(i, j)] = self.factors[(i, j)];
                }
            } else {
                // For tall matrices, only the last n rows have non-zero R
                if i >= m - n {
                    let local_i = i - (m - n);
                    for j in local_i..n {
                        r[(i, j)] = self.factors[(i, j)];
                    }
                }
            }
        }

        r
    }

    /// Returns the dimensions of the factorization.
    #[must_use]
    pub fn dims(&self) -> (usize, usize) {
        (self.factors.nrows(), self.factors.ncols())
    }

    /// Extracts Q as an explicit orthogonal matrix.
    ///
    /// This generates the full orthogonal matrix Q by applying the stored
    /// Householder reflections.
    #[must_use]
    pub fn q_factor(&self) -> Mat<T> {
        let m = self.factors.nrows();
        let n = self.factors.ncols();
        let k = m.min(n);

        // Q = H_1 * H_2 * ... * H_k
        // Start with identity matrix (n x n)
        let mut q = Mat::zeros(n, n);
        for i in 0..n {
            q[(i, i)] = T::one();
        }

        // Apply Householder reflections in reverse order
        // step=k-1 was the last reflection applied during factorization
        // To build Q, we apply them in forward order
        for step in (0..k).rev() {
            let row = m - 1 - step;
            let vec_len = n - step;
            if vec_len == 0 || self.tau[step] == T::zero() {
                continue;
            }
            let pivot_col = vec_len - 1;

            // Apply H = I - tau * v * v^H to Q from the left
            // Q <- H * Q
            // For column j of Q: Q[:,j] <- Q[:,j] - tau * v * (v^H * Q[:,j])
            for qcol in 0..n {
                // Compute w = v^H * Q[0:vec_len, qcol]
                let mut w = q[(pivot_col, qcol)]; // v[pivot_col] = 1
                for j in 0..pivot_col {
                    w = w + self.factors[(row, j)].conj() * q[(j, qcol)];
                }

                // Q[0:vec_len, qcol] -= tau * v * w
                let tw = self.tau[step] * w;
                q[(pivot_col, qcol)] = q[(pivot_col, qcol)] - tw;
                for j in 0..pivot_col {
                    q[(j, qcol)] = q[(j, qcol)] - tw * self.factors[(row, j)];
                }
            }
        }

        q
    }
}

/// Computes Householder reflection for zeroing left part of a row.
///
/// For a row segment x[0..len], computes H such that x * H = [0, ..., 0, beta]
/// where beta = ±||x|| and H = I - tau * v * v^H.
///
/// Returns (tau, beta) and stores v[0..len-1] in factors[(row, 0..len-1)].
/// v[len-1] = 1 is implicit.
fn compute_householder_right<T: Field + Real>(
    factors: &mut Mat<T>,
    row: usize,
    len: usize,
) -> (T, T) {
    if len == 0 {
        return (T::zero(), T::zero());
    }
    if len == 1 {
        // Single element, no reflection needed
        return (T::zero(), factors[(row, 0)]);
    }

    // Compute norm of the row segment
    let mut norm_sq = T::zero();
    for j in 0..len {
        norm_sq = norm_sq + factors[(row, j)] * factors[(row, j)].conj();
    }
    let norm = Real::sqrt(norm_sq);

    if norm == T::zero() {
        return (T::zero(), T::zero());
    }

    // Pivot element is the rightmost: x[len-1]
    let pivot_idx = len - 1;
    let x_pivot = factors[(row, pivot_idx)];

    // beta = -sign(x_pivot) * ||x||
    let beta = if x_pivot.real() >= T::Real::zero() {
        -norm
    } else {
        norm
    };

    // tau = (beta - x_pivot) / beta
    let tau = (beta - x_pivot) / beta;

    // Scale: v[j] = x[j] / (x_pivot - beta) for j < len-1
    // v[len-1] = 1 (implicit)
    let denom = x_pivot - beta;
    if Scalar::abs(denom) > <T as Scalar>::epsilon() {
        let scale = T::one() / denom;
        for j in 0..pivot_idx {
            factors[(row, j)] = factors[(row, j)] * scale;
        }
    }

    (tau, beta)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_rq_square() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let rq = Rq::compute(a.as_ref()).unwrap();
        let r = rq.r_factor();

        // R should be upper triangular
        assert!(r[(1, 0)].abs() < 1e-10);
    }

    #[test]
    fn test_rq_wide() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]]);

        let rq = Rq::compute(a.as_ref()).unwrap();
        let r = rq.r_factor();
        let q = rq.q_factor();

        assert_eq!(r.nrows(), 2);
        assert_eq!(r.ncols(), 4);
        assert_eq!(q.nrows(), 4);
        assert_eq!(q.ncols(), 4);
    }

    #[test]
    fn test_rq_tall() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let rq = Rq::compute(a.as_ref()).unwrap();
        let r = rq.r_factor();

        assert_eq!(r.nrows(), 3);
        assert_eq!(r.ncols(), 2);
    }

    #[test]
    fn test_rq_q_orthogonal() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let rq = Rq::compute(a.as_ref()).unwrap();
        let q = rq.q_factor();

        // Q^T * Q should be identity
        for i in 0..3 {
            for j in 0..3 {
                let mut dot = 0.0;
                for k in 0..3 {
                    dot += q[(k, i)] * q[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(dot, expected, 1e-10),
                    "Q^T*Q[{},{}] = {}, expected {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_rq_identity() {
        let a: Mat<f64> = Mat::eye(3);

        let rq = Rq::compute(a.as_ref()).unwrap();
        let r = rq.r_factor();
        let q = rq.q_factor();

        // For identity, R and Q should both be close to identity (with possible sign)
        for i in 0..3 {
            assert!(r[(i, i)].abs() > 0.99, "R diagonal should be ~1");
            assert!(q[(i, i)].abs() > 0.99, "Q diagonal should be ~1");
        }
    }

    #[test]
    fn test_rq_reconstruction() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let rq = Rq::compute(a.as_ref()).unwrap();
        let r = rq.r_factor();
        let q = rq.q_factor();

        // A = R * Q
        let m = a.nrows();
        let n = a.ncols();
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += r[(i, k)] * q[(k, j)];
                }
                assert!(
                    approx_eq(sum, a[(i, j)], 1e-10),
                    "Reconstruction[{},{}] = {}, expected {}",
                    i,
                    j,
                    sum,
                    a[(i, j)]
                );
            }
        }
    }
}
