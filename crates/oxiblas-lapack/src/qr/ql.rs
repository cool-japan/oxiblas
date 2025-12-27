//! QL factorization.
//!
//! The QL factorization decomposes a matrix A (m×n) into:
//! A = Q * L
//!
//! where:
//! - Q is an m×m orthogonal matrix
//! - L is an m×n lower trapezoidal matrix
//!
//! This is computed using Householder reflections applied from the
//! bottom-right corner, effectively working backwards compared to QR.

use crate::error::LapackError;
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// QL factorization result.
///
/// Stores the decomposition in compact form with Householder reflectors.
#[derive(Debug, Clone)]
pub struct Ql<T: Scalar> {
    /// Combined L and Householder reflectors in compact storage.
    factors: Mat<T>,
    /// Scalar factors (tau) for Householder reflectors.
    tau: Vec<T>,
    /// Number of rows.
    m: usize,
    /// Number of columns.
    n: usize,
}

impl<T: Field + Real + bytemuck::Zeroable> Ql<T> {
    /// Computes the QL factorization of a matrix.
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix (m×n)
    ///
    /// # Returns
    ///
    /// QL factorization containing L factor and Q reflectors.
    ///
    /// # Errors
    ///
    /// Returns error if the factorization fails.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::qr::Ql;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[1.0f64, 2.0],
    ///     &[3.0, 4.0],
    ///     &[5.0, 6.0],
    /// ]);
    ///
    /// let ql = Ql::compute(a.as_ref()).unwrap();
    /// let l = ql.l_factor();
    /// ```
    pub fn compute(a: MatRef<T>) -> Result<Self, LapackError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(LapackError::new(
                crate::error::ErrorCode::InvalidDimension {
                    argument: 1,
                    expected: 1,
                    actual: 0,
                },
                "QL factorization",
            ));
        }

        let k = m.min(n);

        // Copy A to working matrix
        let mut factors = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                factors[(i, j)] = a[(i, j)];
            }
        }

        let mut tau = vec![T::zero(); k];

        // QL factorization: work from bottom-right to top-left
        // For column j (from n-1 down to n-k), apply Householder to zero out
        // elements above the anti-diagonal
        for i in 0..k {
            // Column we're working on (rightmost to leftmost of the k columns)
            let col = n - 1 - i;
            // Number of rows in the Householder vector (decreasing)
            let vec_len = m - i;

            if vec_len == 0 {
                continue;
            }

            // Compute Householder vector to zero out elements [0..vec_len-1]
            // keeping element [vec_len-1] as the pivot
            let (tau_i, beta) = compute_householder_bottom(&mut factors, col, vec_len);
            tau[i] = tau_i;

            // Store the transformed pivot
            factors[(vec_len - 1, col)] = beta;

            // Apply the Householder transformation to columns 0..col
            if tau_i != T::zero() && col > 0 {
                apply_householder_bottom(&mut factors, col, vec_len, tau_i);
            }
        }

        Ok(Self { factors, tau, m, n })
    }

    /// Extracts the L factor (lower trapezoidal).
    #[must_use]
    pub fn l_factor(&self) -> Mat<T> {
        let mut l = Mat::zeros(self.m, self.n);

        // For QL factorization, L is lower trapezoidal
        // The lower triangular part is in the bottom-right corner
        //
        // For tall (m >= n): L is m×n, lower trapezoidal with n×n lower triangular in bottom-right
        // For wide (m < n): L is m×n, lower trapezoidal with m×m lower triangular in bottom-right

        for i in 0..self.m {
            if self.m >= self.n {
                // Tall or square matrix
                if i < self.m - self.n {
                    // These rows are zero in L (above the lower triangular block)
                    // Keep them as zeros (already initialized)
                } else {
                    // Part of the lower triangular block
                    for j in 0..self.n {
                        // Non-zero if j <= n - 1 - (m - 1 - i) = n - m + i
                        if j <= self.n + i - self.m {
                            l[(i, j)] = self.factors[(i, j)];
                        }
                    }
                }
            } else {
                // Wide matrix (m < n)
                // L is lower trapezoidal: zeros in upper-right triangle
                // Row i can have non-zero values in columns 0 through (n - m + i)
                for j in 0..self.n {
                    if j <= self.n - self.m + i {
                        l[(i, j)] = self.factors[(i, j)];
                    }
                }
            }
        }

        l
    }

    /// Extracts Q as an explicit orthogonal matrix.
    #[must_use]
    pub fn q_factor(&self) -> Mat<T> {
        let k = self.m.min(self.n);

        // Start with identity
        let mut q = Mat::zeros(self.m, self.m);
        for i in 0..self.m {
            q[(i, i)] = T::one();
        }

        // Apply Householder reflections in reverse order (forward from stored order)
        // We stored tau[0] for the last column processed, etc.
        for i in (0..k).rev() {
            let col = self.n - 1 - i;
            let vec_len = self.m - i;

            if self.tau[i] == T::zero() {
                continue;
            }

            // Apply H = I - tau * v * v^T to Q
            // v is stored in factors[0..vec_len-1, col] with v[vec_len-1] = 1
            for qcol in 0..self.m {
                // Compute w = v^T * q[:vec_len, qcol]
                let mut w = q[(vec_len - 1, qcol)]; // v[vec_len-1] = 1
                for row in 0..(vec_len - 1) {
                    w = w + self.factors[(row, col)] * q[(row, qcol)];
                }

                // q[:vec_len, qcol] -= tau * w * v
                let tw = self.tau[i] * w;
                q[(vec_len - 1, qcol)] = q[(vec_len - 1, qcol)] - tw;
                for row in 0..(vec_len - 1) {
                    q[(row, qcol)] = q[(row, qcol)] - tw * self.factors[(row, col)];
                }
            }
        }

        q
    }

    /// Returns the dimensions of the factorization.
    #[must_use]
    pub fn dims(&self) -> (usize, usize) {
        (self.m, self.n)
    }
}

/// Computes Householder vector for QL factorization.
/// Zeros out elements [0..len-1], keeping element [len-1] as pivot.
fn compute_householder_bottom<T: Field + Real>(
    factors: &mut Mat<T>,
    col: usize,
    len: usize,
) -> (T, T) {
    if len == 0 {
        return (T::zero(), T::zero());
    }

    // Compute norm of column segment [0..len]
    let mut norm_sq = T::zero();
    for i in 0..len {
        norm_sq = norm_sq + factors[(i, col)] * factors[(i, col)];
    }
    let norm = Real::sqrt(norm_sq);

    if norm == T::zero() {
        return (T::zero(), T::zero());
    }

    // Pivot element is at position len-1 (bottom of the vector)
    let x_pivot = factors[(len - 1, col)];

    // beta = -sign(x_pivot) * ||x||
    let beta = if x_pivot >= T::zero() { -norm } else { norm };

    // tau = (beta - x_pivot) / beta
    let tau = (beta - x_pivot) / beta;

    // Scale v: v[i] = x[i] / (x_pivot - beta) for i < len-1
    // v[len-1] = 1 (implicit)
    let denom = x_pivot - beta;
    if Scalar::abs(denom) > <T as Scalar>::epsilon() {
        let scale = T::one() / denom;
        for i in 0..(len - 1) {
            factors[(i, col)] = factors[(i, col)] * scale;
        }
    }

    (tau, beta)
}

/// Applies Householder reflection for QL factorization.
fn apply_householder_bottom<T: Field + Real>(factors: &mut Mat<T>, col: usize, len: usize, tau: T) {
    if tau == T::zero() || len == 0 {
        return;
    }

    // Apply H = I - tau * v * v^T to columns 0..col
    // v is stored in factors[0..len-1, col] with v[len-1] = 1
    for k in 0..col {
        // Compute w = v^T * factors[:len, k]
        let mut w = factors[(len - 1, k)]; // v[len-1] = 1
        for i in 0..(len - 1) {
            w = w + factors[(i, col)] * factors[(i, k)];
        }

        // factors[:len, k] -= tau * w * v
        let tw = tau * w;
        factors[(len - 1, k)] = factors[(len - 1, k)] - tw;
        for i in 0..(len - 1) {
            factors[(i, k)] = factors[(i, k)] - tw * factors[(i, col)];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_ql_square() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let ql = Ql::compute(a.as_ref()).unwrap();
        let l = ql.l_factor();
        let q = ql.q_factor();

        // L should be lower triangular
        assert!(
            l[(0, 1)].abs() < 1e-10,
            "L should be lower triangular, L[0,1] = {}",
            l[(0, 1)]
        );

        // Q should be orthogonal
        for i in 0..2 {
            for j in 0..2 {
                let mut dot = 0.0;
                for k in 0..2 {
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

        // Q * L should equal A
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += q[(i, k)] * l[(k, j)];
                }
                assert!(
                    approx_eq(sum, a[(i, j)], 1e-10),
                    "QL[{},{}] = {}, A = {}",
                    i,
                    j,
                    sum,
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_ql_tall() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let ql = Ql::compute(a.as_ref()).unwrap();
        let l = ql.l_factor();
        let q = ql.q_factor();

        assert_eq!(l.nrows(), 3);
        assert_eq!(l.ncols(), 2);
        assert_eq!(q.nrows(), 3);
        assert_eq!(q.ncols(), 3);

        // Q should be orthogonal
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

        // Q * L should equal A
        for i in 0..3 {
            for j in 0..2 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += q[(i, k)] * l[(k, j)];
                }
                assert!(
                    approx_eq(sum, a[(i, j)], 1e-10),
                    "QL[{},{}] = {}, A = {}",
                    i,
                    j,
                    sum,
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_ql_wide() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let ql = Ql::compute(a.as_ref()).unwrap();
        let l = ql.l_factor();
        let q = ql.q_factor();

        assert_eq!(l.nrows(), 2);
        assert_eq!(l.ncols(), 3);
        assert_eq!(q.nrows(), 2);
        assert_eq!(q.ncols(), 2);

        // Q should be orthogonal
        for i in 0..2 {
            for j in 0..2 {
                let mut dot = 0.0;
                for k in 0..2 {
                    dot += q[(k, i)] * q[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(dot, expected, 1e-10),
                    "Q^T*Q[{},{}] = {}",
                    i,
                    j,
                    dot
                );
            }
        }

        // Q * L should equal A
        for i in 0..2 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += q[(i, k)] * l[(k, j)];
                }
                assert!(
                    approx_eq(sum, a[(i, j)], 1e-10),
                    "QL[{},{}] = {}, A = {}",
                    i,
                    j,
                    sum,
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_ql_identity() {
        let a: Mat<f64> = Mat::eye(3);

        let ql = Ql::compute(a.as_ref()).unwrap();
        let l = ql.l_factor();
        let q = ql.q_factor();

        // Both Q and L should be close to identity (with possible sign flips)
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert!(q[(i, j)].abs() > 0.99);
                    assert!(l[(i, j)].abs() > 0.99);
                } else {
                    assert!(q[(i, j)].abs() < 1e-10);
                    assert!(l[(i, j)].abs() < 1e-10);
                }
            }
        }
    }
}
