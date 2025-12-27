//! SVD using one-sided Jacobi algorithm.
//!
//! Computes A = U·Σ·V^T where:
//! - U is m×m orthogonal (left singular vectors)
//! - Σ is m×n diagonal (singular values, non-negative, descending)
//! - V is n×n orthogonal (right singular vectors)

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Error type for SVD computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvdError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Algorithm did not converge.
    NotConverged,
}

impl core::fmt::Display for SvdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::NotConverged => write!(f, "SVD algorithm did not converge"),
        }
    }
}

impl std::error::Error for SvdError {}

/// Singular Value Decomposition result.
///
/// A = U·Σ·V^T where Σ contains singular values on the diagonal.
#[derive(Debug, Clone)]
pub struct Svd<T: Scalar> {
    /// Left singular vectors (m×m orthogonal matrix).
    u: Mat<T>,
    /// Singular values (sorted in descending order).
    sigma: Vec<T>,
    /// Right singular vectors (n×n orthogonal matrix, stored as V^T).
    vt: Mat<T>,
    /// Original matrix dimensions.
    m: usize,
    n: usize,
}

impl<T: Field + Real + bytemuck::Zeroable> Svd<T> {
    /// Maximum sweeps for Jacobi iteration.
    const MAX_SWEEPS: usize = 30;

    /// Computes the full SVD of matrix A using one-sided Jacobi algorithm.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::svd::Svd;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[3.0f64, 0.0],
    ///     &[0.0, 4.0],
    /// ]);
    ///
    /// let svd = Svd::compute(a.as_ref()).unwrap();
    /// let sigma = svd.singular_values();
    ///
    /// // Singular values of diagonal matrix are absolute values of diagonal
    /// assert!((sigma[0] - 4.0).abs() < 1e-10);
    /// assert!((sigma[1] - 3.0).abs() < 1e-10);
    /// ```
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, SvdError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(SvdError::EmptyMatrix);
        }

        // Handle 1x1 case
        if m == 1 && n == 1 {
            let val = a[(0, 0)];
            let sigma = vec![Scalar::abs(val)];
            let mut u = Mat::zeros(1, 1);
            let mut vt = Mat::zeros(1, 1);
            u[(0, 0)] = if val >= T::zero() {
                T::one()
            } else {
                -T::one()
            };
            vt[(0, 0)] = T::one();
            return Ok(Self { u, sigma, vt, m, n });
        }

        // Copy A into working matrix B (we'll work on B to get V, then compute U)
        let mut b = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                b[(i, j)] = a[(i, j)];
            }
        }

        // Initialize V as identity
        let mut v = Mat::zeros(n, n);
        for i in 0..n {
            v[(i, i)] = T::one();
        }

        let eps = <T as Scalar>::epsilon();
        let tol = eps * T::from_f64(100.0).unwrap_or(T::one());

        // One-sided Jacobi: Apply Jacobi rotations to columns of B to diagonalize B^T*B
        // After convergence, columns of B are orthogonal (and are U*Σ), V accumulates rotations
        let k = m.min(n);

        for _sweep in 0..Self::MAX_SWEEPS {
            let mut converged = true;

            // Sweep through all column pairs (i, j) with i < j
            // Only sweep up to min(m,n) for the first index since we can have at most k singular values
            for i in 0..n {
                for j in (i + 1)..n {
                    // Compute B[:, i]^T * B[:, j] and norms
                    let mut dot_ij = T::zero();
                    let mut norm_i_sq = T::zero();
                    let mut norm_j_sq = T::zero();

                    for row in 0..m {
                        let bi = b[(row, i)];
                        let bj = b[(row, j)];
                        dot_ij = dot_ij + bi * bj;
                        norm_i_sq = norm_i_sq + bi * bi;
                        norm_j_sq = norm_j_sq + bj * bj;
                    }

                    // Check if rotation is needed
                    let off_diag = Scalar::abs(dot_ij);
                    let threshold = tol * Real::sqrt(norm_i_sq) * Real::sqrt(norm_j_sq);

                    if off_diag > threshold {
                        converged = false;

                        // Compute Jacobi rotation to zero dot_ij
                        // We want to find angle θ such that after rotation, columns are orthogonal
                        // The 2x2 Gram matrix is [[norm_i_sq, dot_ij], [dot_ij, norm_j_sq]]
                        // Jacobi rotation diagonalizes this

                        let tau = (norm_j_sq - norm_i_sq)
                            / (T::from_f64(2.0).unwrap_or(T::one()) * dot_ij);
                        let t = if tau >= T::zero() {
                            T::one() / (tau + Real::sqrt(T::one() + tau * tau))
                        } else {
                            T::one() / (tau - Real::sqrt(T::one() + tau * tau))
                        };
                        let c = T::one() / Real::sqrt(T::one() + t * t);
                        let s = c * t;

                        // Apply rotation to columns i and j of B
                        for row in 0..m {
                            let bi = b[(row, i)];
                            let bj = b[(row, j)];
                            b[(row, i)] = c * bi - s * bj;
                            b[(row, j)] = s * bi + c * bj;
                        }

                        // Accumulate rotation into V
                        for row in 0..n {
                            let vi = v[(row, i)];
                            let vj = v[(row, j)];
                            v[(row, i)] = c * vi - s * vj;
                            v[(row, j)] = s * vi + c * vj;
                        }
                    }
                }
            }

            if converged {
                break;
            }
        }

        // Now columns of B are orthogonal. The singular values are the norms of columns.
        // For an m×n matrix, we have min(m,n) singular values.
        // We need to find the columns with the largest norms.

        // Compute norms of all columns
        let mut col_norms: Vec<(usize, T)> = Vec::with_capacity(n);
        for j in 0..n {
            let mut norm_sq = T::zero();
            for row in 0..m {
                norm_sq = norm_sq + b[(row, j)] * b[(row, j)];
            }
            col_norms.push((j, Real::sqrt(norm_sq)));
        }

        // Sort columns by norm (descending)
        col_norms.sort_by(|a, b| {
            if b.1 > a.1 {
                core::cmp::Ordering::Greater
            } else if b.1 < a.1 {
                core::cmp::Ordering::Less
            } else {
                core::cmp::Ordering::Equal
            }
        });

        // Extract singular values and construct U
        let mut sigma = vec![T::zero(); k];
        let mut u = Mat::zeros(m, m);

        for i in 0..k {
            let (col_idx, norm) = col_norms[i];
            sigma[i] = norm;

            if norm > tol {
                for row in 0..m {
                    u[(row, i)] = b[(row, col_idx)] / norm;
                }
            } else {
                // Zero singular value - will be filled by Gram-Schmidt below
            }
        }

        // Complete U to full orthogonal matrix using Gram-Schmidt
        for col in 0..k {
            if sigma[col] <= tol {
                // Find a standard basis vector orthogonal to existing columns
                for basis in 0..m {
                    u[(basis, col)] = T::one();

                    // Orthogonalize against previous columns
                    for j in 0..col {
                        let mut dot = T::zero();
                        for row in 0..m {
                            dot = dot + u[(row, col)] * u[(row, j)];
                        }
                        for row in 0..m {
                            u[(row, col)] = u[(row, col)] - dot * u[(row, j)];
                        }
                    }

                    // Normalize
                    let mut norm_sq = T::zero();
                    for row in 0..m {
                        norm_sq = norm_sq + u[(row, col)] * u[(row, col)];
                    }
                    let norm = Real::sqrt(norm_sq);
                    if norm > tol {
                        for row in 0..m {
                            u[(row, col)] = u[(row, col)] / norm;
                        }
                        break;
                    }
                    // Reset and try next basis vector
                    for row in 0..m {
                        u[(row, col)] = T::zero();
                    }
                }
            }
        }

        // Fill remaining columns of U using Gram-Schmidt
        for i in k..m {
            for basis in 0..m {
                u[(basis, i)] = T::one();

                // Orthogonalize against all previous columns
                for j in 0..i {
                    let mut dot = T::zero();
                    for row in 0..m {
                        dot = dot + u[(row, i)] * u[(row, j)];
                    }
                    for row in 0..m {
                        u[(row, i)] = u[(row, i)] - dot * u[(row, j)];
                    }
                }

                // Normalize
                let mut norm_sq = T::zero();
                for row in 0..m {
                    norm_sq = norm_sq + u[(row, i)] * u[(row, i)];
                }
                let norm = Real::sqrt(norm_sq);
                if norm > tol {
                    for row in 0..m {
                        u[(row, i)] = u[(row, i)] / norm;
                    }
                    break;
                }
                // Reset and try next basis vector
                for row in 0..m {
                    u[(row, i)] = T::zero();
                }
            }
        }

        // Construct V^T with columns reordered according to col_norms
        // V^T[i, :] should correspond to sigma[i]
        let mut vt = Mat::zeros(n, n);

        // First k rows of V^T correspond to the k singular values
        for i in 0..k {
            let (col_idx, _) = col_norms[i];
            for j in 0..n {
                vt[(i, j)] = v[(j, col_idx)];
            }
        }

        // Fill remaining rows of V^T with orthogonal complement
        for i in k..n {
            let (col_idx, _) = col_norms[i];
            for j in 0..n {
                vt[(i, j)] = v[(j, col_idx)];
            }
        }

        // Singular values are already sorted descending from col_norms extraction
        // No need for additional sorting

        Ok(Self { u, sigma, vt, m, n })
    }

    /// Returns the singular values (sorted in descending order).
    pub fn singular_values(&self) -> &[T] {
        &self.sigma
    }

    /// Returns the left singular vectors U (m×m orthogonal matrix).
    pub fn u(&self) -> MatRef<'_, T> {
        self.u.as_ref()
    }

    /// Returns V^T (n×n orthogonal matrix).
    ///
    /// Note: This returns V^T, not V. To get V, transpose this.
    pub fn vt(&self) -> MatRef<'_, T> {
        self.vt.as_ref()
    }

    /// Returns the thin U matrix (m×k where k = min(m,n)).
    pub fn u_thin(&self) -> Mat<T> {
        let k = self.m.min(self.n);
        let mut u_thin = Mat::zeros(self.m, k);
        for i in 0..self.m {
            for j in 0..k {
                u_thin[(i, j)] = self.u[(i, j)];
            }
        }
        u_thin
    }

    /// Returns the thin V^T matrix (k×n where k = min(m,n)).
    pub fn vt_thin(&self) -> Mat<T> {
        let k = self.m.min(self.n);
        let mut vt_thin = Mat::zeros(k, self.n);
        for i in 0..k {
            for j in 0..self.n {
                vt_thin[(i, j)] = self.vt[(i, j)];
            }
        }
        vt_thin
    }

    /// Computes the rank of the matrix given a tolerance.
    pub fn rank(&self, tol: T) -> usize {
        self.sigma.iter().filter(|&&s| s > tol).count()
    }

    /// Computes the 2-norm (largest singular value).
    pub fn norm2(&self) -> T {
        if self.sigma.is_empty() {
            T::zero()
        } else {
            self.sigma[0]
        }
    }

    /// Computes the condition number (ratio of largest to smallest singular value).
    pub fn condition_number(&self) -> T {
        if self.sigma.is_empty() {
            T::zero()
        } else {
            let max_sv = self.sigma[0];
            let min_sv = self.sigma[self.sigma.len() - 1];
            if min_sv > T::zero() {
                max_sv / min_sv
            } else {
                <T as Scalar>::max_value()
            }
        }
    }

    /// Reconstructs the original matrix: A = U·Σ·V^T
    pub fn reconstruct(&self) -> Mat<T> {
        let mut a = Mat::zeros(self.m, self.n);
        let k = self.m.min(self.n);

        for i in 0..self.m {
            for j in 0..self.n {
                let mut sum = T::zero();
                for l in 0..k {
                    sum = sum + self.u[(i, l)] * self.sigma[l] * self.vt[(l, j)];
                }
                a[(i, j)] = sum;
            }
        }

        a
    }

    /// Computes the pseudoinverse using SVD.
    ///
    /// A^+ = V·Σ^+·U^T where Σ^+ has 1/σ_i on diagonal (for non-zero σ_i).
    pub fn pseudoinverse(&self, tol: T) -> Mat<T> {
        let mut pinv = Mat::zeros(self.n, self.m);
        let k = self.m.min(self.n);

        for i in 0..self.n {
            for j in 0..self.m {
                let mut sum = T::zero();
                for l in 0..k {
                    if self.sigma[l] > tol {
                        // V[i,l] * (1/sigma[l]) * U[j,l]
                        // V = Vt^T, so V[i,l] = Vt[l,i]
                        sum = sum + self.vt[(l, i)] * (T::one() / self.sigma[l]) * self.u[(j, l)];
                    }
                }
                pinv[(i, j)] = sum;
            }
        }

        pinv
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_svd_diagonal() {
        let a = Mat::from_rows(&[&[3.0f64, 0.0], &[0.0, 4.0]]);

        let svd = Svd::compute(a.as_ref()).unwrap();
        let sigma = svd.singular_values();

        assert!(approx_eq(sigma[0], 4.0, 1e-10));
        assert!(approx_eq(sigma[1], 3.0, 1e-10));
    }

    #[test]
    fn test_svd_2x2() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let svd = Svd::compute(a.as_ref()).unwrap();
        let reconstructed = svd.reconstruct();

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-9),
                    "reconstructed[{},{}] = {}, a = {}",
                    i,
                    j,
                    reconstructed[(i, j)],
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_svd_tall() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let svd = Svd::compute(a.as_ref()).unwrap();
        assert_eq!(svd.singular_values().len(), 2);

        let reconstructed = svd.reconstruct();
        for i in 0..3 {
            for j in 0..2 {
                assert!(approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-9));
            }
        }
    }

    #[test]
    fn test_svd_wide() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let svd = Svd::compute(a.as_ref()).unwrap();
        assert_eq!(svd.singular_values().len(), 2);

        let reconstructed = svd.reconstruct();
        for i in 0..2 {
            for j in 0..3 {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-9),
                    "reconstructed[{},{}] = {}, a = {}",
                    i,
                    j,
                    reconstructed[(i, j)],
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_svd_identity() {
        let eye = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);

        let svd = Svd::compute(eye.as_ref()).unwrap();
        let sigma = svd.singular_values();

        for &s in sigma {
            assert!(approx_eq(s, 1.0, 1e-10));
        }
    }

    #[test]
    fn test_svd_single() {
        let a = Mat::from_rows(&[&[5.0f64]]);

        let svd = Svd::compute(a.as_ref()).unwrap();
        let sigma = svd.singular_values();

        assert_eq!(sigma.len(), 1);
        assert!(approx_eq(sigma[0], 5.0, 1e-10));
    }

    #[test]
    fn test_svd_rank() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[2.0, 4.0, 6.0], &[3.0, 6.0, 9.0]]);

        let svd = Svd::compute(a.as_ref()).unwrap();
        assert!(svd.rank(1e-10) == 1);
    }

    #[test]
    fn test_svd_norm() {
        let a = Mat::from_rows(&[&[3.0f64, 0.0], &[0.0, 4.0]]);

        let svd = Svd::compute(a.as_ref()).unwrap();
        assert!(approx_eq(svd.norm2(), 4.0, 1e-10));
    }

    #[test]
    fn test_svd_condition() {
        let a = Mat::from_rows(&[&[2.0f64, 0.0], &[0.0, 4.0]]);

        let svd = Svd::compute(a.as_ref()).unwrap();
        assert!(approx_eq(svd.condition_number(), 2.0, 1e-10));
    }

    #[test]
    fn test_svd_pseudoinverse() {
        let a = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 2.0]]);

        let svd = Svd::compute(a.as_ref()).unwrap();
        let pinv = svd.pseudoinverse(1e-10);

        assert!(approx_eq(pinv[(0, 0)], 1.0, 1e-10));
        assert!(approx_eq(pinv[(1, 1)], 0.5, 1e-10));
        assert!(approx_eq(pinv[(0, 1)], 0.0, 1e-10));
        assert!(approx_eq(pinv[(1, 0)], 0.0, 1e-10));
    }

    #[test]
    fn test_svd_orthogonality() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 10.0]]);

        let svd = Svd::compute(a.as_ref()).unwrap();
        let u = svd.u();
        let vt = svd.vt();

        // Verify U^T * U = I
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += u[(k, i)] * u[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(sum, expected, 1e-9));
            }
        }

        // Verify V^T * V = I
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += vt[(i, k)] * vt[(j, k)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(sum, expected, 1e-9));
            }
        }
    }

    #[test]
    fn test_svd_f32() {
        let a = Mat::from_rows(&[&[1.0f32, 2.0], &[3.0, 4.0]]);

        let svd = Svd::compute(a.as_ref()).unwrap();
        let reconstructed = svd.reconstruct();

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (reconstructed[(i, j)] - a[(i, j)]).abs() < 1e-4,
                    "reconstructed[{},{}] = {}, a = {}",
                    i,
                    j,
                    reconstructed[(i, j)],
                    a[(i, j)]
                );
            }
        }
    }
}
