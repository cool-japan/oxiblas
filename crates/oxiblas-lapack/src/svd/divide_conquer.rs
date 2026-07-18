//! SVD using divide-and-conquer algorithm.
//!
//! This algorithm is more efficient for large matrices than the Jacobi method.
//! It first reduces the matrix to bidiagonal form, then applies divide-and-conquer
//! to compute the SVD of the bidiagonal matrix.
//!
//! Complexity: O(n²) for the bidiagonal SVD, O(mn²) or O(m²n) for bidiagonalization.

use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Error type for divide-and-conquer SVD computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvdDcError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Algorithm did not converge.
    NotConverged,
    /// Secular equation solver failed.
    SecularEquationFailed,
}

impl core::fmt::Display for SvdDcError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::NotConverged => write!(f, "SVD algorithm did not converge"),
            Self::SecularEquationFailed => write!(f, "Secular equation solver failed"),
        }
    }
}

impl std::error::Error for SvdDcError {}

/// Singular Value Decomposition using divide-and-conquer algorithm.
///
/// A = U·Σ·V^T where Σ contains singular values on the diagonal.
#[derive(Debug, Clone)]
pub struct SvdDc<T: Scalar> {
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

impl<T: Field + Real + bytemuck::Zeroable> SvdDc<T> {

    /// Computes the full SVD of matrix A using divide-and-conquer algorithm.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::svd::SvdDc;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[3.0f64, 0.0],
    ///     &[0.0, 4.0],
    /// ]);
    ///
    /// let svd = SvdDc::compute(a.as_ref()).unwrap();
    /// let sigma = svd.singular_values();
    ///
    /// // Singular values of diagonal matrix are absolute values of diagonal
    /// assert!((sigma[0] - 4.0).abs() < 1e-10);
    /// assert!((sigma[1] - 3.0).abs() < 1e-10);
    /// ```
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, SvdDcError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(SvdDcError::EmptyMatrix);
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

        // For wide matrices (m < n), compute SVD of transpose then swap U and Vt
        // A^T = U' * Σ * V'^T => A = V' * Σ * U'^T
        if m < n {
            let mut at = Mat::zeros(n, m);
            for i in 0..m {
                for j in 0..n {
                    at[(j, i)] = a[(i, j)];
                }
            }

            let svd_t = Self::compute_tall(at.as_ref())?;

            // Swap: U = V', Vt = U'^T
            let mut u = Mat::zeros(m, m);
            let mut vt = Mat::zeros(n, n);

            // U = (Vt')^T = V'
            for i in 0..m {
                for j in 0..m {
                    u[(i, j)] = svd_t.vt[(j, i)];
                }
            }

            // Vt = U'^T
            for i in 0..n {
                for j in 0..n {
                    vt[(i, j)] = svd_t.u[(j, i)];
                }
            }

            return Ok(Self {
                u,
                sigma: svd_t.sigma,
                vt,
                m,
                n,
            });
        }

        Self::compute_tall(a)
    }

    /// Computes SVD for tall or square matrices (m >= n).
    fn compute_tall(a: MatRef<'_, T>) -> Result<Self, SvdDcError> {
        let m = a.nrows();
        let n = a.ncols();

        // Step 1: Reduce to bidiagonal form
        // A = U_b · B · V_b^T where B is bidiagonal
        let (u_b, d, e, v_b) = bidiagonalize_tall(a)?;

        let k = m.min(n);

        // Step 2: Compute SVD of bidiagonal matrix using divide-and-conquer
        let (u_bd, sigma, vt_bd) = Self::bidiagonal_svd_dc(&d, &e)?;

        // Step 3: Combine: U = U_b · U_bd_ext, V^T = Vt_bd_ext · V_b
        // where U_bd_ext embeds the k×k U_bd into m×m (with identity in remaining diagonal)
        // and Vt_bd_ext embeds the k×k Vt_bd into n×n (with identity in remaining diagonal)

        // U = U_b * [U_bd | 0  ]
        //           [0    | I_{m-k}]
        let mut u = Mat::zeros(m, m);
        for i in 0..m {
            for j in 0..m {
                if j < k {
                    // Columns 0..k: multiply U_b[:, 0..k] * U_bd[0..k, j]
                    let mut sum = T::zero();
                    for l in 0..k {
                        sum = sum + u_b[(i, l)] * u_bd[(l, j)];
                    }
                    u[(i, j)] = sum;
                } else {
                    // Columns k..m: just copy from U_b (identity part)
                    u[(i, j)] = u_b[(i, j)];
                }
            }
        }

        // V^T = [Vt_bd | 0      ] * V_b
        //       [0     | I_{n-k}]
        // Since Vt_bd is k×k and V_b is n×n, for rows i < k we multiply by Vt_bd,
        // for rows i >= k we just take V_b[i, :]
        let mut vt = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                if i < k {
                    // Rows 0..k: multiply Vt_bd[i, 0..k] * V_b[0..k, j]
                    let mut sum = T::zero();
                    for l in 0..k {
                        sum = sum + vt_bd[(i, l)] * v_b[(l, j)];
                    }
                    vt[(i, j)] = sum;
                } else {
                    // Rows k..n: just copy from V_b (identity part)
                    vt[(i, j)] = v_b[(i, j)];
                }
            }
        }

        Ok(Self { u, sigma, vt, m, n })
    }

    /// Computes the SVD of the bidiagonal matrix with diagonal `d` and
    /// super-diagonal `e` using the shared real divide-and-conquer kernel
    /// (secular-equation merge with deflation; see [`crate::svd::bidiag_dc`]).
    fn bidiagonal_svd_dc(d: &[T], e: &[T]) -> Result<(Mat<T>, Vec<T>, Mat<T>), SvdDcError> {
        crate::svd::bidiag_dc::bidiagonal_svd_dc(d, e).map_err(|err| match err {
            crate::svd::bidiag_dc::BidiagDcError::NotConverged => SvdDcError::NotConverged,
            crate::svd::bidiag_dc::BidiagDcError::SecularEquationFailed => {
                SvdDcError::SecularEquationFailed
            }
        })
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
    pub fn pseudoinverse(&self, tol: T) -> Mat<T> {
        let mut pinv = Mat::zeros(self.n, self.m);
        let k = self.m.min(self.n);

        for i in 0..self.n {
            for j in 0..self.m {
                let mut sum = T::zero();
                for l in 0..k {
                    if self.sigma[l] > tol {
                        sum = sum + self.vt[(l, i)] * (T::one() / self.sigma[l]) * self.u[(j, l)];
                    }
                }
                pinv[(i, j)] = sum;
            }
        }

        pinv
    }
}

/// Bidiagonalize a tall or square matrix (m >= n).
fn bidiagonalize_tall<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
) -> Result<(Mat<T>, Vec<T>, Vec<T>, Mat<T>), SvdDcError> {
    let m = a.nrows();
    let n = a.ncols();
    let k = m.min(n);

    // Copy A to working matrix
    let mut work = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            work[(i, j)] = a[(i, j)];
        }
    }

    // Store Householder vectors and tau values for later U and V computation
    let mut tau_left = vec![T::zero(); k];
    let num_right = k.saturating_sub(1);
    let mut tau_right = vec![T::zero(); num_right];

    let mut d = vec![T::zero(); k];
    let mut e = vec![T::zero(); num_right];

    for j in 0..k {
        // Apply Householder from the left to zero column j below diagonal
        let (tau, beta) = householder_left(&mut work, j, m, n);
        d[j] = beta;
        tau_left[j] = tau;

        // Apply to remaining columns
        apply_householder_left_bidiag(&mut work, j, m, n, tau);

        // Apply Householder from the right to zero row j to the right of superdiagonal
        if j < n - 1 {
            let (tau, beta) = householder_right(&mut work, j, m, n);
            if j < e.len() {
                e[j] = beta;
                tau_right[j] = tau;
            }

            // Apply to remaining rows
            apply_householder_right_bidiag(&mut work, j, m, n, tau);
        }
    }

    // Build U: start with identity and apply reflections from right
    let mut u = Mat::zeros(m, m);
    for i in 0..m {
        u[(i, i)] = T::one();
    }

    for j in 0..k {
        let tau = tau_left[j];
        if tau != T::zero() {
            for r in 0..m {
                let mut w = u[(r, j)];
                for i in (j + 1)..m {
                    w = w + u[(r, i)] * work[(i, j)];
                }

                let tw = tau * w;
                u[(r, j)] = u[(r, j)] - tw;
                for i in (j + 1)..m {
                    u[(r, i)] = u[(r, i)] - tw * work[(i, j)];
                }
            }
        }
    }

    // Build V: start with identity and apply reflections from right
    let mut v = Mat::zeros(n, n);
    for i in 0..n {
        v[(i, i)] = T::one();
    }

    for j in 0..tau_right.len() {
        let tau = tau_right[j];
        if tau != T::zero() {
            let start = j + 1;
            for r in 0..n {
                let mut w = v[(r, start)];
                for i in (start + 1)..n {
                    w = w + v[(r, i)] * work[(j, i)];
                }

                let tw = tau * w;
                v[(r, start)] = v[(r, start)] - tw;
                for i in (start + 1)..n {
                    v[(r, i)] = v[(r, i)] - tw * work[(j, i)];
                }
            }
        }
    }

    // V^T
    let mut vt = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            vt[(i, j)] = v[(j, i)];
        }
    }

    Ok((u, d, e, vt))
}

/// Computes Householder vector for zeroing column j below diagonal.
fn householder_left<T: Field + Real>(work: &mut Mat<T>, j: usize, m: usize, _n: usize) -> (T, T) {
    let mut norm_sq = T::zero();
    for i in j..m {
        norm_sq = norm_sq + work[(i, j)] * work[(i, j)];
    }
    let norm = Real::sqrt(norm_sq);

    if norm == T::zero() {
        return (T::zero(), T::zero());
    }

    let x_j = work[(j, j)];
    let beta = if x_j >= T::zero() { -norm } else { norm };
    let tau = (beta - x_j) / beta;

    let scale = T::one() / (x_j - beta);
    for i in (j + 1)..m {
        work[(i, j)] = work[(i, j)] * scale;
    }

    (tau, beta)
}

/// Computes Householder vector for zeroing row j to the right of superdiagonal.
fn householder_right<T: Field + Real>(work: &mut Mat<T>, j: usize, _m: usize, n: usize) -> (T, T) {
    let start_col = j + 1;
    let mut norm_sq = T::zero();
    for i in start_col..n {
        norm_sq = norm_sq + work[(j, i)] * work[(j, i)];
    }
    let norm = Real::sqrt(norm_sq);

    if norm == T::zero() {
        return (T::zero(), T::zero());
    }

    let x_j = work[(j, start_col)];
    let beta = if x_j >= T::zero() { -norm } else { norm };
    let tau = (beta - x_j) / beta;

    let scale = T::one() / (x_j - beta);
    for i in (start_col + 1)..n {
        work[(j, i)] = work[(j, i)] * scale;
    }

    (tau, beta)
}

/// Applies Householder reflection from the left to trailing submatrix.
fn apply_householder_left_bidiag<T: Field + Real>(
    work: &mut Mat<T>,
    j: usize,
    m: usize,
    n: usize,
    tau: T,
) {
    if tau == T::zero() {
        return;
    }

    for col in (j + 1)..n {
        let mut w = work[(j, col)];
        for i in (j + 1)..m {
            w = w + work[(i, j)] * work[(i, col)];
        }

        let tw = tau * w;
        work[(j, col)] = work[(j, col)] - tw;
        for i in (j + 1)..m {
            work[(i, col)] = work[(i, col)] - tw * work[(i, j)];
        }
    }
}

/// Applies Householder reflection from the right to trailing submatrix.
fn apply_householder_right_bidiag<T: Field + Real>(
    work: &mut Mat<T>,
    j: usize,
    m: usize,
    n: usize,
    tau: T,
) {
    if tau == T::zero() {
        return;
    }

    let start_col = j + 1;
    for row in (j + 1)..m {
        let mut w = work[(row, start_col)];
        for i in (start_col + 1)..n {
            w = w + work[(j, i)] * work[(row, i)];
        }

        let tw = tau * w;
        work[(row, start_col)] = work[(row, start_col)] - tw;
        for i in (start_col + 1)..n {
            work[(row, i)] = work[(row, i)] - tw * work[(j, i)];
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
    fn test_svd_dc_diagonal() {
        let a = Mat::from_rows(&[&[3.0f64, 0.0], &[0.0, 4.0]]);

        let svd = SvdDc::compute(a.as_ref()).unwrap();
        let sigma = svd.singular_values();

        assert!(approx_eq(sigma[0], 4.0, 1e-10));
        assert!(approx_eq(sigma[1], 3.0, 1e-10));
    }

    #[test]
    fn test_svd_dc_2x2() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let svd = SvdDc::compute(a.as_ref()).unwrap();
        let reconstructed = svd.reconstruct();

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-8),
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
    fn test_svd_dc_tall() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let svd = SvdDc::compute(a.as_ref()).unwrap();
        assert_eq!(svd.singular_values().len(), 2);

        let reconstructed = svd.reconstruct();
        for i in 0..3 {
            for j in 0..2 {
                assert!(approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-8));
            }
        }
    }

    #[test]
    fn test_svd_dc_wide() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let svd = SvdDc::compute(a.as_ref()).unwrap();
        assert_eq!(svd.singular_values().len(), 2);

        let reconstructed = svd.reconstruct();
        for i in 0..2 {
            for j in 0..3 {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-8),
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
    fn test_svd_dc_identity() {
        let eye = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);

        let svd = SvdDc::compute(eye.as_ref()).unwrap();
        let sigma = svd.singular_values();

        for &s in sigma {
            assert!(approx_eq(s, 1.0, 1e-10));
        }
    }

    #[test]
    fn test_svd_dc_single() {
        let a = Mat::from_rows(&[&[5.0f64]]);

        let svd = SvdDc::compute(a.as_ref()).unwrap();
        let sigma = svd.singular_values();

        assert_eq!(sigma.len(), 1);
        assert!(approx_eq(sigma[0], 5.0, 1e-10));
    }

    #[test]
    fn test_svd_dc_norm() {
        let a = Mat::from_rows(&[&[3.0f64, 0.0], &[0.0, 4.0]]);

        let svd = SvdDc::compute(a.as_ref()).unwrap();
        assert!(approx_eq(svd.norm2(), 4.0, 1e-10));
    }

    #[test]
    fn test_svd_dc_condition() {
        let a = Mat::from_rows(&[&[2.0f64, 0.0], &[0.0, 4.0]]);

        let svd = SvdDc::compute(a.as_ref()).unwrap();
        assert!(approx_eq(svd.condition_number(), 2.0, 1e-10));
    }

    #[test]
    fn test_svd_dc_3x3() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 10.0]]);

        let svd = SvdDc::compute(a.as_ref()).unwrap();
        let reconstructed = svd.reconstruct();

        // Divide-and-conquer has slightly lower precision due to secular equation solving
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-5),
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
    fn test_bidiagonalize() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let (u, d, e, vt) = bidiagonalize_tall(a.as_ref()).unwrap();

        // Reconstruct bidiagonal matrix
        let mut b: Mat<f64> = Mat::zeros(2, 2);
        b[(0, 0)] = d[0];
        if !e.is_empty() {
            b[(0, 1)] = e[0];
        }
        b[(1, 1)] = d[1];

        // Verify U * B * V^T = A
        let mut ub: Mat<f64> = Mat::zeros(2, 2);
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    ub[(i, j)] = ub[(i, j)] + u[(i, k)] * b[(k, j)];
                }
            }
        }

        let mut ubvt: Mat<f64> = Mat::zeros(2, 2);
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    ubvt[(i, j)] = ubvt[(i, j)] + ub[(i, k)] * vt[(k, j)];
                }
            }
        }

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    approx_eq(ubvt[(i, j)], a[(i, j)], 1e-10),
                    "UBV^T[{},{}] = {}, A = {}",
                    i,
                    j,
                    ubvt[(i, j)],
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_bidiagonalize_tall() {
        // Test bidiagonalization on a tall matrix (3×2)
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let (u, d, e, vt) = bidiagonalize_tall(a.as_ref()).unwrap();

        // Reconstruct bidiagonal matrix (3×2)
        let mut b: Mat<f64> = Mat::zeros(3, 2);
        b[(0, 0)] = d[0];
        if !e.is_empty() {
            b[(0, 1)] = e[0];
        }
        b[(1, 1)] = d[1];

        // Verify U * B * V^T = A
        // First: U (3×3) * B (3×2) = 3×2
        let mut ub: Mat<f64> = Mat::zeros(3, 2);
        for i in 0..3 {
            for j in 0..2 {
                for k in 0..3 {
                    ub[(i, j)] = ub[(i, j)] + u[(i, k)] * b[(k, j)];
                }
            }
        }

        // Then: UB (3×2) * V^T (2×2) = 3×2
        let mut ubvt: Mat<f64> = Mat::zeros(3, 2);
        for i in 0..3 {
            for j in 0..2 {
                for k in 0..2 {
                    ubvt[(i, j)] = ubvt[(i, j)] + ub[(i, k)] * vt[(k, j)];
                }
            }
        }

        for i in 0..3 {
            for j in 0..2 {
                assert!(
                    approx_eq(ubvt[(i, j)], a[(i, j)], 1e-10),
                    "UBV^T[{},{}] = {}, A = {}",
                    i,
                    j,
                    ubvt[(i, j)],
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_svd_dc_f32() {
        let a = Mat::from_rows(&[&[1.0f32, 2.0], &[3.0, 4.0]]);

        let svd = SvdDc::compute(a.as_ref()).unwrap();
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

    /// Deterministic pseudo-random values in `[-1, 1)` (no external rng).
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    }

    #[test]
    fn test_svd_dc_vs_reference_across_sizes() {
        use crate::svd::{QrSvd, Svd};

        // Sizes straddle the DIRECT_THRESHOLD (25): 5, 25 use the direct shifted
        // QR path; 26, 50, 100, 200 exercise the divide-and-conquer secular
        // merge with deflation.
        for &n in &[5usize, 25, 26, 50, 100, 200] {
            let mut state: u64 = 0x2545_f491_4f6c_dd1d ^ (n as u64);
            let mut a = Mat::<f64>::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    a[(i, j)] = lcg(&mut state);
                }
            }

            let dc = SvdDc::compute(a.as_ref()).unwrap();
            // High-accuracy reference: one-sided Jacobi SVD (used for all sizes).
            let reference = Svd::compute(a.as_ref()).unwrap();

            let s_dc = dc.singular_values();
            let s_ref = reference.singular_values();
            let smax = s_dc[0].abs().max(1.0);

            // Singular values agree tightly with the Jacobi reference SVD.
            for k in 0..n {
                assert!(
                    (s_dc[k] - s_ref[k]).abs() < 1e-8 * smax,
                    "n={n}: sigma[{k}] dc={} jacobi={}",
                    s_dc[k],
                    s_ref[k]
                );
            }

            // Independent cross-check against the QR-based reference SVD
            // (Golub–Kahan–Reinsch). QrSvd's convergence/deflation logic was
            // fixed to be reliable at these sizes (direction-aware shifted
            // sweeps, O(n^2) sweep budget), so this is a hard assertion.
            if n <= 50 {
                let qr_reference = QrSvd::compute(a.as_ref()).unwrap();
                let s_qr = qr_reference.singular_values();
                for k in 0..n {
                    assert!(
                        (s_dc[k] - s_qr[k]).abs() < 5e-3 * smax,
                        "n={n}: sigma[{k}] dc={} qr={}",
                        s_dc[k],
                        s_qr[k]
                    );
                }
            }

            // UᵀU = I and VᵀV = I.
            let u = dc.u();
            let vt = dc.vt();
            for i in 0..n {
                for j in 0..n {
                    let mut utu = 0.0;
                    let mut vtv = 0.0;
                    for k in 0..n {
                        utu += u[(k, i)] * u[(k, j)];
                        vtv += vt[(i, k)] * vt[(j, k)];
                    }
                    let expect = if i == j { 1.0 } else { 0.0 };
                    assert!((utu - expect).abs() < 1e-6, "n={n}: UtU[{i},{j}]={utu}");
                    assert!((vtv - expect).abs() < 1e-9, "n={n}: VtV[{i},{j}]={vtv}");
                }
            }

            // Reconstruction ||A - U Σ Vᵀ|| is tight.
            let rec = dc.reconstruct();
            let mut rec_err = 0.0f64;
            for i in 0..n {
                for j in 0..n {
                    rec_err = rec_err.max((rec[(i, j)] - a[(i, j)]).abs());
                }
            }
            assert!(rec_err < 1e-8 * smax, "n={n}: reconstruction error {rec_err}");
        }
    }
}
