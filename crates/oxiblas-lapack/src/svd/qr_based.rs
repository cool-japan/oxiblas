//! QR-based SVD algorithm (Golub-Kahan-Reinsch).
//!
//! This implementation uses:
//! 1. Bidiagonal reduction: A = Q * B * P^T
//! 2. Implicit QR iteration on the bidiagonal matrix to compute singular values
//!
//! This is the classical LAPACK approach (DGESVD).

use crate::svd::bidiag_reduce::{BidiagVect, gebrd, orgbr};
use oxiblas_core::scalar::{Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Error type for QR-based SVD computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QrSvdError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Algorithm did not converge.
    NotConverged {
        /// Number of singular values that did not converge.
        num_unconverged: usize,
    },
    /// Internal error during bidiagonal reduction.
    BidiagError,
}

impl core::fmt::Display for QrSvdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::NotConverged { num_unconverged } => {
                write!(
                    f,
                    "SVD did not converge: {} singular values unconverged",
                    num_unconverged
                )
            }
            Self::BidiagError => write!(f, "Bidiagonal reduction failed"),
        }
    }
}

impl std::error::Error for QrSvdError {}

/// QR-based SVD using Golub-Kahan-Reinsch algorithm.
///
/// Computes the economy SVD: A = U·Σ·V^T where:
/// - U is m×k (first k left singular vectors)
/// - Σ is k×k diagonal (singular values, sorted descending)
/// - V^T is k×n (first k right singular vectors transposed)
/// - k = min(m, n)
///
/// This approach is numerically stable and efficient for medium-sized matrices.
#[derive(Debug, Clone)]
pub struct QrSvd<T: Scalar> {
    /// Left singular vectors (m×k matrix where k = min(m, n)).
    u: Mat<T>,
    /// Singular values (sorted in descending order).
    sigma: Vec<T>,
    /// Right singular vectors transposed (k×n matrix where k = min(m, n)).
    vt: Mat<T>,
    /// Original matrix dimensions.
    m: usize,
    n: usize,
}

impl<T: Field + Real + bytemuck::Zeroable> QrSvd<T> {
    /// Maximum iterations per singular value for bidiagonal QR.
    const MAX_BIDIAG_ITER: usize = 30;

    /// Computes the economy SVD of matrix A using QR-based algorithm.
    ///
    /// Returns U (m×k), Σ (k values), V^T (k×n) where k = min(m, n).
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::svd::QrSvd;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[3.0f64, 0.0],
    ///     &[0.0, 4.0],
    /// ]);
    ///
    /// let svd = QrSvd::compute(a.as_ref()).unwrap();
    /// let sigma = svd.singular_values();
    ///
    /// assert!((sigma[0] - 4.0).abs() < 1e-10);
    /// assert!((sigma[1] - 3.0).abs() < 1e-10);
    /// ```
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, QrSvdError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(QrSvdError::EmptyMatrix);
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

        let k = m.min(n);

        // Step 1: Bidiagonal reduction A = Q * B * P^T
        let factors = gebrd(a).map_err(|_| QrSvdError::BidiagError)?;

        // Extract bidiagonal elements
        let d = factors.d.clone();
        let e = factors.e.clone();

        // Step 2: Compute SVD of the bidiagonal matrix using QR iteration
        let (u_bidiag, sigma, vt_bidiag) = Self::bidiagonal_svd_qr(&d, &e)?;

        // Step 3: Generate Q and P explicitly
        let q = orgbr(&factors, BidiagVect::Q).map_err(|_| QrSvdError::BidiagError)?;
        let p = orgbr(&factors, BidiagVect::P).map_err(|_| QrSvdError::BidiagError)?;

        // Step 4: Combine: U = Q * U_bidiag, V^T = V^T_bidiag * P^T
        // Q: m×k (for tall) or m×m (for wide)
        // U_bidiag: k×k
        // P: n×n (for tall) or k×n (for wide)
        // V^T_bidiag: k×k

        let q_rows = q.nrows();
        let q_cols = q.ncols();
        let p_rows = p.nrows();
        let p_cols = p.ncols();

        // U = Q * U_bidiag
        let mut u = Mat::zeros(q_rows, k);
        for i in 0..q_rows {
            for j in 0..k {
                let mut sum = T::zero();
                for l in 0..k.min(q_cols) {
                    sum = sum + q[(i, l)] * u_bidiag[(l, j)];
                }
                u[(i, j)] = sum;
            }
        }

        // V^T = V^T_bidiag * P^T = (P * V_bidiag)^T
        // First compute P * V_bidiag, then transpose
        let mut vt = Mat::zeros(k, p_cols);
        for i in 0..k {
            for j in 0..p_cols {
                let mut sum = T::zero();
                for l in 0..k.min(p_rows) {
                    // V^T_bidiag[i, l] * P[l, j]^T = V^T_bidiag[i, l] * P^T[j, l]
                    // But P is stored as P, so we want: vt_bidiag[i, l] * p[j, l] (transposed access)
                    // Actually: V^T = (P * V_bidiag)^T but if P is stored in column-major,
                    // we need to be careful.
                    // Simpler: vt[i, j] = sum_l vt_bidiag[i, l] * p^T[l, j] = sum_l vt_bidiag[i, l] * p[j, l]
                    if j < p_rows {
                        sum = sum + vt_bidiag[(i, l)] * p[(j, l)];
                    }
                }
                vt[(i, j)] = sum;
            }
        }

        Ok(Self { u, sigma, vt, m, n })
    }

    /// Compute SVD of a bidiagonal matrix using implicit QR iteration.
    /// Returns (U, sigma, V^T) where U and V^T are orthogonal and sigma are singular values.
    fn bidiagonal_svd_qr(d: &[T], e: &[T]) -> Result<(Mat<T>, Vec<T>, Mat<T>), QrSvdError> {
        let n = d.len();
        if n == 0 {
            return Ok((Mat::zeros(0, 0), vec![], Mat::zeros(0, 0)));
        }

        let mut d_work: Vec<T> = d.to_vec();
        let mut e_work: Vec<T> = e.to_vec();

        // Initialize U and V^T as identity
        let mut u = Mat::zeros(n, n);
        let mut vt = Mat::zeros(n, n);
        for i in 0..n {
            u[(i, i)] = T::one();
            vt[(i, i)] = T::one();
        }

        let eps = <T as Scalar>::epsilon();
        let tol = eps * T::from_f64(100.0).unwrap_or(T::one());

        // Implicit QR iteration (Golub-Kahan SVD step)
        for _iter in 0..Self::MAX_BIDIAG_ITER * n {
            // Check for convergence
            let mut converged = true;
            for i in 0..e_work.len() {
                if Scalar::abs(e_work[i])
                    > tol * (Scalar::abs(d_work[i]) + Scalar::abs(d_work[i + 1]))
                {
                    converged = false;
                    break;
                }
            }
            if converged {
                break;
            }

            // Find the largest unreduced block from the bottom
            let mut p = e_work.len();
            while p > 0
                && Scalar::abs(e_work[p - 1])
                    <= tol * (Scalar::abs(d_work[p - 1]) + Scalar::abs(d_work[p]))
            {
                p -= 1;
            }

            if p == 0 {
                break;
            }

            // Apply Golub-Kahan SVD step to the unreduced block [0..p+1]
            Self::golub_kahan_step(&mut d_work, &mut e_work, &mut u, &mut vt, 0, p + 1);
        }

        // Make all diagonal elements positive
        for i in 0..n {
            if d_work[i] < T::zero() {
                d_work[i] = -d_work[i];
                for j in 0..n {
                    u[(j, i)] = -u[(j, i)];
                }
            }
        }

        // Sort singular values in descending order
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            if d_work[b] > d_work[a] {
                core::cmp::Ordering::Greater
            } else if d_work[b] < d_work[a] {
                core::cmp::Ordering::Less
            } else {
                core::cmp::Ordering::Equal
            }
        });

        let mut sigma = vec![T::zero(); n];
        let mut u_sorted = Mat::zeros(n, n);
        let mut vt_sorted = Mat::zeros(n, n);

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            sigma[new_idx] = d_work[old_idx];
            for j in 0..n {
                u_sorted[(j, new_idx)] = u[(j, old_idx)];
                vt_sorted[(new_idx, j)] = vt[(old_idx, j)];
            }
        }

        Ok((u_sorted, sigma, vt_sorted))
    }

    /// Golub-Kahan SVD step (implicit zero-shift QR).
    fn golub_kahan_step(
        d: &mut [T],
        e: &mut [T],
        u: &mut Mat<T>,
        vt: &mut Mat<T>,
        start: usize,
        end: usize,
    ) {
        let n = u.nrows();

        // Initial rotation using Wilkinson shift on B^T * B
        let last = end - 1;

        // Initial values for bulge chase
        let mut f = d[start] * d[start];
        let mut g = d[start] * e[start];

        for k in start..last {
            // Compute Givens rotation to zero g
            let (c, s, r) = givens_rotation(f, g);

            if k > start {
                e[k - 1] = r;
            }

            f = c * d[k] + s * e[k];
            e[k] = -s * d[k] + c * e[k];
            g = s * d[k + 1];
            d[k + 1] = c * d[k + 1];

            // Accumulate V^T rotation
            for j in 0..n {
                let vk = vt[(k, j)];
                let vk1 = vt[(k + 1, j)];
                vt[(k, j)] = c * vk + s * vk1;
                vt[(k + 1, j)] = -s * vk + c * vk1;
            }

            // Compute Givens rotation to zero g
            let (c, s, r) = givens_rotation(f, g);
            d[k] = r;
            f = c * e[k] + s * d[k + 1];
            d[k + 1] = -s * e[k] + c * d[k + 1];

            if k < last - 1 {
                g = s * e[k + 1];
                e[k + 1] = c * e[k + 1];
            }

            // Accumulate U rotation
            for j in 0..n {
                let uk = u[(j, k)];
                let uk1 = u[(j, k + 1)];
                u[(j, k)] = c * uk + s * uk1;
                u[(j, k + 1)] = -s * uk + c * uk1;
            }
        }

        e[last - 1] = f;
    }

    /// Returns the left singular vectors U (m×k matrix).
    pub fn u(&self) -> &Mat<T> {
        &self.u
    }

    /// Returns the singular values in descending order.
    pub fn singular_values(&self) -> &[T] {
        &self.sigma
    }

    /// Returns the right singular vectors as V^T (k×n matrix).
    pub fn vt(&self) -> &Mat<T> {
        &self.vt
    }

    /// Returns the original matrix dimensions (m, n).
    pub fn dims(&self) -> (usize, usize) {
        (self.m, self.n)
    }

    /// Reconstructs the original matrix A = U·Σ·V^T.
    pub fn reconstruct(&self) -> Mat<T> {
        let mut result = Mat::zeros(self.m, self.n);
        let k = self.sigma.len();

        let u_cols = self.u.ncols();
        let vt_rows = self.vt.nrows();

        for i in 0..self.m {
            for j in 0..self.n {
                let mut sum = T::zero();
                for l in 0..k.min(u_cols).min(vt_rows) {
                    sum = sum + self.u[(i, l)] * self.sigma[l] * self.vt[(l, j)];
                }
                result[(i, j)] = sum;
            }
        }

        result
    }

    /// Returns the condition number (ratio of largest to smallest singular value).
    pub fn condition_number(&self) -> T {
        if self.sigma.is_empty() {
            return T::one();
        }

        let max_sv = self.sigma[0];
        let min_sv = *self.sigma.last().expect("collection should be non-empty");

        if min_sv > T::zero() {
            max_sv / min_sv
        } else {
            <T as Scalar>::max_value()
        }
    }

    /// Returns the numerical rank with given tolerance.
    pub fn rank(&self, tol: T) -> usize {
        self.sigma.iter().filter(|&&s| s > tol).count()
    }
}

/// Compute Givens rotation parameters.
/// Returns (c, s, r) such that [c s; -s c] * [f; g] = [r; 0]
fn givens_rotation<T: Field + Real>(f: T, g: T) -> (T, T, T) {
    let eps = <T as Scalar>::epsilon();

    if Scalar::abs(g) < eps {
        (T::one(), T::zero(), f)
    } else if Scalar::abs(f) < eps {
        (
            T::zero(),
            if g >= T::zero() { T::one() } else { -T::one() },
            Scalar::abs(g),
        )
    } else {
        let h = Real::sqrt(f * f + g * g);
        let c = Scalar::abs(f) / h;
        let s = g / h * (if f >= T::zero() { T::one() } else { -T::one() });
        let r = if f >= T::zero() { h } else { -h };
        (c, s, r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_qr_svd_diagonal() {
        let a = Mat::from_rows(&[&[3.0f64, 0.0], &[0.0, 4.0]]);

        let svd = QrSvd::compute(a.as_ref()).unwrap();
        let sigma = svd.singular_values();

        // Singular values should be 4 and 3 (sorted descending)
        assert!(approx_eq(sigma[0], 4.0, 1e-8), "sigma[0] = {}", sigma[0]);
        assert!(approx_eq(sigma[1], 3.0, 1e-8), "sigma[1] = {}", sigma[1]);
    }

    #[test]
    fn test_qr_svd_identity() {
        let a: Mat<f64> = Mat::eye(3);

        let svd = QrSvd::compute(a.as_ref()).unwrap();
        let sigma = svd.singular_values();

        for (i, &s) in sigma.iter().enumerate() {
            assert!(approx_eq(s, 1.0, 1e-8), "sigma[{}] = {}", i, s);
        }
    }

    #[test]
    fn test_qr_svd_reconstruction() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let svd = QrSvd::compute(a.as_ref()).unwrap();
        let reconstructed = svd.reconstruct();

        for i in 0..3 {
            for j in 0..2 {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-8),
                    "Mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    reconstructed[(i, j)],
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_qr_svd_singular_values_descending() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let svd = QrSvd::compute(a.as_ref()).unwrap();
        let sigma = svd.singular_values();

        for i in 0..sigma.len() - 1 {
            assert!(
                sigma[i] >= sigma[i + 1],
                "Singular values not descending: sigma[{}]={} < sigma[{}]={}",
                i,
                sigma[i],
                i + 1,
                sigma[i + 1]
            );
        }
    }

    #[test]
    fn test_qr_svd_u_orthogonal() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let svd = QrSvd::compute(a.as_ref()).unwrap();
        let u = svd.u();

        // Check U^T * U = I (for thin SVD, this should be k×k identity)
        let k = u.ncols();
        for i in 0..k {
            for j in 0..k {
                let mut dot = 0.0;
                for l in 0..u.nrows() {
                    dot += u[(l, i)] * u[(l, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(dot, expected, 1e-8),
                    "U^T*U not identity at ({}, {}): {} vs {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_qr_svd_vt_orthogonal() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let svd = QrSvd::compute(a.as_ref()).unwrap();
        let vt = svd.vt();

        // Check V^T * V = I (for thin SVD, this should be k×k identity)
        let k = vt.nrows();
        for i in 0..k {
            for j in 0..k {
                let mut dot = 0.0;
                for l in 0..vt.ncols() {
                    dot += vt[(i, l)] * vt[(j, l)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(dot, expected, 1e-8),
                    "V^T*V not identity at ({}, {}): {} vs {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_qr_svd_condition_number() {
        // Well-conditioned matrix
        let a = Mat::from_rows(&[&[2.0f64, 0.0], &[0.0, 1.0]]);

        let svd = QrSvd::compute(a.as_ref()).unwrap();
        let cond = svd.condition_number();

        assert!(approx_eq(cond, 2.0, 1e-8), "cond = {}", cond);
    }

    #[test]
    fn test_qr_svd_rank() {
        // Rank-1 matrix
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[2.0, 4.0, 6.0]]);

        let svd = QrSvd::compute(a.as_ref()).unwrap();
        let r = svd.rank(1e-8);

        assert_eq!(r, 1, "rank = {}", r);
    }

    #[test]
    fn test_qr_svd_1x1() {
        let a = Mat::from_rows(&[&[-5.0f64]]);

        let svd = QrSvd::compute(a.as_ref()).unwrap();
        let sigma = svd.singular_values();

        assert!(approx_eq(sigma[0], 5.0, 1e-10));
    }

    #[test]
    fn test_qr_svd_wide_matrix() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0, 4.0]]);

        let svd = QrSvd::compute(a.as_ref()).unwrap();
        let sigma = svd.singular_values();

        // sqrt(1 + 4 + 9 + 16) = sqrt(30)
        assert!(
            approx_eq(sigma[0], 30.0f64.sqrt(), 1e-8),
            "sigma[0] = {}",
            sigma[0]
        );
    }

    #[test]
    fn test_qr_svd_square_matrix() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let svd = QrSvd::compute(a.as_ref()).unwrap();
        let reconstructed = svd.reconstruct();

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-8),
                    "Mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    reconstructed[(i, j)],
                    a[(i, j)]
                );
            }
        }
    }
}
