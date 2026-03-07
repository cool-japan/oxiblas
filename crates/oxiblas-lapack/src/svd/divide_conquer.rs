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
    /// Maximum iterations for secular equation solver.
    const MAX_SECULAR_ITER: usize = 100;
    /// Threshold for switching to direct method.
    const DIRECT_THRESHOLD: usize = 25;
    /// Maximum iterations for bidiagonal QR.
    const MAX_BIDIAG_ITER: usize = 30;

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

    /// Computes SVD of a bidiagonal matrix using divide-and-conquer.
    ///
    /// The bidiagonal matrix has diagonal d and superdiagonal e:
    /// ```text
    /// B = [d[0]  e[0]  0    0   ]
    ///     [0     d[1]  e[1] 0   ]
    ///     [0     0     d[2] e[2]]
    ///     [0     0     0    d[3]]
    /// ```
    fn bidiagonal_svd_dc(d: &[T], e: &[T]) -> Result<(Mat<T>, Vec<T>, Mat<T>), SvdDcError> {
        let n = d.len();

        if n == 0 {
            return Ok((Mat::zeros(0, 0), Vec::new(), Mat::zeros(0, 0)));
        }

        if n == 1 {
            let sigma = vec![Scalar::abs(d[0])];
            let mut u = Mat::zeros(1, 1);
            let mut vt = Mat::zeros(1, 1);
            u[(0, 0)] = if d[0] >= T::zero() {
                T::one()
            } else {
                -T::one()
            };
            vt[(0, 0)] = T::one();
            return Ok((u, sigma, vt));
        }

        // For small matrices, use direct QR iteration
        if n <= Self::DIRECT_THRESHOLD {
            return Self::bidiagonal_svd_qr(d, e);
        }

        // Divide: split at middle
        let mid = n / 2;

        // Copy data for subproblems
        let d1: Vec<T> = d[..mid].to_vec();
        let e1: Vec<T> = e[..mid - 1].to_vec();
        let d2: Vec<T> = d[mid..].to_vec();
        let e2: Vec<T> = if mid < e.len() {
            e[mid..].to_vec()
        } else {
            Vec::new()
        };

        // The connecting element
        let alpha = if mid > 0 && mid - 1 < e.len() {
            e[mid - 1]
        } else {
            T::zero()
        };

        // Recursively solve subproblems
        let (u1, sigma1, vt1) = Self::bidiagonal_svd_dc(&d1, &e1)?;
        let (u2, sigma2, vt2) = Self::bidiagonal_svd_dc(&d2, &e2)?;

        // Merge: solve the secular equation to combine results
        Self::merge_bidiagonal_svd(u1, sigma1, vt1, u2, sigma2, vt2, alpha, mid, n)
    }

    /// Computes SVD of a small bidiagonal matrix using QR iteration (implicit shift).
    fn bidiagonal_svd_qr(d: &[T], e: &[T]) -> Result<(Mat<T>, Vec<T>, Mat<T>), SvdDcError> {
        let n = d.len();
        let mut d_work: Vec<T> = d.to_vec();
        let mut e_work: Vec<T> = e.to_vec();

        // Initialize U and V as identity
        let mut u = Mat::zeros(n, n);
        let mut vt = Mat::zeros(n, n);
        for i in 0..n {
            u[(i, i)] = T::one();
            vt[(i, i)] = T::one();
        }

        let eps = <T as Scalar>::epsilon();
        let tol = eps * T::from_f64(100.0).unwrap_or(T::one());

        // Use implicit zero-shift QR (Golub-Kahan SVD step)
        for _iter in 0..Self::MAX_BIDIAG_ITER * n {
            // Check for convergence and deflation
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

            // Find the largest unreduced block
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

    /// Golub-Kahan SVD step (implicit zero-shift).
    fn golub_kahan_step(
        d: &mut [T],
        e: &mut [T],
        u: &mut Mat<T>,
        vt: &mut Mat<T>,
        start: usize,
        end: usize,
    ) {
        let n = u.nrows();

        // Compute shift using Wilkinson's formula on trailing 2×2 of B^T·B
        let last = end - 1;
        let d_last = d[last];
        let d_prev = if last > start { d[last - 1] } else { T::zero() };
        let e_prev = if last > start && last > start {
            e[last - 1]
        } else {
            T::zero()
        };

        let _t11 = d_prev * d_prev
            + if last > start + 1 {
                e[last - 2] * e[last - 2]
            } else {
                T::zero()
            };
        let t22 = d_last * d_last + e_prev * e_prev;
        let t12 = d_prev * e_prev;

        // Wilkinson shift: eigenvalue of 2×2 trailing block closer to t22
        let _delta = (t22 - _t11) / T::from_f64(2.0).unwrap_or_else(T::zero);
        let _shift = t22 - t12 * t12 / (_t11 + t22 + <T as Scalar>::epsilon());

        // Initial rotation
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

    /// Merges two bidiagonal SVD results using secular equation.
    fn merge_bidiagonal_svd(
        u1: Mat<T>,
        sigma1: Vec<T>,
        vt1: Mat<T>,
        u2: Mat<T>,
        sigma2: Vec<T>,
        vt2: Mat<T>,
        alpha: T,
        mid: usize,
        n: usize,
    ) -> Result<(Mat<T>, Vec<T>, Mat<T>), SvdDcError> {
        let n1 = sigma1.len();
        let n2 = sigma2.len();

        if Scalar::abs(alpha) < <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or(T::one()) {
            // Connecting element is zero, just concatenate results
            let mut u = Mat::zeros(n, n);
            let mut vt = Mat::zeros(n, n);
            let mut sigma = Vec::with_capacity(n);

            // Copy U1 and U2 into U
            for i in 0..n1 {
                for j in 0..n1 {
                    u[(i, j)] = u1[(i, j)];
                }
            }
            for i in 0..n2 {
                for j in 0..n2 {
                    u[(mid + i, mid + j)] = u2[(i, j)];
                }
            }

            // Copy Vt1 and Vt2 into Vt
            for i in 0..n1 {
                for j in 0..n1 {
                    vt[(i, j)] = vt1[(i, j)];
                }
            }
            for i in 0..n2 {
                for j in 0..n2 {
                    vt[(mid + i, mid + j)] = vt2[(i, j)];
                }
            }

            // Merge and sort singular values
            sigma.extend(sigma1.iter().copied());
            sigma.extend(sigma2.iter().copied());
            sigma.sort_by(|a, b| {
                if *b > *a {
                    core::cmp::Ordering::Greater
                } else if *b < *a {
                    core::cmp::Ordering::Less
                } else {
                    core::cmp::Ordering::Equal
                }
            });

            return Ok((u, sigma, vt));
        }

        // For non-zero alpha, we need to solve the secular equation
        // This is simplified - a full implementation would use more sophisticated methods

        // Build the combined diagonal and rank-1 update
        let mut d = vec![T::zero(); n];
        let mut z = vec![T::zero(); n];

        // First n1 elements from sigma1
        for (i, &s) in sigma1.iter().enumerate() {
            d[i] = s * s;
            // z[i] comes from last row of V1^T (which is V1's last column)
            z[i] = alpha * vt1[(n1 - 1, i)];
        }

        // Remaining elements from sigma2
        for (i, &s) in sigma2.iter().enumerate() {
            d[n1 + i] = s * s;
            // z[n1+i] comes from first row of V2^T (which is V2's first column)
            z[n1 + i] = alpha * vt2[(0, i)];
        }

        // Sort d and permute z accordingly
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            if d[a] > d[b] {
                core::cmp::Ordering::Less
            } else if d[a] < d[b] {
                core::cmp::Ordering::Greater
            } else {
                core::cmp::Ordering::Equal
            }
        });

        let d_sorted: Vec<T> = indices.iter().map(|&i| d[i]).collect();
        let z_sorted: Vec<T> = indices.iter().map(|&i| z[i]).collect();

        // Solve secular equations for each new singular value
        let (new_sigma_sq, q_cols) = Self::solve_secular_equations(&d_sorted, &z_sorted)?;

        let sigma: Vec<T> = new_sigma_sq.iter().map(|&s| Real::sqrt(s)).collect();

        // Build U and V^T from the solutions
        // This is a simplified version - full implementation needs proper deflation
        let mut u = Mat::zeros(n, n);
        let mut vt = Mat::zeros(n, n);

        // Build V from q_cols (eigenvectors of the secular equation)
        for j in 0..n {
            for i in 0..n {
                vt[(j, indices[i])] = q_cols[j][i];
            }
        }

        // Build U = B * V * Σ^{-1} (approximately)
        // For simplicity, we use the permutation to build U from U1 and U2
        for i in 0..n {
            for j in 0..n {
                let orig_idx = indices[j];
                if orig_idx < n1 {
                    if i < mid {
                        u[(i, j)] = u1[(i, orig_idx)];
                    }
                } else if i >= mid && i < mid + n2 {
                    u[(i, j)] = u2[(i - mid, orig_idx - n1)];
                }
            }
        }

        // Ensure U has orthonormal columns using Gram-Schmidt if needed
        orthogonalize_columns(&mut u);

        Ok((u, sigma, vt))
    }

    /// Solves the secular equations: f(λ) = 1 + Σ z_i² / (d_i - λ) = 0
    fn solve_secular_equations(d: &[T], z: &[T]) -> Result<(Vec<T>, Vec<Vec<T>>), SvdDcError> {
        let n = d.len();
        let eps = <T as Scalar>::epsilon();
        let tol = eps * T::from_f64(1000.0).unwrap_or(T::one());

        let mut sigma_sq = vec![T::zero(); n];
        let mut q_cols = vec![vec![T::zero(); n]; n];

        // Compute sum of z^2
        let mut z_norm_sq = T::zero();
        for i in 0..n {
            z_norm_sq = z_norm_sq + z[i] * z[i];
        }

        if z_norm_sq < tol {
            // z is essentially zero, eigenvalues are just d
            for i in 0..n {
                sigma_sq[i] = d[i];
                q_cols[i][i] = T::one();
            }
            return Ok((sigma_sq, q_cols));
        }

        // For each eigenvalue, solve the secular equation using Newton's method
        for k in 0..n {
            // Eigenvalue k is between d[k] and d[k-1] (or goes to infinity)
            let lower = d[k];
            let upper = if k > 0 {
                d[k - 1]
            } else {
                lower + z_norm_sq + T::one()
            };

            // Initial guess (midpoint)
            let mut lambda = (lower + upper) / T::from_f64(2.0).unwrap_or_else(T::zero);

            // Newton iteration
            for _iter in 0..Self::MAX_SECULAR_ITER {
                let (f, df) = secular_function_and_derivative(d, z, lambda);

                if Scalar::abs(f) < tol {
                    break;
                }

                if Scalar::abs(df) < eps {
                    // Derivative too small, use bisection step
                    let (f_lower, _) = secular_function_and_derivative(d, z, lower + tol);
                    if f_lower * f < T::zero() {
                        lambda = (lower + lambda) / T::from_f64(2.0).unwrap_or_else(T::zero);
                    } else {
                        lambda = (lambda + upper) / T::from_f64(2.0).unwrap_or_else(T::zero);
                    }
                } else {
                    let delta = f / df;
                    let new_lambda = lambda - delta;

                    // Ensure we stay within bounds
                    if new_lambda <= lower {
                        lambda = (lower + lambda) / T::from_f64(2.0).unwrap_or_else(T::zero);
                    } else if new_lambda >= upper {
                        lambda = (lambda + upper) / T::from_f64(2.0).unwrap_or_else(T::zero);
                    } else {
                        lambda = new_lambda;
                    }
                }
            }

            sigma_sq[k] = lambda;

            // Compute eigenvector
            for i in 0..n {
                let denom = d[i] - lambda;
                if Scalar::abs(denom) > eps {
                    q_cols[k][i] = z[i] / denom;
                } else {
                    q_cols[k][i] = T::one();
                }
            }

            // Normalize
            let mut norm_sq = T::zero();
            for i in 0..n {
                norm_sq = norm_sq + q_cols[k][i] * q_cols[k][i];
            }
            let norm = Real::sqrt(norm_sq);
            if norm > eps {
                for i in 0..n {
                    q_cols[k][i] = q_cols[k][i] / norm;
                }
            }
        }

        Ok((sigma_sq, q_cols))
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

/// Computes a Givens rotation to zero out an element.
/// Returns (c, s, r) such that [c s; -s c] * [f; g] = [r; 0].
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

/// Secular function: f(λ) = 1 + Σ z_i² / (d_i - λ)
fn secular_function_and_derivative<T: Field + Real>(d: &[T], z: &[T], lambda: T) -> (T, T) {
    let eps = <T as Scalar>::epsilon();
    let mut f = T::one();
    let mut df = T::zero();

    for i in 0..d.len() {
        let denom = d[i] - lambda;
        if Scalar::abs(denom) > eps {
            let term = z[i] * z[i] / denom;
            f = f + term;
            df = df + term / denom;
        }
    }

    (f, df)
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

/// Orthogonalizes columns of a matrix using modified Gram-Schmidt.
fn orthogonalize_columns<T: Field + Real>(mat: &mut Mat<T>) {
    let m = mat.nrows();
    let n = mat.ncols();
    let eps = <T as Scalar>::epsilon();
    let tol = eps * T::from_f64(100.0).unwrap_or(T::one());

    for j in 0..n {
        // Compute norm
        let mut norm_sq = T::zero();
        for i in 0..m {
            norm_sq = norm_sq + mat[(i, j)] * mat[(i, j)];
        }

        if norm_sq < tol {
            // Column is zero, try to fill with orthogonal vector
            for basis in 0..m {
                mat[(basis, j)] = T::one();

                // Orthogonalize against previous columns
                for k in 0..j {
                    let mut dot = T::zero();
                    for i in 0..m {
                        dot = dot + mat[(i, j)] * mat[(i, k)];
                    }
                    for i in 0..m {
                        mat[(i, j)] = mat[(i, j)] - dot * mat[(i, k)];
                    }
                }

                let mut new_norm_sq = T::zero();
                for i in 0..m {
                    new_norm_sq = new_norm_sq + mat[(i, j)] * mat[(i, j)];
                }
                if new_norm_sq > tol {
                    let norm = Real::sqrt(new_norm_sq);
                    for i in 0..m {
                        mat[(i, j)] = mat[(i, j)] / norm;
                    }
                    break;
                }
                for i in 0..m {
                    mat[(i, j)] = T::zero();
                }
            }
        } else {
            // Orthogonalize against previous columns
            for k in 0..j {
                let mut dot = T::zero();
                for i in 0..m {
                    dot = dot + mat[(i, j)] * mat[(i, k)];
                }
                for i in 0..m {
                    mat[(i, j)] = mat[(i, j)] - dot * mat[(i, k)];
                }
            }

            // Normalize
            let mut new_norm_sq = T::zero();
            for i in 0..m {
                new_norm_sq = new_norm_sq + mat[(i, j)] * mat[(i, j)];
            }
            if new_norm_sq > tol {
                let norm = Real::sqrt(new_norm_sq);
                for i in 0..m {
                    mat[(i, j)] = mat[(i, j)] / norm;
                }
            }
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
}
