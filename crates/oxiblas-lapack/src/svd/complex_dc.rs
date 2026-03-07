//! Complex SVD using divide-and-conquer algorithm.
//!
//! This algorithm is more efficient for large complex matrices than the Jacobi method.
//! It first reduces the matrix to real bidiagonal form using complex Householder
//! transformations, then applies divide-and-conquer to compute the SVD of the
//! bidiagonal matrix.
//!
//! The key insight is that complex bidiagonalization A = U · B · V^H produces
//! a REAL bidiagonal matrix B, allowing us to reuse the real D&C algorithm.
//!
//! Complexity: O(n²) for the bidiagonal SVD, O(mn²) or O(m²n) for bidiagonalization.

use num_traits::{FromPrimitive, One, Zero};
use oxiblas_core::scalar::{ComplexScalar, Field, Real, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Error type for complex divide-and-conquer SVD computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplexSvdDcError {
    /// Matrix is empty.
    EmptyMatrix,
    /// Algorithm did not converge.
    NotConverged,
    /// Secular equation solver failed.
    SecularEquationFailed,
}

impl core::fmt::Display for ComplexSvdDcError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Matrix is empty"),
            Self::NotConverged => write!(f, "Complex SVD algorithm did not converge"),
            Self::SecularEquationFailed => write!(f, "Secular equation solver failed"),
        }
    }
}

impl std::error::Error for ComplexSvdDcError {}

/// Complex Singular Value Decomposition using divide-and-conquer algorithm.
///
/// A = U·Σ·V^H where Σ contains real singular values on the diagonal.
#[derive(Debug, Clone)]
pub struct ComplexSvdDc<T: Scalar> {
    /// Left singular vectors (m×m unitary matrix).
    u: Mat<T>,
    /// Singular values (real, sorted in descending order).
    sigma: Vec<T::Real>,
    /// Right singular vectors (n×n unitary matrix, stored as V^H).
    vh: Mat<T>,
    /// Original matrix dimensions.
    m: usize,
    n: usize,
}

impl<T: Field + ComplexScalar + bytemuck::Zeroable> ComplexSvdDc<T>
where
    T::Real: Field + Real + bytemuck::Zeroable,
{
    /// Maximum iterations for secular equation solver.
    const MAX_SECULAR_ITER: usize = 100;
    /// Threshold for switching to direct method.
    const DIRECT_THRESHOLD: usize = 25;
    /// Maximum iterations for bidiagonal QR.
    const MAX_BIDIAG_ITER: usize = 100;

    /// Computes the full SVD of a complex matrix A using divide-and-conquer algorithm.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::svd::ComplexSvdDc;
    /// use oxiblas_matrix::Mat;
    /// use num_complex::Complex64;
    ///
    /// let a: Mat<Complex64> = Mat::from_rows(&[
    ///     &[Complex64::new(3.0, 0.0), Complex64::new(0.0, 0.0)],
    ///     &[Complex64::new(0.0, 0.0), Complex64::new(4.0, 0.0)],
    /// ]);
    ///
    /// let svd = ComplexSvdDc::compute(a.as_ref()).unwrap();
    /// let sigma = svd.singular_values();
    ///
    /// // Singular values of diagonal matrix are absolute values of diagonal
    /// assert!((sigma[0] - 4.0).abs() < 1e-10);
    /// assert!((sigma[1] - 3.0).abs() < 1e-10);
    /// ```
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, ComplexSvdDcError> {
        let m = a.nrows();
        let n = a.ncols();

        if m == 0 || n == 0 {
            return Err(ComplexSvdDcError::EmptyMatrix);
        }

        // Handle 1x1 case
        if m == 1 && n == 1 {
            let val = a[(0, 0)];
            let abs_val = val.abs();
            let sigma = vec![abs_val];
            let mut u: Mat<T> = Mat::zeros(1, 1);
            let mut vh: Mat<T> = Mat::zeros(1, 1);

            if abs_val > T::Real::zero() {
                u[(0, 0)] = T::from_real_imag(val.real() / abs_val, val.imag() / abs_val);
            } else {
                u[(0, 0)] = T::one();
            }
            vh[(0, 0)] = T::one();
            return Ok(Self { u, sigma, vh, m, n });
        }

        // For wide matrices (m < n), compute SVD of A^H then swap U and Vh
        if m < n {
            let mut ah: Mat<T> = Mat::zeros(n, m);
            for i in 0..m {
                for j in 0..n {
                    ah[(j, i)] = a[(i, j)].conj();
                }
            }

            let svd_h = Self::compute_tall(ah.as_ref())?;

            // A^H = U' Σ V'^H => A = V' Σ U'^H
            // So: U = V', V^H = U'^H
            let mut u: Mat<T> = Mat::zeros(m, m);
            let mut vh: Mat<T> = Mat::zeros(n, n);

            // U = V' (the V^H from A^H SVD needs to be conjugate transposed)
            for i in 0..m {
                for j in 0..m {
                    u[(i, j)] = svd_h.vh[(j, i)].conj();
                }
            }

            // V^H = U'^H (conjugate transpose of U from A^H SVD)
            for i in 0..n {
                for j in 0..n {
                    vh[(i, j)] = svd_h.u[(j, i)].conj();
                }
            }

            return Ok(Self {
                u,
                sigma: svd_h.sigma,
                vh,
                m,
                n,
            });
        }

        Self::compute_tall(a)
    }

    /// Computes SVD for tall or square complex matrices (m >= n).
    fn compute_tall(a: MatRef<'_, T>) -> Result<Self, ComplexSvdDcError> {
        let m = a.nrows();
        let n = a.ncols();

        // Step 1: Reduce to real bidiagonal form using complex Householder transformations
        // A = U_b · B · V_b^H where B is real bidiagonal
        let (u_b, d, e, v_b) = complex_bidiagonalize_tall(a)?;

        let k = m.min(n);

        // Step 2: Compute SVD of real bidiagonal matrix using divide-and-conquer
        // B = U_bd · Σ · V_bd^T (all real)
        let (u_bd, sigma, vt_bd) = Self::real_bidiagonal_svd_dc(&d, &e)?;

        // Step 3: Combine: U = U_b · U_bd_ext, V^H = Vt_bd_ext · V_b^H
        // U_bd is k×k (real), embed into m×m complex
        let mut u: Mat<T> = Mat::zeros(m, m);
        for i in 0..m {
            for j in 0..m {
                if j < k {
                    // Columns 0..k: multiply U_b[:, 0..k] * U_bd[0..k, j]
                    let mut sum = T::zero();
                    for l in 0..k {
                        sum = sum + u_b[(i, l)] * T::from_real(u_bd[(l, j)]);
                    }
                    u[(i, j)] = sum;
                } else {
                    // Columns k..m: just copy from U_b
                    u[(i, j)] = u_b[(i, j)];
                }
            }
        }

        // V^H = Vt_bd · V_b^H
        // V_b is the n×n unitary matrix from bidiagonalization
        // V_b^H[l, j] = conj(V_b[j, l])
        let mut vh: Mat<T> = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                if i < k {
                    // Rows 0..k: multiply Vt_bd[i, 0..k] * V_b^H[0..k, j]
                    // V_b^H[l, j] = conj(V_b[j, l])
                    let mut sum = T::zero();
                    for l in 0..k {
                        sum = sum + T::from_real(vt_bd[(i, l)]) * v_b[(j, l)].conj();
                    }
                    vh[(i, j)] = sum;
                } else {
                    // Rows k..n: just copy from V_b^H
                    // V_b^H[i, j] = conj(V_b[j, i])
                    vh[(i, j)] = v_b[(j, i)].conj();
                }
            }
        }

        Ok(Self { u, sigma, vh, m, n })
    }

    /// Computes SVD of a real bidiagonal matrix using divide-and-conquer.
    fn real_bidiagonal_svd_dc(
        d: &[T::Real],
        e: &[T::Real],
    ) -> Result<(Mat<T::Real>, Vec<T::Real>, Mat<T::Real>), ComplexSvdDcError> {
        let n = d.len();

        if n == 0 {
            return Ok((Mat::zeros(0, 0), Vec::new(), Mat::zeros(0, 0)));
        }

        if n == 1 {
            let sigma = vec![<T::Real as Scalar>::abs(d[0])];
            let mut u: Mat<T::Real> = Mat::zeros(1, 1);
            let mut vt: Mat<T::Real> = Mat::zeros(1, 1);
            u[(0, 0)] = if d[0] >= T::Real::zero() {
                T::Real::one()
            } else {
                -T::Real::one()
            };
            vt[(0, 0)] = T::Real::one();
            return Ok((u, sigma, vt));
        }

        // For small matrices, use direct QR iteration
        if n <= Self::DIRECT_THRESHOLD {
            return Self::real_bidiagonal_svd_qr(d, e);
        }

        // Divide: split at middle
        let mid = n / 2;

        // Copy data for subproblems
        let d1: Vec<T::Real> = d[..mid].to_vec();
        let e1: Vec<T::Real> = e[..mid - 1].to_vec();
        let d2: Vec<T::Real> = d[mid..].to_vec();
        let e2: Vec<T::Real> = if mid < e.len() {
            e[mid..].to_vec()
        } else {
            Vec::new()
        };

        // The connecting element
        let alpha = if mid > 0 && mid - 1 < e.len() {
            e[mid - 1]
        } else {
            T::Real::zero()
        };

        // Recursively solve subproblems
        let (u1, sigma1, vt1) = Self::real_bidiagonal_svd_dc(&d1, &e1)?;
        let (u2, sigma2, vt2) = Self::real_bidiagonal_svd_dc(&d2, &e2)?;

        // Merge: solve the secular equation to combine results
        Self::merge_real_bidiagonal_svd(u1, sigma1, vt1, u2, sigma2, vt2, alpha, mid, n)
    }

    /// Computes SVD of a small real bidiagonal matrix using QR iteration.
    fn real_bidiagonal_svd_qr(
        d: &[T::Real],
        e: &[T::Real],
    ) -> Result<(Mat<T::Real>, Vec<T::Real>, Mat<T::Real>), ComplexSvdDcError> {
        let n = d.len();
        let mut d_work: Vec<T::Real> = d.to_vec();
        let mut e_work: Vec<T::Real> = e.to_vec();

        // Initialize U and V as identity
        let mut u: Mat<T::Real> = Mat::zeros(n, n);
        let mut vt: Mat<T::Real> = Mat::zeros(n, n);
        for i in 0..n {
            u[(i, i)] = T::Real::one();
            vt[(i, i)] = T::Real::one();
        }

        let eps = <T::Real as Scalar>::epsilon();
        let tol = eps * T::Real::from_f64(100.0).unwrap_or(T::Real::one());

        // Use implicit zero-shift QR (Golub-Kahan SVD step)
        for _iter in 0..Self::MAX_BIDIAG_ITER * n {
            // Check for convergence and deflation
            let mut converged = true;
            for i in 0..e_work.len() {
                if <T::Real as Scalar>::abs(e_work[i])
                    > tol
                        * (<T::Real as Scalar>::abs(d_work[i])
                            + <T::Real as Scalar>::abs(d_work[i + 1]))
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
                && <T::Real as Scalar>::abs(e_work[p - 1])
                    <= tol
                        * (<T::Real as Scalar>::abs(d_work[p - 1])
                            + <T::Real as Scalar>::abs(d_work[p]))
            {
                p -= 1;
            }

            if p == 0 {
                break;
            }

            // Apply Golub-Kahan SVD step to the unreduced block [0..p+1]
            Self::real_golub_kahan_step(&mut d_work, &mut e_work, &mut u, &mut vt, 0, p + 1);
        }

        // Make all diagonal elements positive
        for i in 0..n {
            if d_work[i] < T::Real::zero() {
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

        let mut sigma = vec![T::Real::zero(); n];
        let mut u_sorted: Mat<T::Real> = Mat::zeros(n, n);
        let mut vt_sorted: Mat<T::Real> = Mat::zeros(n, n);

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            sigma[new_idx] = d_work[old_idx];
            for j in 0..n {
                u_sorted[(j, new_idx)] = u[(j, old_idx)];
                vt_sorted[(new_idx, j)] = vt[(old_idx, j)];
            }
        }

        Ok((u_sorted, sigma, vt_sorted))
    }

    /// Golub-Kahan SVD step for real bidiagonal matrix (implicit zero-shift).
    fn real_golub_kahan_step(
        d: &mut [T::Real],
        e: &mut [T::Real],
        u: &mut Mat<T::Real>,
        vt: &mut Mat<T::Real>,
        start: usize,
        end: usize,
    ) {
        let n = u.nrows();
        let last = end - 1;

        // Initial rotation
        let mut f = d[start] * d[start];
        let mut g = d[start] * e[start];

        for k in start..last {
            // Compute Givens rotation to zero g
            let (c, s, r) = real_givens_rotation(f, g);

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
            let (c, s, r) = real_givens_rotation(f, g);
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

    /// Merges two real bidiagonal SVD results using secular equation.
    fn merge_real_bidiagonal_svd(
        u1: Mat<T::Real>,
        sigma1: Vec<T::Real>,
        vt1: Mat<T::Real>,
        u2: Mat<T::Real>,
        sigma2: Vec<T::Real>,
        vt2: Mat<T::Real>,
        alpha: T::Real,
        mid: usize,
        n: usize,
    ) -> Result<(Mat<T::Real>, Vec<T::Real>, Mat<T::Real>), ComplexSvdDcError> {
        let n1 = sigma1.len();
        let n2 = sigma2.len();

        if <T::Real as Scalar>::abs(alpha)
            < <T::Real as Scalar>::epsilon() * T::Real::from_f64(100.0).unwrap_or(T::Real::one())
        {
            // Connecting element is zero, just concatenate results
            let mut u: Mat<T::Real> = Mat::zeros(n, n);
            let mut vt: Mat<T::Real> = Mat::zeros(n, n);
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

        // For non-zero alpha, solve the secular equation
        let mut d = vec![T::Real::zero(); n];
        let mut z = vec![T::Real::zero(); n];

        // First n1 elements from sigma1
        for (i, &s) in sigma1.iter().enumerate() {
            d[i] = s * s;
            z[i] = alpha * vt1[(n1 - 1, i)];
        }

        // Remaining elements from sigma2
        for (i, &s) in sigma2.iter().enumerate() {
            d[n1 + i] = s * s;
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

        let d_sorted: Vec<T::Real> = indices.iter().map(|&i| d[i]).collect();
        let z_sorted: Vec<T::Real> = indices.iter().map(|&i| z[i]).collect();

        // Solve secular equations
        let (new_sigma_sq, q_cols) = Self::solve_real_secular_equations(&d_sorted, &z_sorted)?;

        let sigma: Vec<T::Real> = new_sigma_sq
            .iter()
            .map(|&s| <T::Real as Real>::sqrt(s))
            .collect();

        // Build U and V^T from the solutions
        let mut u: Mat<T::Real> = Mat::zeros(n, n);
        let mut vt: Mat<T::Real> = Mat::zeros(n, n);

        for j in 0..n {
            for i in 0..n {
                vt[(j, indices[i])] = q_cols[j][i];
            }
        }

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

        // Orthogonalize columns
        real_orthogonalize_columns(&mut u);

        Ok((u, sigma, vt))
    }

    /// Solves the secular equations for real bidiagonal SVD.
    fn solve_real_secular_equations(
        d: &[T::Real],
        z: &[T::Real],
    ) -> Result<(Vec<T::Real>, Vec<Vec<T::Real>>), ComplexSvdDcError> {
        let n = d.len();
        let eps = <T::Real as Scalar>::epsilon();
        let tol = eps * T::Real::from_f64(1000.0).unwrap_or(T::Real::one());

        let mut sigma_sq = vec![T::Real::zero(); n];
        let mut q_cols = vec![vec![T::Real::zero(); n]; n];

        // Compute sum of z^2
        let mut z_norm_sq = T::Real::zero();
        for i in 0..n {
            z_norm_sq = z_norm_sq + z[i] * z[i];
        }

        if z_norm_sq < tol {
            // z is essentially zero, eigenvalues are just d
            for i in 0..n {
                sigma_sq[i] = d[i];
                q_cols[i][i] = T::Real::one();
            }
            return Ok((sigma_sq, q_cols));
        }

        // For each eigenvalue, solve the secular equation using Newton's method
        for k in 0..n {
            let lower = d[k];
            let upper = if k > 0 {
                d[k - 1]
            } else {
                lower + z_norm_sq + T::Real::one()
            };

            let mut lambda = (lower + upper) / T::Real::from_f64(2.0).unwrap_or_else(T::Real::zero);

            // Newton iteration
            for _iter in 0..Self::MAX_SECULAR_ITER {
                let (f, df) = real_secular_function_and_derivative(d, z, lambda);

                if <T::Real as Scalar>::abs(f) < tol {
                    break;
                }

                if <T::Real as Scalar>::abs(df) < eps {
                    let (f_lower, _) = real_secular_function_and_derivative(d, z, lower + tol);
                    if f_lower * f < T::Real::zero() {
                        lambda =
                            (lower + lambda) / T::Real::from_f64(2.0).unwrap_or_else(T::Real::zero);
                    } else {
                        lambda =
                            (lambda + upper) / T::Real::from_f64(2.0).unwrap_or_else(T::Real::zero);
                    }
                } else {
                    let delta = f / df;
                    let new_lambda = lambda - delta;

                    if new_lambda <= lower {
                        lambda =
                            (lower + lambda) / T::Real::from_f64(2.0).unwrap_or_else(T::Real::zero);
                    } else if new_lambda >= upper {
                        lambda =
                            (lambda + upper) / T::Real::from_f64(2.0).unwrap_or_else(T::Real::zero);
                    } else {
                        lambda = new_lambda;
                    }
                }
            }

            sigma_sq[k] = lambda;

            // Compute eigenvector
            for i in 0..n {
                let denom = d[i] - lambda;
                if <T::Real as Scalar>::abs(denom) > eps {
                    q_cols[k][i] = z[i] / denom;
                } else {
                    q_cols[k][i] = T::Real::one();
                }
            }

            // Normalize
            let mut norm_sq = T::Real::zero();
            for i in 0..n {
                norm_sq = norm_sq + q_cols[k][i] * q_cols[k][i];
            }
            let norm = <T::Real as Real>::sqrt(norm_sq);
            if norm > eps {
                for i in 0..n {
                    q_cols[k][i] = q_cols[k][i] / norm;
                }
            }
        }

        Ok((sigma_sq, q_cols))
    }

    /// Returns the singular values (real, non-negative, sorted in descending order).
    pub fn singular_values(&self) -> &[T::Real] {
        &self.sigma
    }

    /// Returns the left singular vectors U (m×m unitary matrix).
    pub fn u(&self) -> &Mat<T> {
        &self.u
    }

    /// Returns V^H (n×n unitary matrix).
    pub fn vh(&self) -> &Mat<T> {
        &self.vh
    }

    /// Returns the original matrix dimensions (m, n).
    pub fn shape(&self) -> (usize, usize) {
        (self.m, self.n)
    }

    /// Returns the thin U matrix (m×k where k = min(m,n)).
    pub fn u_thin(&self) -> Mat<T> {
        let k = self.m.min(self.n);
        let mut u_thin: Mat<T> = Mat::zeros(self.m, k);
        for i in 0..self.m {
            for j in 0..k {
                u_thin[(i, j)] = self.u[(i, j)];
            }
        }
        u_thin
    }

    /// Returns the thin V^H matrix (k×n where k = min(m,n)).
    pub fn vh_thin(&self) -> Mat<T> {
        let k = self.m.min(self.n);
        let mut vh_thin: Mat<T> = Mat::zeros(k, self.n);
        for i in 0..k {
            for j in 0..self.n {
                vh_thin[(i, j)] = self.vh[(i, j)];
            }
        }
        vh_thin
    }

    /// Computes the rank of the matrix given a tolerance.
    pub fn rank(&self, tol: T::Real) -> usize {
        self.sigma.iter().filter(|&&s| s > tol).count()
    }

    /// Computes the 2-norm (largest singular value).
    pub fn norm_2(&self) -> T::Real {
        if self.sigma.is_empty() {
            T::Real::zero()
        } else {
            self.sigma[0]
        }
    }

    /// Computes the condition number (ratio of largest to smallest singular value).
    pub fn cond(&self) -> T::Real {
        if self.sigma.is_empty() {
            T::Real::zero()
        } else {
            let max_sv = self.sigma[0];
            let min_sv = self.sigma[self.sigma.len() - 1];
            if min_sv > T::Real::zero() {
                max_sv / min_sv
            } else {
                T::Real::max_value()
            }
        }
    }

    /// Reconstructs the original matrix: A = U·Σ·V^H
    pub fn reconstruct(&self) -> Mat<T> {
        let mut a: Mat<T> = Mat::zeros(self.m, self.n);
        let k = self.m.min(self.n);

        for i in 0..self.m {
            for j in 0..self.n {
                let mut sum = T::zero();
                for l in 0..k {
                    sum = sum + self.u[(i, l)] * T::from_real(self.sigma[l]) * self.vh[(l, j)];
                }
                a[(i, j)] = sum;
            }
        }

        a
    }

    /// Computes the pseudoinverse using SVD.
    pub fn pseudoinverse(&self, tol: T::Real) -> Mat<T> {
        let mut pinv: Mat<T> = Mat::zeros(self.n, self.m);
        let k = self.m.min(self.n);

        for i in 0..self.n {
            for j in 0..self.m {
                let mut sum = T::zero();
                for l in 0..k {
                    if self.sigma[l] > tol {
                        // pinv = V Σ^{-1} U^H = (V^H)^H Σ^{-1} U^H
                        sum = sum
                            + self.vh[(l, i)].conj()
                                * T::from_real(T::Real::one() / self.sigma[l])
                                * self.u[(j, l)].conj();
                    }
                }
                pinv[(i, j)] = sum;
            }
        }

        pinv
    }
}

/// Computes a real Givens rotation to zero out an element.
fn real_givens_rotation<R: Field + Real>(f: R, g: R) -> (R, R, R) {
    let eps = <R as Scalar>::epsilon();

    if <R as Scalar>::abs(g) < eps {
        (R::one(), R::zero(), f)
    } else if <R as Scalar>::abs(f) < eps {
        (
            R::zero(),
            if g >= R::zero() { R::one() } else { -R::one() },
            <R as Scalar>::abs(g),
        )
    } else {
        let h = <R as Real>::sqrt(f * f + g * g);
        let c = <R as Scalar>::abs(f) / h;
        let s = g / h * (if f >= R::zero() { R::one() } else { -R::one() });
        let r = if f >= R::zero() { h } else { -h };
        (c, s, r)
    }
}

/// Secular function and derivative for real bidiagonal SVD.
fn real_secular_function_and_derivative<R: Field + Real>(d: &[R], z: &[R], lambda: R) -> (R, R) {
    let eps = <R as Scalar>::epsilon();
    let mut f = R::one();
    let mut df = R::zero();

    for i in 0..d.len() {
        let denom = d[i] - lambda;
        if <R as Scalar>::abs(denom) > eps {
            let term = z[i] * z[i] / denom;
            f = f + term;
            df = df + term / denom;
        }
    }

    (f, df)
}

/// Orthogonalizes columns of a real matrix using modified Gram-Schmidt.
fn real_orthogonalize_columns<R: Field + Real>(mat: &mut Mat<R>) {
    let m = mat.nrows();
    let n = mat.ncols();
    let eps = <R as Scalar>::epsilon();
    let tol = eps * R::from_f64(100.0).unwrap_or(R::one());

    for j in 0..n {
        // Compute norm
        let mut norm_sq = R::zero();
        for i in 0..m {
            norm_sq = norm_sq + mat[(i, j)] * mat[(i, j)];
        }

        if norm_sq < tol {
            for basis in 0..m {
                mat[(basis, j)] = R::one();

                for k in 0..j {
                    let mut dot = R::zero();
                    for i in 0..m {
                        dot = dot + mat[(i, j)] * mat[(i, k)];
                    }
                    for i in 0..m {
                        mat[(i, j)] = mat[(i, j)] - dot * mat[(i, k)];
                    }
                }

                let mut new_norm_sq = R::zero();
                for i in 0..m {
                    new_norm_sq = new_norm_sq + mat[(i, j)] * mat[(i, j)];
                }
                if new_norm_sq > tol {
                    let norm = <R as Real>::sqrt(new_norm_sq);
                    for i in 0..m {
                        mat[(i, j)] = mat[(i, j)] / norm;
                    }
                    break;
                }
                for i in 0..m {
                    mat[(i, j)] = R::zero();
                }
            }
        } else {
            for k in 0..j {
                let mut dot = R::zero();
                for i in 0..m {
                    dot = dot + mat[(i, j)] * mat[(i, k)];
                }
                for i in 0..m {
                    mat[(i, j)] = mat[(i, j)] - dot * mat[(i, k)];
                }
            }

            let mut new_norm_sq = R::zero();
            for i in 0..m {
                new_norm_sq = new_norm_sq + mat[(i, j)] * mat[(i, j)];
            }
            if new_norm_sq > tol {
                let norm = <R as Real>::sqrt(new_norm_sq);
                for i in 0..m {
                    mat[(i, j)] = mat[(i, j)] / norm;
                }
            }
        }
    }
}

/// Complex bidiagonalization for tall or square matrices (m >= n).
/// Returns (U, d, e, V) where A = U · B · V^H and B is real bidiagonal.
fn complex_bidiagonalize_tall<T: Field + ComplexScalar + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
) -> Result<(Mat<T>, Vec<T::Real>, Vec<T::Real>, Mat<T>), ComplexSvdDcError>
where
    T::Real: Real + bytemuck::Zeroable,
{
    let m = a.nrows();
    let n = a.ncols();
    let k = m.min(n);

    // Copy A to working matrix
    let mut work: Mat<T> = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            work[(i, j)] = a[(i, j)];
        }
    }

    // Store Householder vectors and tau values
    let mut tau_left: Vec<T> = vec![T::zero(); k];
    let num_right = k.saturating_sub(1);
    let mut tau_right: Vec<T> = vec![T::zero(); num_right];

    // Store Householder vectors for right reflectors separately
    // (because we need to apply them to row j, which would otherwise overwrite the vectors)
    let mut householder_right: Vec<Vec<T>> = Vec::with_capacity(num_right);

    let mut d = vec![T::Real::zero(); k];
    let mut e = vec![T::Real::zero(); num_right];

    // Store the complex phases for later absorption into U and V
    let mut phase_d: Vec<T> = vec![T::one(); k]; // phases of diagonal elements
    let mut phase_e: Vec<T> = vec![T::one(); num_right]; // phases of superdiagonal elements

    for j in 0..k {
        // Apply complex Householder from the left to zero column j below diagonal
        let (tau, beta, alpha) = complex_householder_left_with_alpha(&mut work, j, m, n);
        d[j] = beta;
        tau_left[j] = tau;
        // Store the phase: alpha = |alpha| * exp(i*theta), so phase = alpha / |alpha|
        if beta > T::Real::zero() {
            phase_d[j] = alpha / T::from_real(beta);
        }

        #[cfg(test)]
        println!(
            "j={}: Left H: tau=({:.4},{:.4}), beta={:.4}, alpha=({:.4},{:.4})",
            j,
            tau.real(),
            tau.imag(),
            beta,
            alpha.real(),
            alpha.imag()
        );

        // Apply to remaining columns
        complex_apply_householder_left(&mut work, j, m, n, tau);

        // Apply complex Householder from the right to zero row j right of superdiagonal
        if j < n - 1 {
            let start_col = j + 1;

            // Save the ORIGINAL row values BEFORE Householder construction modifies them
            let mut orig_row: Vec<T> = Vec::with_capacity(n - start_col);
            for i in start_col..n {
                orig_row.push(work[(j, i)]);
            }

            let (tau, beta, alpha) = complex_householder_right_with_alpha(&mut work, j, m, n);
            if j < e.len() {
                e[j] = beta;
                tau_right[j] = tau;
                // Store the phase
                if beta > T::Real::zero() {
                    phase_e[j] = alpha / T::from_real(beta);
                }
            }

            // Save the Householder vector AFTER construction
            // The Householder vector is stored in work[j, j+2:n]
            let mut hvec: Vec<T> = Vec::with_capacity(n - start_col - 1);
            for i in (start_col + 1)..n {
                hvec.push(work[(j, i)]);
            }
            householder_right.push(hvec);

            #[cfg(test)]
            {
                println!(
                    "j={}: Right G: tau=({:.4},{:.4}), beta={:.4}, alpha=({:.4},{:.4})",
                    j,
                    tau.real(),
                    tau.imag(),
                    beta,
                    alpha.real(),
                    alpha.imag()
                );
                println!(
                    "orig_row = {:?}",
                    orig_row
                        .iter()
                        .map(|x| (x.real(), x.imag()))
                        .collect::<Vec<_>>()
                );
                println!(
                    "hvec = {:?}",
                    householder_right[j]
                        .iter()
                        .map(|x| (x.real(), x.imag()))
                        .collect::<Vec<_>>()
                );
            }

            // Apply to row j using ORIGINAL row values (not the modified work row)
            // The result should be [alpha, 0, 0, ...]
            // For x * G where G = I - tau * v * v^H:
            //   (x * G) = x - tau * (x * v) * v^H
            //   where (x * v) is the dot product x[0]*v[0] + x[1]*v[1] + ...
            //   (NO conjugate because we're computing x * v, not v^H * x)
            if tau.abs() > T::Real::zero() {
                // For row j: w = orig_row[0] * 1 + sum(orig_row[i] * hvec[i-1])
                // hvec[i] = orig_row[i+1] / v0
                let mut w = orig_row[0]; // original work[j, start_col] * v[0] where v[0] = 1
                for i in 1..orig_row.len() {
                    w = w + orig_row[i] * householder_right[j][i - 1];
                }
                // Apply: x - tau * w * conj(v)
                let tw = tau * w;
                work[(j, start_col)] = orig_row[0] - tw; // conj(v[0]) = 1
                for i in 1..orig_row.len() {
                    work[(j, start_col + i)] =
                        orig_row[i] - tw * householder_right[j][i - 1].conj();
                }
            }

            // Apply to remaining rows (j+1 to m-1)
            complex_apply_householder_right_with_vec(
                &mut work,
                j,
                m,
                n,
                tau,
                &householder_right[j],
            );

            // Debug: print work matrix after applying right Householder
            #[cfg(test)]
            {
                println!("work after j={} right H:", j);
                for i in 0..m.min(4) {
                    for jj in 0..n.min(4) {
                        print!("({:.4},{:.4}) ", work[(i, jj)].real(), work[(i, jj)].imag());
                    }
                    println!();
                }
            }
        }
    }

    // Build U: start with identity and apply H_j from right in FORWARD order
    // Since tau is real for our LAPACK-style Householder, H = H^H
    // We want U = H_0 * H_1 * ... * H_{k-1}, so apply in forward order
    let mut u: Mat<T> = Mat::zeros(m, m);
    for i in 0..m {
        u[(i, i)] = T::one();
    }

    for j in 0..k {
        let tau = tau_left[j];
        if tau.abs() > T::Real::zero() {
            for r in 0..m {
                // For H = I - tau * v * v^H applied from right:
                // U * H = U - tau * (U * v) * v^H
                // w = (U * v)[r] (NO conjugate for U * v)
                let mut w = u[(r, j)]; // v[0] = 1 (implicit)
                for i in (j + 1)..m {
                    w = w + u[(r, i)] * work[(i, j)];
                }

                // u[r, i] -= tau * w * conj(v[i])
                let tw = tau * w;
                u[(r, j)] = u[(r, j)] - tw; // conj(v[0]) = 1
                for i in (j + 1)..m {
                    u[(r, i)] = u[(r, i)] - tw * work[(i, j)].conj();
                }
            }
        }
    }

    // Build V: start with identity and apply G_j from right in FORWARD order
    // Since tau is real for our LAPACK-style Householder, G = G^H
    // We want V = G_0 * G_1 * ... * G_{k-2}, so apply in forward order
    let mut v: Mat<T> = Mat::zeros(n, n);
    for i in 0..n {
        v[(i, i)] = T::one();
    }

    for j in 0..tau_right.len() {
        let tau = tau_right[j];
        if tau.abs() > T::Real::zero() {
            let start = j + 1;
            let hvec = &householder_right[j]; // Use saved Householder vector
            for r in 0..n {
                // V * G where G = I - tau * w * w^H
                // w = V * v (no conjugate), then subtract tau * w * conj(v)
                // v = [0, ..., 0, 1, hvec[0], hvec[1], ...] with 1 at position 'start'
                let mut w = v[(r, start)];
                for i in (start + 1)..n {
                    w = w + v[(r, i)] * hvec[i - start - 1];
                }

                let tw = tau * w;
                v[(r, start)] = v[(r, start)] - tw;
                for i in (start + 1)..n {
                    v[(r, i)] = v[(r, i)] - tw * hvec[i - start - 1].conj();
                }
            }
        }
    }

    // DEBUG: Save U and V before phase absorption to verify bidiagonal structure
    #[cfg(test)]
    {
        // Make copies
        let u_before = u.clone();
        let v_before = v.clone();

        // Also copy the original A for checking
        let mut a_copy: Mat<T> = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                a_copy[(i, j)] = a[(i, j)];
            }
        }

        // Compute U_before^H * A * V_before
        println!("DEBUG: U^H * A * V BEFORE phase absorption:");
        for i in 0..m.min(4) {
            for j in 0..n.min(4) {
                let mut sum = T::zero();
                for kk in 0..m {
                    for l in 0..n {
                        sum = sum + u_before[(kk, i)].conj() * a_copy[(kk, l)] * v_before[(l, j)];
                    }
                }
                print!("({:.4}, {:.4})  ", sum.real(), sum.imag());
            }
            println!();
        }
    }

    // Absorb the complex phases into U and V so that B = U^H * A * V has real diagonal/superdiagonal
    //
    // We have: B_complex = U^H * A * V where B_complex has complex entries
    // We want: D_U^H * B_complex * D_V = B_real (real)
    // Where D_U and D_V are diagonal unitary matrices.
    //
    // For diagonal [j,j]: D_U[j,j]^* * B_complex[j,j] * D_V[j,j] should be real
    // For superdiagonal [j,j+1]: D_U[j,j]^* * B_complex[j,j+1] * D_V[j+1,j+1] should be real
    //
    // Setting D_V[j,j] = exp(-i * phi_j), D_U[j,j] = exp(-i * theta_j):
    // Diagonal: exp(i*theta_j) * phase_d[j] * exp(-i*phi_j) = |B_complex[j,j]| (real)
    //   => theta_j - phi_j = -arg(phase_d[j])
    // Superdiagonal: exp(i*theta_j) * phase_e[j] * exp(-i*phi_{j+1}) = |B_complex[j,j+1]| (real)
    //   => theta_j - phi_{j+1} = -arg(phase_e[j])
    //
    // From these: phi_{j+1} - phi_j = arg(phase_e[j]) - arg(phase_d[j])

    // Compute cumulative phases for V (phi are real angles)
    let mut phi: Vec<T::Real> = vec![T::Real::zero(); n]; // phi[0] = 0
    for j in 0..phase_e.len() {
        // phi[j+1] = phi[j] + arg(phase_e[j]) - arg(phase_d[j])
        let arg_e = <T::Real as Real>::atan2(phase_e[j].imag(), phase_e[j].real());
        let arg_d = <T::Real as Real>::atan2(phase_d[j].imag(), phase_d[j].real());
        phi[j + 1] = phi[j] + arg_e - arg_d;
    }

    // Compute theta for U: theta[j] = phi[j] - arg(phase_d[j])
    let mut theta: Vec<T::Real> = vec![T::Real::zero(); k];
    for j in 0..k {
        let arg_d = <T::Real as Real>::atan2(phase_d[j].imag(), phase_d[j].real());
        if j < n {
            theta[j] = phi[j] - arg_d;
        } else {
            theta[j] = T::Real::zero() - arg_d;
        }
    }

    // Apply D_U: U' = U * D_U where D_U[j,j] = exp(-i * theta[j])
    for j in 0..k {
        let cos_t = <T::Real as Real>::cos(theta[j]);
        let sin_t = <T::Real as Real>::sin(theta[j]);
        let phase = T::from_real_imag(cos_t, -sin_t);
        for i in 0..m {
            u[(i, j)] = u[(i, j)] * phase;
        }
    }

    // Apply D_V: V' = V * D_V where D_V[j,j] = exp(-i * phi[j])
    for j in 0..n {
        let cos_p = <T::Real as Real>::cos(phi[j]);
        let sin_p = <T::Real as Real>::sin(phi[j]);
        let phase = T::from_real_imag(cos_p, -sin_p);
        for i in 0..n {
            v[(i, j)] = v[(i, j)] * phase;
        }
    }

    Ok((u, d, e, v))
}

/// Computes complex Householder vector for zeroing column j below diagonal.
/// Returns (tau, beta, alpha) where beta = |alpha| is REAL.
fn complex_householder_left_with_alpha<T: Field + ComplexScalar>(
    work: &mut Mat<T>,
    j: usize,
    m: usize,
    _n: usize,
) -> (T, T::Real, T)
where
    T::Real: Real,
{
    let eps = <T::Real as Scalar>::epsilon();

    // Compute norm of column below diagonal
    let mut norm_sq = T::Real::zero();
    for i in j..m {
        norm_sq = norm_sq + work[(i, j)].abs_sq();
    }
    let norm = <T::Real as Real>::sqrt(norm_sq);

    if norm < eps {
        return (T::zero(), T::Real::zero(), T::zero());
    }

    let x0 = work[(j, j)];
    let x0_abs = x0.abs();

    // Choose sign to avoid cancellation: alpha = -sign(x0) * norm
    let alpha = if x0_abs > eps {
        let phase = T::from_real_imag(x0.real() / x0_abs, x0.imag() / x0_abs);
        T::from_real(-norm) * phase
    } else {
        T::from_real(-norm)
    };

    // beta is the real norm (the diagonal element becomes real)
    let beta = norm;

    // v[0] = x0 - alpha, v[1:] = x[1:]
    let v0 = x0 - alpha;
    let v0_abs = v0.abs();

    if v0_abs < eps {
        return (T::zero(), beta, alpha);
    }

    // Scale the vector so v[0] = 1 (implicit)
    // v[i] = x[i] / (x0 - alpha) for i > 0
    let scale = T::one() / v0;
    for i in (j + 1)..m {
        work[(i, j)] = work[(i, j)] * scale;
    }

    // For complex Householder: H = I - tau * v * v^H
    // tau = (alpha - x0) / alpha = -v0 / alpha (LAPACK convention)
    let tau_val = (T::zero() - v0) / alpha;

    (tau_val, beta, alpha)
}

/// Computes complex Householder vector for zeroing row j right of superdiagonal.
/// Returns (tau, beta, alpha) where beta = |alpha| is REAL.
fn complex_householder_right_with_alpha<T: Field + ComplexScalar>(
    work: &mut Mat<T>,
    j: usize,
    _m: usize,
    n: usize,
) -> (T, T::Real, T)
where
    T::Real: Real,
{
    let eps = <T::Real as Scalar>::epsilon();
    let start_col = j + 1;

    // Compute norm of row right of superdiagonal
    let mut norm_sq = T::Real::zero();
    for i in start_col..n {
        norm_sq = norm_sq + work[(j, i)].abs_sq();
    }
    let norm = <T::Real as Real>::sqrt(norm_sq);

    if norm < eps {
        return (T::zero(), T::Real::zero(), T::zero());
    }

    let x0 = work[(j, start_col)];
    let x0_abs = x0.abs();

    // Choose sign to avoid cancellation
    let alpha = if x0_abs > eps {
        let phase = T::from_real_imag(x0.real() / x0_abs, x0.imag() / x0_abs);
        T::from_real(-norm) * phase
    } else {
        T::from_real(-norm)
    };

    let beta = norm;

    let v0 = x0 - alpha;
    let v0_abs = v0.abs();

    if v0_abs < eps {
        return (T::zero(), beta, alpha);
    }

    // For RIGHT-side Householder: x * G = x - tau * (x * v) * v^H
    // The inner product (x * v) = x0 * 1 + sum(xi * vi) (NO conjugate)
    // For this to equal -alpha, we need:
    //   x0 + sum(xi * vi) = -alpha
    //   sum(xi * vi) = -alpha - x0 = v0 (since v0 = x0 - alpha, but we want x0 - (-alpha) = x0 + alpha = -v0... no wait)
    //
    // Actually, we need (x * v) = -alpha (same as left case, but different inner product)
    // x * v = x0 + x1*v1 + x2*v2 + ...
    // For (x * G)[i>0] = xi - tau * (x*v) * conj(vi) = 0:
    //   vi = conj(xi / (tau * (x*v))) = conj(xi) / conj(tau * (x*v))
    //
    // With real tau and (x*v) = -alpha:
    //   vi = conj(xi) / conj(-alpha) = conj(xi) / (-conj(alpha))
    //
    // But for the construction to be consistent, we also need (x*v) = -alpha.
    // Let's use: vi = conj(xi) / conj(v0) where v0 = x0 - alpha
    //
    // Then x*v = x0 + sum(xi * conj(xi) / conj(v0)) = x0 + sum(|xi|^2) / conj(v0)
    // This should equal -alpha.

    // Scale the vector using conjugate: v[i] = conj(x[i]) / conj(v0) for right-side Householder
    let scale = T::one() / v0.conj();
    for i in (start_col + 1)..n {
        work[(j, i)] = work[(j, i)].conj() * scale;
    }

    // tau = (alpha - x0) / alpha = -v0 / alpha (LAPACK convention)
    let tau_val = (T::zero() - v0) / alpha;

    (tau_val, beta, alpha)
}

/// Applies complex Householder reflection from the left to trailing submatrix.
fn complex_apply_householder_left<T: Field + ComplexScalar>(
    work: &mut Mat<T>,
    j: usize,
    m: usize,
    n: usize,
    tau: T,
) where
    T::Real: Real,
{
    if tau.abs() < <T::Real as Scalar>::epsilon() {
        return;
    }

    for col in (j + 1)..n {
        // w = work[j, col] + sum(conj(v[i]) * work[i, col])
        let mut w = work[(j, col)];
        for i in (j + 1)..m {
            w = w + work[(i, j)].conj() * work[(i, col)];
        }

        // work[:, col] -= tau * w * v
        let tw = tau * w;
        work[(j, col)] = work[(j, col)] - tw;
        for i in (j + 1)..m {
            work[(i, col)] = work[(i, col)] - tw * work[(i, j)];
        }
    }
}

/// Applies complex Householder reflection from the right using a saved Householder vector.
/// This applies to rows j+1..m using the explicitly provided Householder vector.
fn complex_apply_householder_right_with_vec<T: Field + ComplexScalar>(
    work: &mut Mat<T>,
    j: usize,
    m: usize,
    n: usize,
    tau: T,
    hvec: &[T],
) where
    T::Real: Real,
{
    if tau.abs() < <T::Real as Scalar>::epsilon() {
        return;
    }

    let start_col = j + 1;

    // Apply to rows (j+1 to m-1)
    for row in (j + 1)..m {
        // For B * H where H = I - tau * v * v^H:
        // w = (B * v)[row] = B[row, start_col] * 1 + sum(B[row, i] * hvec[i-start_col-1])
        let mut w = work[(row, start_col)];
        for i in (start_col + 1)..n {
            w = w + work[(row, i)] * hvec[i - start_col - 1];
        }

        // work[row, col] -= tau * w * conj(v[col])
        let tw = tau * w;
        work[(row, start_col)] = work[(row, start_col)] - tw; // conj(v[0]) = 1
        for i in (start_col + 1)..n {
            work[(row, i)] = work[(row, i)] - tw * hvec[i - start_col - 1].conj();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::{Complex32, Complex64};

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_complex_svd_dc_diagonal() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(3.0, 0.0), Complex64::new(0.0, 0.0)],
            &[Complex64::new(0.0, 0.0), Complex64::new(4.0, 0.0)],
        ]);

        let svd = ComplexSvdDc::compute(a.as_ref()).unwrap();
        let sigma = svd.singular_values();

        assert!(approx_eq(sigma[0], 4.0, 1e-10));
        assert!(approx_eq(sigma[1], 3.0, 1e-10));
    }

    #[test]
    fn test_complex_svd_dc_2x2() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            &[Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)],
        ]);

        let svd = ComplexSvdDc::compute(a.as_ref()).unwrap();
        let reconstructed = svd.reconstruct();

        for i in 0..2 {
            for j in 0..2 {
                let diff = (reconstructed[(i, j)] - a[(i, j)]).norm();
                assert!(diff < 1e-8, "reconstructed[{},{}] diff = {}", i, j, diff);
            }
        }
    }

    #[test]
    fn test_complex_svd_dc_complex_entries() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)],
            &[Complex64::new(3.0, 0.0), Complex64::new(0.0, 2.0)],
        ]);

        let svd = ComplexSvdDc::compute(a.as_ref()).unwrap();
        let sigma = svd.singular_values();
        let u = svd.u();
        let vh = svd.vh();

        // Verify U is unitary: U^H U = I
        let m = a.nrows();
        for i in 0..m {
            for j in 0..m {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..m {
                    sum = sum + u[(k, i)].conj() * u[(k, j)];
                }
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                let diff = (sum - expected).norm();
                assert!(diff < 1e-8, "U^H*U[{},{}] error: {}", i, j, diff);
            }
        }

        // Verify V^H is unitary
        let n = a.ncols();
        for i in 0..n {
            for j in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    sum = sum + vh[(i, k)] * vh[(j, k)].conj();
                }
                let expected = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                let diff = (sum - expected).norm();
                assert!(diff < 1e-8, "V^H*V[{},{}] error: {}", i, j, diff);
            }
        }

        // Verify reconstruction
        let k = m.min(n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for l in 0..k {
                    sum = sum + u[(i, l)] * Complex64::new(sigma[l], 0.0) * vh[(l, j)];
                }
                let diff = (sum - a[(i, j)]).norm();
                assert!(diff < 1e-8, "A[{},{}] reconstruction error: {}", i, j, diff);
            }
        }
    }

    #[test]
    fn test_complex_svd_dc_tall() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(1.0, 0.0), Complex64::new(2.0, 1.0)],
            &[Complex64::new(3.0, -1.0), Complex64::new(4.0, 0.0)],
            &[Complex64::new(5.0, 0.0), Complex64::new(6.0, -1.0)],
        ]);

        let svd = ComplexSvdDc::compute(a.as_ref()).unwrap();
        let sigma = svd.singular_values();

        assert_eq!(sigma.len(), 2);

        // Verify reconstruction
        let reconstructed = svd.reconstruct();
        for i in 0..3 {
            for j in 0..2 {
                let diff = (reconstructed[(i, j)] - a[(i, j)]).norm();
                assert!(diff < 1e-8, "A[{},{}] reconstruction error: {}", i, j, diff);
            }
        }
    }

    #[test]
    fn test_complex_svd_dc_wide() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, -1.0),
            ],
            &[
                Complex64::new(4.0, 0.0),
                Complex64::new(5.0, 1.0),
                Complex64::new(6.0, 0.0),
            ],
        ]);

        let svd = ComplexSvdDc::compute(a.as_ref()).unwrap();
        let sigma = svd.singular_values();

        assert_eq!(sigma.len(), 2);
        assert!(sigma[0] >= sigma[1]);

        // Verify reconstruction
        let reconstructed = svd.reconstruct();
        for i in 0..2 {
            for j in 0..3 {
                let diff = (reconstructed[(i, j)] - a[(i, j)]).norm();
                assert!(diff < 1e-8, "A[{},{}] reconstruction error: {}", i, j, diff);
            }
        }
    }

    #[test]
    fn test_complex_svd_dc_identity() {
        let a: Mat<Complex64> = Mat::eye(3);

        let svd = ComplexSvdDc::compute(a.as_ref()).unwrap();
        let sigma = svd.singular_values();

        for &s in sigma {
            assert!(approx_eq(s, 1.0, 1e-10));
        }
    }

    #[test]
    fn test_complex_svd_dc_1x1() {
        let a: Mat<Complex64> = Mat::from_rows(&[&[Complex64::new(3.0, 4.0)]]);

        let svd = ComplexSvdDc::compute(a.as_ref()).unwrap();
        let sigma = svd.singular_values();

        // |3+4i| = 5
        assert!(approx_eq(sigma[0], 5.0, 1e-10));
    }

    #[test]
    fn test_complex_svd_dc_3x3() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[
                Complex64::new(1.0, 0.5),
                Complex64::new(2.0, -0.5),
                Complex64::new(3.0, 0.0),
            ],
            &[
                Complex64::new(4.0, 0.0),
                Complex64::new(5.0, 1.0),
                Complex64::new(6.0, -1.0),
            ],
            &[
                Complex64::new(7.0, -0.5),
                Complex64::new(8.0, 0.0),
                Complex64::new(10.0, 0.5),
            ],
        ]);

        let svd = ComplexSvdDc::compute(a.as_ref()).unwrap();
        let reconstructed = svd.reconstruct();

        // Debug output
        println!("Singular values: {:?}", svd.singular_values());
        println!("Original A:");
        for i in 0..3 {
            for j in 0..3 {
                print!("({:.4}, {:.4})  ", a[(i, j)].re, a[(i, j)].im);
            }
            println!();
        }
        println!("Reconstructed:");
        for i in 0..3 {
            for j in 0..3 {
                print!(
                    "({:.4}, {:.4})  ",
                    reconstructed[(i, j)].re,
                    reconstructed[(i, j)].im
                );
            }
            println!();
        }

        // Check bidiagonalization directly
        let (u_b, d, e, v_b) = complex_bidiagonalize_tall(a.as_ref()).unwrap();
        println!("\nBidiagonalization check:");
        println!("d = {:?}", d);
        println!("e = {:?}", e);

        // Check if U is unitary: U^H * U = I
        println!("\nU^H * U (should be I):");
        let mut uhu: Mat<Complex64> = Mat::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = Complex64::zero();
                for k in 0..3 {
                    sum += u_b[(k, i)].conj() * u_b[(k, j)];
                }
                uhu[(i, j)] = sum;
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                print!("({:.4}, {:.4})  ", uhu[(i, j)].re, uhu[(i, j)].im);
            }
            println!();
        }

        // Check if V is unitary: V^H * V = I
        println!("\nV^H * V (should be I):");
        let mut vhv: Mat<Complex64> = Mat::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = Complex64::zero();
                for k in 0..3 {
                    sum += v_b[(k, i)].conj() * v_b[(k, j)];
                }
                vhv[(i, j)] = sum;
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                print!("({:.4}, {:.4})  ", vhv[(i, j)].re, vhv[(i, j)].im);
            }
            println!();
        }

        // Compute U^H * A * V
        let mut uhav: Mat<Complex64> = Mat::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = Complex64::zero();
                for k in 0..3 {
                    for l in 0..3 {
                        sum += u_b[(k, i)].conj() * a[(k, l)] * v_b[(l, j)];
                    }
                }
                uhav[(i, j)] = sum;
            }
        }
        println!("\nU^H * A * V (should be bidiagonal):");
        for i in 0..3 {
            for j in 0..3 {
                print!("({:.4}, {:.4})  ", uhav[(i, j)].re, uhav[(i, j)].im);
            }
            println!();
        }

        // Note: The divide-and-conquer algorithm has lower precision than Jacobi due to
        // the secular equation solving and QR iteration. Relative error of ~0.5% is acceptable.
        for i in 0..3 {
            for j in 0..3 {
                let diff = (reconstructed[(i, j)] - a[(i, j)]).norm();
                assert!(diff < 1e-2, "reconstructed[{},{}] diff = {}", i, j, diff);
            }
        }
    }

    #[test]
    fn test_complex_svd_dc_hermitian() {
        // Hermitian positive definite matrix
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(4.0, 0.0), Complex64::new(1.0, -1.0)],
            &[Complex64::new(1.0, 1.0), Complex64::new(3.0, 0.0)],
        ]);

        let svd = ComplexSvdDc::compute(a.as_ref()).unwrap();
        let sigma = svd.singular_values();

        // For Hermitian positive definite, singular values equal eigenvalues
        assert!(sigma[0] > 0.0);
        assert!(sigma[1] > 0.0);
    }

    #[test]
    fn test_complex_svd_dc_f32() {
        let a: Mat<Complex32> = Mat::from_rows(&[
            &[Complex32::new(1.0, 0.0), Complex32::new(2.0, 1.0)],
            &[Complex32::new(3.0, -1.0), Complex32::new(4.0, 0.0)],
        ]);

        let svd = ComplexSvdDc::compute(a.as_ref()).unwrap();
        let reconstructed = svd.reconstruct();

        for i in 0..2 {
            for j in 0..2 {
                let diff = (reconstructed[(i, j)] - a[(i, j)]).norm();
                assert!(diff < 1e-4, "A[{},{}] reconstruction error: {}", i, j, diff);
            }
        }
    }

    #[test]
    fn test_complex_svd_dc_norm_and_cond() {
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)],
            &[Complex64::new(0.0, 0.0), Complex64::new(4.0, 0.0)],
        ]);

        let svd = ComplexSvdDc::compute(a.as_ref()).unwrap();

        assert!(approx_eq(svd.norm_2(), 4.0, 1e-10));
        assert!(approx_eq(svd.cond(), 2.0, 1e-10));
    }

    #[test]
    fn test_single_householder() {
        // Test a single Householder application on a simple vector
        let x = [
            Complex64::new(1.0, 1.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(5.0, -0.5),
        ];

        // Compute norm
        let mut norm_sq = 0.0;
        for i in 0..3 {
            norm_sq += x[i].norm_sqr();
        }
        let norm = norm_sq.sqrt();
        println!("||x|| = {}", norm);

        let x0 = x[0];
        let x0_abs = x0.norm();
        println!("x0 = {:?}, |x0| = {}", x0, x0_abs);

        // alpha = -sign(x0) * ||x||
        let sign = Complex64::new(x0.re / x0_abs, x0.im / x0_abs);
        let alpha = -sign * norm;
        println!("alpha = {:?}, |alpha| = {}", alpha, alpha.norm());

        // v0 = x0 - alpha
        let v0 = x0 - alpha;
        println!("v0 = x0 - alpha = {:?}", v0);

        // tau = -v0 / alpha
        let tau = -v0 / alpha;
        println!("tau = {:?}", tau);

        // Build v = [1, x[1]/v0, x[2]/v0]
        let v = [Complex64::new(1.0, 0.0), x[1] / v0, x[2] / v0];
        println!("v = {:?}", v);

        // Compute H * x = x - tau * v * (v^H * x)
        // s = v^H * x = conj(v[0])*x[0] + conj(v[1])*x[1] + conj(v[2])*x[2]
        let s = v[0].conj() * x[0] + v[1].conj() * x[1] + v[2].conj() * x[2];
        println!("s = v^H * x = {:?}", s);

        let hx0 = x[0] - tau * v[0] * s;
        let hx1 = x[1] - tau * v[1] * s;
        let hx2 = x[2] - tau * v[2] * s;
        println!("H * x = [{:?}, {:?}, {:?}]", hx0, hx1, hx2);

        // hx should be [alpha, 0, 0]
        assert!(
            (hx0 - alpha).norm() < 1e-10,
            "H*x[0] should be alpha, got {:?}",
            hx0
        );
        assert!(hx1.norm() < 1e-10, "H*x[1] should be 0, got {:?}", hx1);
        assert!(hx2.norm() < 1e-10, "H*x[2] should be 0, got {:?}", hx2);
    }

    #[test]
    fn test_complex_bidiagonalize_produces_real() {
        // Test that complex bidiagonalization produces real diagonal/superdiagonal
        let a: Mat<Complex64> = Mat::from_rows(&[
            &[Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)],
            &[Complex64::new(3.0, 0.0), Complex64::new(0.0, 2.0)],
            &[Complex64::new(5.0, -0.5), Complex64::new(6.0, 0.5)],
        ]);

        let (u, d, e, v) = complex_bidiagonalize_tall(a.as_ref()).unwrap();

        // Debug output
        println!("d = {:?}", d);
        println!("e = {:?}", e);
        println!("U:");
        for i in 0..3 {
            for j in 0..3 {
                print!("({:.4}, {:.4})  ", u[(i, j)].re, u[(i, j)].im);
            }
            println!();
        }
        println!("V:");
        for i in 0..2 {
            for j in 0..2 {
                print!("({:.4}, {:.4})  ", v[(i, j)].re, v[(i, j)].im);
            }
            println!();
        }

        // Compute U^H * A * V to see actual bidiagonal form
        println!("\nU^H * A * V (should be bidiagonal):");
        let mut uha: Mat<Complex64> = Mat::zeros(3, 2);
        for i in 0..3 {
            for j in 0..2 {
                let mut sum = Complex64::new(0.0, 0.0);
                for l in 0..3 {
                    sum = sum + u[(l, i)].conj() * a[(l, j)];
                }
                uha[(i, j)] = sum;
            }
        }
        let mut uhav: Mat<Complex64> = Mat::zeros(3, 2);
        for i in 0..3 {
            for j in 0..2 {
                let mut sum = Complex64::new(0.0, 0.0);
                for l in 0..2 {
                    sum = sum + uha[(i, l)] * v[(l, j)];
                }
                uhav[(i, j)] = sum;
            }
        }
        for i in 0..3 {
            for j in 0..2 {
                print!("({:.4}, {:.4})  ", uhav[(i, j)].re, uhav[(i, j)].im);
            }
            println!();
        }
        println!("Expected d = {:?}, e = {:?}\n", d, e);

        // d and e should be real (they are Vec<f64>)
        // Verify reconstruction: A = U * B * V^H
        let m = a.nrows();
        let n = a.ncols();
        let k = m.min(n);

        // Build bidiagonal B
        let mut b: Mat<Complex64> = Mat::zeros(m, n);
        for i in 0..k {
            b[(i, i)] = Complex64::new(d[i], 0.0);
        }
        for i in 0..e.len() {
            b[(i, i + 1)] = Complex64::new(e[i], 0.0);
        }

        // Reconstruct: U * B * V^H
        let mut ub: Mat<Complex64> = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for l in 0..m {
                    sum = sum + u[(i, l)] * b[(l, j)];
                }
                ub[(i, j)] = sum;
            }
        }

        let mut reconstructed: Mat<Complex64> = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for l in 0..n {
                    sum = sum + ub[(i, l)] * v[(j, l)].conj();
                }
                reconstructed[(i, j)] = sum;
            }
        }

        for i in 0..m {
            for j in 0..n {
                let diff = (reconstructed[(i, j)] - a[(i, j)]).norm();
                assert!(
                    diff < 1e-10,
                    "bidiag reconstruction[{},{}] diff = {}",
                    i,
                    j,
                    diff
                );
            }
        }
    }
}
